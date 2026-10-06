"""Run a generation (answer) evaluation and print refusal, fact and citation metrics.

scripts/run_eval.py measures whether retrieval finds the right passages; this
script measures what the assistant then SAYS. Each question goes through the same
path as the app: `retrieve` with the app's BM25 loading and configured mode, then
`answer_question` (a real LLM call), then rag.verify on the sources the answer
saw. Scores are defined in rag/answer_metrics.py.

Per-question results are written as JSONL (with the answer text, so refusal
decisions and fact matches can be audited) and a summary table is printed per
corpus. `--judge` adds one LLM faithfulness-judge call per non-refused answer;
`--sleep` spaces questions out for free-tier rate limits.

Each row also records `provider` (which backend actually answered, since the
fallback chain can switch models silently) and a `diagnosis` that follows the
two-step RAG diagnostic: first, were the gold sources retrieved? then, did
generation use them? See `diagnose` for the labels. The refusal rates and the
verify ok rate are 0/1 outcomes per question, so the summary prints them with
a 95% Wilson interval (rag/stats.py); the other aggregates are means of
fractions and get none.

The summary also prints how the refusals split between the retrieval gate and
the model, right above the refusal scores, which merge the two. Without that
line a refusal F1 of 1.000 reads as proof that the cosine threshold works, when
it can equally mean the threshold never fired and the model declined every time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# Allow `python scripts/run_answer_eval.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import rag.llm  # noqa: E402
from rag.answer import answer_question  # noqa: E402
from rag.answer_metrics import (  # noqa: E402
    JUDGE_SYSTEM,
    build_judge_prompt,
    parse_judge_score,
    score_answer,
    summarize_by_corpus,
)
from rag.embed import get_embedder  # noqa: E402
from rag.lexical import BM25Index  # noqa: E402
from rag.logging_utils import never_crash_on_console_encoding  # noqa: E402
from rag.retrieve import retrieve  # noqa: E402
from rag.stats import wilson  # noqa: E402
from rag.store import VectorStore  # noqa: E402
from scripts.run_eval import load_eval  # noqa: E402

DEFAULT_CORPORA = ["ai_act_sections", "filings_sections"]
DEFAULT_OUT = config.ROOT / "logs" / "answer_eval_results.jsonl"

# Summary rows: (label, key in the summary dict, format). A tuple of two keys is
# printed as "a/b"; the gate/model split is shown next to the refusal scores,
# which merge the two, so "refusal F1" is never read as evidence about the gate.
_SUMMARY_ROWS = [
    ("questions", "n", "d"),
    ("errors (excluded)", "n_errors", "d"),
    ("answerable / unanswerable", ("n_answerable", "n_unanswerable"), None),
    ("refusals: gate / model", ("n_gate_refused", "n_model_refused"), None),
    ("refusal precision", "refusal_precision", ".3f"),
    ("refusal recall", "refusal_recall", ".3f"),
    ("refusal F1", "refusal_f1", ".3f"),
    ("false-refusal rate", "false_refusal_rate", ".3f"),
    ("key-fact recall", "key_fact_recall", ".3f"),
    ("key-fact recall (answered)", "key_fact_recall_answered", ".3f"),
    ("citation validity", "citation_validity", ".3f"),
    ("mean grounding", "mean_grounding", ".3f"),
    ("verify ok rate", "verify_ok_rate", ".3f"),
]
# Binomial summary metrics: key -> (successes, trials) read from the summary
# dict, for the Wilson interval. The confusion counts come from
# rag.answer_metrics.refusal_metrics; the verify counts are added by `with_intervals`.
_BINOMIAL = {
    "refusal_precision": lambda s: (s["tp"], s["tp"] + s["fp"]),
    "refusal_recall": lambda s: (s["tp"], s["tp"] + s["fn"]),
    "false_refusal_rate": lambda s: (s["fp"], s["fp"] + s["tn"]),
    "verify_ok_rate": lambda s: (s["verify_ok_count"], s["verify_checked"]),
}
# Diagnosis labels in the order the summary table prints them.
DIAGNOSES = ["ok", "retrieval", "generation_refusal", "generation", "generation_overanswer",
             "error"]
_COL = 24


def diagnose(row: dict) -> str:
    """Attribute one scored row to retrieval or generation (two-step RAG diagnostic).

    Answerable questions: "ok" when every key fact is in a non-refused answer;
    otherwise "retrieval" when none of the gold sources was retrieved (generation
    never saw the evidence, so its output is not judged); "generation_refusal"
    when the evidence was there but the model refused; "generation" when it
    answered from the right passages and still missed facts. Unanswerable
    questions: "ok" when refused, "generation_overanswer" otherwise. Rows with an
    `error` are "error". An answerable item without key facts (recall None)
    counts as "ok" when answered: there is nothing to grade the content against.
    """
    if row.get("error"):
        return "error"
    if not row["answerable"]:
        return "ok" if row["refused"] else "generation_overanswer"
    if not row["refused"] and row.get("key_fact_recall") in (None, 1.0):
        return "ok"
    gold, retrieved = set(row.get("gold_sources") or []), set(row.get("retrieved_sources") or [])
    if not gold & retrieved:
        return "retrieval"
    return "generation_refusal" if row["refused"] else "generation"


def diagnosis_counts(rows: list[dict]) -> dict[str, Counter]:
    """Count diagnoses per corpus, plus "all" when there are several corpora."""
    corpora = list(dict.fromkeys(r.get("corpus", "unknown") for r in rows))
    out = {c: Counter(r["diagnosis"] for r in rows if r.get("corpus", "unknown") == c)
           for c in corpora}
    if len(corpora) > 1:
        out["all"] = Counter(r["diagnosis"] for r in rows)
    return out


def with_intervals(summaries: dict[str, dict], rows: list[dict]) -> dict[str, dict]:
    """Add `ci[metric] = (low, high) | None` to each summary for the binomial metrics.

    verify_ok_rate's counts are not in the summary (it is a mean there), so they
    are recomputed from the rows of that summary's corpus ("all" = every row).
    """
    for name, summary in summaries.items():
        own = [r for r in rows if name == "all" or r.get("corpus", "unknown") == name]
        checked = [r for r in own if not r.get("error") and not r["refused"]
                   and r.get("verify_ok") is not None]
        summary["verify_checked"] = len(checked)
        summary["verify_ok_count"] = sum(1 for r in checked if r["verify_ok"])
        summary["ci"] = {key: wilson(*count(summary)) for key, count in _BINOMIAL.items()}
    return summaries


def load_corpus(corpus: str):
    """Load the vector store and optional BM25 index for a corpus, as run_eval and the app do.

    An older index without bm25.json is dense-only, exactly like the app.
    """
    store = VectorStore.load(config.index_path(corpus))
    try:
        bm25 = BM25Index.load(config.bm25_path(corpus))
    except FileNotFoundError:
        bm25 = None
    return store, bm25


def evaluate_item(item: dict, corpus: str, store, bm25, embedder, k: int, mode: str,
                  judge: bool) -> dict:
    """Answer one eval question through the app's pipeline and score it.

    An LLM failure (all providers down or rate-limited) yields an `error` row
    instead of crashing the run, so a long free-tier eval keeps its other results.
    """
    base = {"corpus": corpus, "id": item["id"], "question": item["question"],
            "answerable": bool(item["answerable"]),
            "gold_sources": list(item.get("gold_sources") or [])}
    retrieved = retrieve(item["question"], store, embedder, top_k=k, bm25=bm25, mode=mode)
    base["retrieved_sources"] = [r.chunk.source for r in retrieved]
    try:
        answer = answer_question(item["question"], retrieved)
    except rag.llm.LLMError as exc:
        row = {**base, "refused": False, "error": str(exc), "provider": None}
        row["diagnosis"] = diagnose(row)
        return row
    row = {**base, **score_answer(item, answer.text, answer.refused, answer.sources,
                                  min_grounding=config.VERIFY_MIN_GROUNDING)}
    row["answer"] = answer.text
    row["top_score"] = retrieved[0].score if retrieved else None
    # A gate refusal made no LLM call, so no provider answered.
    row["provider"] = None if answer.refused else rag.llm.last_provider()
    row["diagnosis"] = diagnose(row)
    if judge and not row["refused"]:
        try:
            raw = rag.llm.complete(
                build_judge_prompt(item["question"], answer.text, answer.sources),
                system=JUDGE_SYSTEM,
            )
        except rag.llm.LLMError as exc:
            raw = None
            row["judge_error"] = str(exc)
        row["judge_score"] = parse_judge_score(raw)
        row["judge_raw"] = (raw or "")[:500]
    return row


def _fmt(value, spec: str) -> str:
    """Format a metric value, printing undefined (None) metrics as n/a."""
    return "n/a" if value is None else format(value, spec)


def _fmt_ci(ci: tuple[float, float] | None) -> str:
    """Render a Wilson interval as " [lo-hi]", or "" when undefined."""
    return "" if ci is None else f" [{ci[0]:.2f}-{ci[1]:.2f}]"


def print_summary(summaries: dict[str, dict], judge: bool) -> None:
    """Print one column per corpus (and "all") with every aggregate metric.

    Binomial metrics carry their 95% interval when `with_intervals` has run, and
    a row whose key is a pair of keys prints both counts as "a/b".
    """
    names = list(summaries)
    print(f"{'metric':<28}" + "".join(f"{n:>{_COL}}" for n in names))
    rows = list(_SUMMARY_ROWS)
    if judge:
        rows += [("judge faithfulness (0-2)", "judge_mean", ".2f"),
                 ("judge unparsed", "judge_unparsed", "d")]
    for label, key, spec in rows:
        if isinstance(key, tuple):
            cells = [f"{s[key[0]]}/{s[key[1]]}" for s in summaries.values()]
        else:
            cells = [_fmt(s[key], spec) + _fmt_ci(s.get("ci", {}).get(key))
                     for s in summaries.values()]
        print(f"{label:<28}" + "".join(f"{c:>{_COL}}" for c in cells))


def print_diagnosis(counts: dict[str, Counter]) -> None:
    """Print the diagnosis count table, one column per corpus (and "all")."""
    print(f"{'diagnosis':<28}" + "".join(f"{n:>{_COL}}" for n in counts))
    for label in DIAGNOSES:
        print(f"{label:<28}" + "".join(f"{c[label]:>{_COL}}" for c in counts.values()))


def main() -> None:
    """Parse arguments, evaluate every question, write JSONL results and print the summary."""
    never_crash_on_console_encoding()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", nargs="+", default=DEFAULT_CORPORA)
    parser.add_argument("--eval-file", default=None,
                        help="defaults to eval/<corpus>_answers_eval.jsonl; one corpus only")
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--mode", choices=["hybrid", "dense"], default=config.RETRIEVAL_MODE)
    parser.add_argument("--limit", type=int, default=None, help="first N questions per corpus")
    parser.add_argument("--judge", action="store_true", help="add the LLM faithfulness judge")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="seconds to wait between questions (free-tier rate limits)")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    if args.eval_file and len(args.corpus) != 1:
        parser.error("--eval-file needs exactly one --corpus")

    embedder = get_embedder()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    first = True
    print(f"{'id':<8} {'ans':>4} {'ref':>4} {'facts':>6} {'cite':>6} {'grnd':>6} "
          f"{'diagnosis':<22} question")
    with out_path.open("w", encoding="utf-8") as out:
        for corpus in args.corpus:
            store, bm25 = load_corpus(corpus)
            eval_file = (Path(args.eval_file) if args.eval_file
                         else config.ROOT / "eval" / f"{corpus}_answers_eval.jsonl")
            items = load_eval(eval_file)[: args.limit] if args.limit else load_eval(eval_file)
            effective = args.mode if bm25 is not None else "dense"
            print(f"-- corpus={corpus} k={args.k} mode={effective} questions={len(items)}")
            for item in items:
                if args.sleep > 0 and not first:
                    time.sleep(args.sleep)
                first = False
                row = evaluate_item(item, corpus, store, bm25, embedder, args.k, args.mode,
                                    args.judge)
                rows.append(row)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                if row.get("error"):
                    print(f"{row['id']:<8} ERROR {row['error'][:60]}")
                    continue
                cites = row["valid_citations"]
                print(
                    f"{row['id']:<8} {'y' if row['answerable'] else 'n':>4} "
                    f"{'y' if row['refused'] else 'n':>4} "
                    f"{_fmt(row['key_fact_recall'], '.2f'):>6} "
                    f"{'-' if cites is None else len(cites):>6} "
                    f"{_fmt(row['grounding_score'], '.2f'):>6} "
                    f"{row['diagnosis']:<22} {row['question'][:50]}"
                )
    print("-" * 80)
    print_summary(with_intervals(summarize_by_corpus(rows), rows), args.judge)
    print("-" * 80)
    print_diagnosis(diagnosis_counts(rows))
    print(f"results: {out_path}")


if __name__ == "__main__":
    main()
