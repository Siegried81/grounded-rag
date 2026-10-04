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
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
from rag.retrieve import retrieve  # noqa: E402
from rag.store import VectorStore  # noqa: E402
from scripts.run_eval import load_eval  # noqa: E402

DEFAULT_CORPORA = ["ai_act_sections", "filings_sections"]
DEFAULT_OUT = config.ROOT / "logs" / "answer_eval_results.jsonl"

# Summary rows: (label, key in the summary dict, format).
_SUMMARY_ROWS = [
    ("questions", "n", "d"),
    ("errors (excluded)", "n_errors", "d"),
    ("answerable / unanswerable", None, None),
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
            "answerable": bool(item["answerable"])}
    retrieved = retrieve(item["question"], store, embedder, top_k=k, bm25=bm25, mode=mode)
    try:
        answer = answer_question(item["question"], retrieved)
    except rag.llm.LLMError as exc:
        return {**base, "refused": False, "error": str(exc)}
    row = {**base, **score_answer(item, answer.text, answer.refused, answer.sources,
                                  min_grounding=config.VERIFY_MIN_GROUNDING)}
    row["answer"] = answer.text
    row["retrieved_sources"] = [r.chunk.source for r in retrieved]
    row["top_score"] = retrieved[0].score if retrieved else None
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


def print_summary(summaries: dict[str, dict], judge: bool) -> None:
    """Print one column per corpus (and "all") with every aggregate metric."""
    names = list(summaries)
    print(f"{'metric':<28}" + "".join(f"{n:>18}" for n in names))
    rows = list(_SUMMARY_ROWS)
    if judge:
        rows += [("judge faithfulness (0-2)", "judge_mean", ".2f"),
                 ("judge unparsed", "judge_unparsed", "d")]
    for label, key, spec in rows:
        if key is None:
            cells = [f"{s['n_answerable']}/{s['n_unanswerable']}" for s in summaries.values()]
        else:
            cells = [_fmt(s[key], spec) for s in summaries.values()]
        print(f"{label:<28}" + "".join(f"{c:>18}" for c in cells))


def main() -> None:
    """Parse arguments, evaluate every question, write JSONL results and print the summary."""
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
    print(f"{'id':<8} {'ans':>4} {'ref':>4} {'facts':>6} {'cite':>6} {'grnd':>6}  question")
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
                    f"{_fmt(row['grounding_score'], '.2f'):>6}  {row['question'][:50]}"
                )
    print("-" * 80)
    print_summary(summarize_by_corpus(rows), args.judge)
    print(f"results: {out_path}")


if __name__ == "__main__":
    main()
