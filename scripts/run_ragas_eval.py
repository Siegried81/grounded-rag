"""Score an answer-eval run with RAGAS, using an Azure OpenAI deployment as the judge.

This adds the metric names the field uses next to the project's own; it does not
replace them. scripts/run_answer_eval.py stays the reference (key facts,
citations, grounding, refusals). This script reads its JSONL output, which
records each answer and the exact passages it saw (`contexts`), and asks RAGAS
three questions per answer:

- faithfulness: are the answer's claims supported by those passages?
- answer_relevancy: does the answer address the question? (needs embeddings)
- context_precision: were the passages useful, and ranked first? This is the
  variant WITHOUT a reference answer: the eval sets carry key facts and gold
  files, not reference answers, so RAGAS's context recall is not computed.
  Retrieval recall against the gold files is already measured by
  scripts/run_eval.py.

Which rows are scored, and why:
- Only answerable questions that were answered. RAGAS has no notion of
  cite-or-refuse: a correct refusal scores near 0 on answer relevancy, so
  refusals would read as failures. They stay measured by the project's refusal
  metrics, and the excluded counts are printed so nothing disappears silently.
- Rows with an LLM error are left out, as in the answer eval.

A judge LLM does not give the same score twice, so every row is scored
`--repeats` times (3 by default) and the summary gives the mean of the run means
with their min and max. A judge from the same deployment that wrote the answers
grades its own work: `--judge-deployment` lets a different model judge.

Cost guard: without `--yes` the script only prints how many rows it would score
and how many metric calls that is, then stops. RAGAS makes several LLM calls
per metric (faithfulness splits the answer into statements, then checks each),
so the count is a lower bound on requests.

RAGAS sends anonymous usage events by default; this script turns that off
before importing it. Needs `ragas` and `openai` (requirements-optional.txt).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from rag.logging_utils import never_crash_on_console_encoding  # noqa: E402

METRICS = ("faithfulness", "answer_relevancy", "context_precision")
DEFAULT_IN = config.ROOT / "logs" / "answer_eval_results.jsonl"
DEFAULT_OUT = config.ROOT / "logs" / "ragas_results.jsonl"


def select_rows(rows: list[dict]) -> tuple[list[dict], Counter]:
    """Rows RAGAS can score fairly, and a count of the others by reason."""
    kept, excluded = [], Counter()
    for row in rows:
        if row.get("error"):
            excluded["llm error"] += 1
        elif not row.get("answerable"):
            excluded["unanswerable question"] += 1
        elif row.get("refused"):
            excluded["refused (answerable)"] += 1
        elif not row.get("contexts"):
            excluded["no contexts recorded (older run)"] += 1
        else:
            kept.append(row)
    return kept, excluded


def build_scorers(judge_deployment: str, embed_deployment: str) -> dict:
    """RAGAS metrics bound to an Azure OpenAI judge: name -> async fn(row) -> float.

    Imported here, not at module level, so the module (and its tests) load
    without ragas installed. Temperature 0 keeps the judge as steady as it gets;
    the repeats measure what is left.
    """
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    try:
        from openai import AsyncOpenAI
        from ragas.embeddings.base import embedding_factory
        from ragas.llms.base import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecisionWithoutReference,
            Faithfulness,
        )
    except ImportError as exc:
        raise SystemExit(
            "RAGAS scoring needs `ragas` and `openai`: pip install -r requirements-optional.txt"
        ) from exc
    if not (config.azure_openai_v1_url() and config.AZURE_OPENAI_API_KEY):
        raise SystemExit("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY first.")
    if not (judge_deployment and embed_deployment):
        raise SystemExit("Set the judge deployment (AZURE_OPENAI_DEPLOYMENT or "
                         "--judge-deployment) and AZURE_OPENAI_EMBED_DEPLOYMENT.")
    client = AsyncOpenAI(base_url=config.azure_openai_v1_url() + "/",
                         api_key=config.AZURE_OPENAI_API_KEY)
    llm = llm_factory(judge_deployment, provider="openai", client=client, temperature=0)
    embeddings = embedding_factory("openai", model=embed_deployment, client=client)
    faith = Faithfulness(llm=llm)
    relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    precision = ContextPrecisionWithoutReference(llm=llm)

    async def _faithfulness(row):
        return (await faith.ascore(user_input=row["question"], response=row["answer"],
                                   retrieved_contexts=row["contexts"])).value

    async def _relevancy(row):
        return (await relevancy.ascore(user_input=row["question"], response=row["answer"])).value

    async def _precision(row):
        return (await precision.ascore(user_input=row["question"], response=row["answer"],
                                       retrieved_contexts=row["contexts"])).value

    return {"faithfulness": _faithfulness, "answer_relevancy": _relevancy,
            "context_precision": _precision}


async def score_rows(rows: list[dict], scorers: dict, repeats: int) -> list[dict]:
    """Score every row `repeats` times, one call at a time.

    Sequential on purpose: a small Azure deployment has a tokens-per-minute
    quota, and parallel judges would turn it into 429 errors. A failed metric
    call is recorded as None with its error, never as a 0.
    """
    out = []
    for repeat in range(repeats):
        for row in rows:
            result = {"corpus": row["corpus"], "id": row["id"], "repeat": repeat}
            for name, scorer in scorers.items():
                try:
                    value = await scorer(row)
                    result[name] = None if value is None else float(value)
                except Exception as exc:  # one failed judge call must not end the run
                    result[name] = None
                    result.setdefault("errors", {})[name] = str(exc)[:300]
            out.append(result)
    return out


def summarize(results: list[dict]) -> dict[str, dict[str, dict]]:
    """Per corpus and metric: mean of the per-repeat means, their min and max, and failures.

    Averaging inside each repeat first, then across repeats, gives the spread
    between whole runs, which is what a reader comparing two runs needs.
    """
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    failed = defaultdict(Counter)
    for r in results:
        for name in METRICS:
            if name not in r:
                continue
            if r[name] is None:
                failed[r["corpus"]][name] += 1
            else:
                by[r["corpus"]][name][r["repeat"]].append(r[name])
    summary = {}
    for corpus in sorted({r["corpus"] for r in results}):
        summary[corpus] = {}
        for name in METRICS:
            run_means = [statistics.fmean(v) for _, v in sorted(by[corpus][name].items()) if v]
            summary[corpus][name] = {
                "mean": statistics.fmean(run_means) if run_means else None,
                "min": min(run_means) if run_means else None,
                "max": max(run_means) if run_means else None,
                "repeats": len(run_means),
                "failed_calls": failed[corpus][name],
            }
    return summary


def print_summary(summary: dict, excluded: Counter, n_scored: int) -> None:
    """Print one block per corpus and the rows left out."""
    print(f"scored rows: {n_scored}")
    for reason, n in sorted(excluded.items()):
        print(f"  left out, {reason}: {n}")
    for corpus, metrics in summary.items():
        print(f"-- corpus={corpus}")
        for name, s in metrics.items():
            if s["mean"] is None:
                print(f"  {name:<18} n/a (failed calls: {s['failed_calls']})")
                continue
            print(f"  {name:<18} {s['mean']:.3f}  (runs {s['min']:.3f} to {s['max']:.3f}, "
                  f"{s['repeats']} repeats, failed calls: {s['failed_calls']})")


def main() -> None:
    """Parse arguments, select rows, check the cost guard, score and print the summary."""
    never_crash_on_console_encoding()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(DEFAULT_IN),
                        help="JSONL written by scripts/run_answer_eval.py")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="first N scorable rows")
    parser.add_argument("--judge-deployment", default=config.AZURE_OPENAI_DEPLOYMENT)
    parser.add_argument("--yes", action="store_true", help="really call the judge")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    lines = Path(args.results).read_text(encoding="utf-8").splitlines()
    rows, excluded = select_rows([json.loads(line) for line in lines if line.strip()])
    rows = rows[: args.limit] if args.limit else rows
    providers = Counter(r.get("provider") for r in rows)
    print(f"answers by provider: {dict(providers)}")
    print(f"would score {len(rows)} rows x {args.repeats} repeats x {len(METRICS)} metrics "
          f"= {len(rows) * args.repeats * len(METRICS)} metric calls "
          f"(each is several LLM requests), judge={args.judge_deployment or 'unset'}")
    if not rows:
        reasons = ", ".join(f"{reason}: {n}" for reason, n in sorted(excluded.items()))
        raise SystemExit(f"Nothing to score ({reasons or 'empty file'}). Run "
                         "scripts/run_answer_eval.py first: its rows carry the contexts.")
    if not args.yes:
        for reason, n in sorted(excluded.items()):
            print(f"  left out, {reason}: {n}")
        print("dry run: add --yes to score")
        return

    scorers = build_scorers(args.judge_deployment, config.AZURE_OPENAI_EMBED_DEPLOYMENT)
    results = asyncio.run(score_rows(rows, scorers, args.repeats))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({**r, "judge": args.judge_deployment}, ensure_ascii=False) + "\n")
    print_summary(summarize(results), excluded, len(rows))
    print(f"results: {out}")


if __name__ == "__main__":
    main()
