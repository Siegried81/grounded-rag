"""Re-score recorded answers with the current rules, with no LLM call and no retrieval.

The answer text, the passages it saw (`contexts`) and the gold key facts are all
on disk, so a change to how answers are scored (rag/answer_metrics.py) can be
applied to an earlier run without paying for it again. The answer eval rows are
re-scored with `score_answer`; the red-team rows have their `must_refuse` check
and their over-refusal flag recomputed with `is_refusal`.

Usage:
    python scripts/rescore_offline.py logs/answer_eval_results.jsonl
    python scripts/rescore_offline.py logs/redteam_results.jsonl --write out.jsonl

Each file prints the summary as recorded ("before") and as re-scored ("after").
`--write` saves the re-scored rows; the model's answers are never touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config  # noqa: E402
from rag.answer_metrics import is_refusal, score_answer, summarize  # noqa: E402
from rag.types import Chunk, Retrieved  # noqa: E402

ANSWER_METRICS = [
    "n_refused", "n_gate_refused", "n_model_refused", "n_cited_refused",
    "refusal_precision", "refusal_recall", "refusal_f1", "false_refusal_rate",
    "key_fact_recall", "key_fact_recall_answered", "citation_validity",
    "mean_grounding", "verify_ok_rate",
]


def load_jsonl(path: Path) -> list[dict]:
    """Read one JSON object per non-empty line."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _eval_items(corpus: str) -> dict[str, dict]:
    """Gold items of a corpus's answer eval file, by id."""
    return {it["id"]: it for it in load_jsonl(ROOT / "eval" / f"{corpus}_answers_eval.jsonl")}


def rescore_answer_rows(rows: list[dict]) -> list[dict]:
    """Re-score recorded answer-eval rows; rows without contexts skip citation and grounding."""
    items: dict[str, dict[str, dict]] = {}
    out = []
    for row in rows:
        if row.get("error"):
            out.append(row)
            continue
        corpus = row.get("corpus", "unknown")
        item = items.setdefault(corpus, _eval_items(corpus))[row["id"]]
        contexts = row.get("contexts")
        sources = [Retrieved(Chunk(f"c{i}", text, "recorded", i, corpus), 1.0)
                   for i, text in enumerate(contexts or [], 1)]
        gate = bool(row.get("gate_refused"))
        new = score_answer(item, row["answer"], gate, sources,
                           min_grounding=config.VERIFY_MIN_GROUNDING)
        if contexts is None:  # nothing to verify against: keep what was recorded
            for key in ("valid_citations", "invalid_citations", "grounding_score", "verify_ok"):
                new[key] = row.get(key)
        merged = {**row, **new}
        from run_answer_eval import diagnose
        merged["diagnosis"] = diagnose(merged)
        out.append(merged)
    return out


def rescore_redteam_rows(rows: list[dict]) -> list[dict]:
    """Recompute `must_refuse` and the over-refusal flag of recorded red-team rows."""
    from run_redteam import over_refused
    cases = {c["id"]: c for c in load_jsonl(ROOT / "eval" / "redteam.jsonl")}
    out = []
    for row in rows:
        if row.get("error") or row.get("answer") is None:
            out.append(row)
            continue
        case = cases[row["id"]]
        failures = [f for f in row["failures"] if f != "did not refuse"]
        if case.get("must_refuse") and not is_refusal(row["answer"], row["refused"]):
            failures.append("did not refuse")
        over = (not row.get("leak_blocked")
                and over_refused(case, row["answer"], row["refused"], config.TOP_K))
        out.append({**row, "failures": failures, "passed": not failures, "over_refused": over})
    return out


def _fmt(v) -> str:
    return "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def print_answer_comparison(before: list[dict], after: list[dict]) -> None:
    """Print each summary metric as recorded and as re-scored, per corpus and overall."""
    corpora = list(dict.fromkeys(r.get("corpus", "unknown") for r in before))
    for name in [*corpora, *(["all"] if len(corpora) > 1 else [])]:
        pick = (lambda rs: rs) if name == "all" else (lambda rs: [r for r in rs if r.get("corpus") == name])
        b, a = summarize(pick(before)), summarize(pick(after))
        b["n_cited_refused"] = 0  # the recorded run had no such notion
        print(f"-- {name}")
        for key in ANSWER_METRICS:
            mark = "  <-" if _fmt(b[key]) != _fmt(a[key]) else ""
            print(f"   {key:<26} {_fmt(b[key]):>8} -> {_fmt(a[key]):>8}{mark}")
        changed = [r["id"] for r, n in zip(pick(before), pick(after), strict=True)
                   if r.get("diagnosis") != n.get("diagnosis")]
        print(f"   diagnosis changed: {changed}")


def print_redteam_comparison(before: list[dict], after: list[dict]) -> None:
    """Print pass counts as recorded and as re-scored."""
    for label, rs in (("before", before), ("after", after)):
        print(f"   {label}: {sum(r['passed'] for r in rs)}/{len(rs)} passed, "
              f"{sum(bool(r.get('over_refused')) for r in rs)} refused where an answer was expected")
    for b, a in zip(before, after, strict=True):
        if b["passed"] != a["passed"] or bool(b.get("over_refused")) != bool(a.get("over_refused")):
            print(f"   {a['id']}: passed {b['passed']} -> {a['passed']}, "
                  f"over_refused {b.get('over_refused')} -> {a.get('over_refused')}")


def main(argv: list[str] | None = None) -> int:
    """Re-score one results file and print the before/after comparison."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("file", type=Path)
    parser.add_argument("--write", type=Path, default=None, help="save the re-scored rows here")
    args = parser.parse_args(argv)
    rows = load_jsonl(args.file)
    print(args.file)
    if "category" in rows[0]:
        after = rescore_redteam_rows(rows)
        print_redteam_comparison(rows, after)
    else:
        after = rescore_answer_rows(rows)
        print_answer_comparison(rows, after)
    if args.write:
        args.write.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in after) + "\n",
                              encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
