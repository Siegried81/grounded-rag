"""Linguistic-equity check: does retrieval find the same passages however a question is phrased?

Each group in eval/phrasings_eval.jsonl asks one question three ways: the
`original` wording from the retrieval eval set, a `casual` rewording in the same
language, and an `other_language` translation (FR<->EN). All three share the
group's `relevant_sources`, so a difference in hit@k between styles is caused by
the phrasing alone, not by the question being easier.

Two numbers come out, both 0/1 per item and so printed with a 95% Wilson
interval (rag/stats.py):
- hit@k per style, judged at the source level exactly as scripts/run_eval.py;
- phrasing consistency: the share of groups whose three phrasings all get the
  same hit/miss verdict. A group where only the casual or translated wording
  misses is the one a user in that language would feel.

Retrieval runs as the app does (same BM25 loading and configured mode); the
LLM is never called. Needs the built indexes; run it manually, not in CI.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# Allow `python scripts/run_phrasing_eval.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from rag.embed import get_embedder  # noqa: E402
from rag.logging_utils import never_crash_on_console_encoding  # noqa: E402
from rag.metrics import hit_rate_at_k, mrr  # noqa: E402
from rag.retrieve import retrieve  # noqa: E402
from rag.stats import format_rate  # noqa: E402
from scripts.run_eval import load_corpus, load_eval  # noqa: E402

DEFAULT_EVAL = config.ROOT / "eval" / "phrasings_eval.jsonl"
STYLES = ("original", "casual", "other_language")


def score_items(items: list[dict], k: int, mode: str, embedder, corpora: dict) -> list[dict]:
    """Run retrieval for every item and attach `hit` (0/1) and `rr`; `corpora` caches (store, bm25)."""
    rows = []
    for item in items:
        if item["corpus"] not in corpora:
            corpora[item["corpus"]] = load_corpus(item["corpus"])
        store, bm25 = corpora[item["corpus"]]
        results = retrieve(item["question"], store, embedder, top_k=k, bm25=bm25, mode=mode)
        sources = [r.chunk.source for r in results]
        relevant = set(item["relevant_sources"])
        rows.append({**item, "hit": int(hit_rate_at_k(sources, relevant, k)),
                     "rr": mrr(sources, relevant)})
    return rows


def hits_by_style(rows: list[dict]) -> dict[str, tuple[int, int]]:
    """Map each style to (hits, items), in STYLES order, then any other style seen."""
    table: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        table[r["style"]][0] += r["hit"]
        table[r["style"]][1] += 1
    order = [s for s in STYLES if s in table] + sorted(set(table) - set(STYLES))
    return {s: (table[s][0], table[s][1]) for s in order}


def consistency(rows: list[dict]) -> tuple[int, int]:
    """(consistent groups, groups): a group is consistent when every phrasing has the same hit."""
    verdicts: dict[str, set[int]] = defaultdict(set)
    for r in rows:
        verdicts[r["group"]].add(r["hit"])
    return sum(1 for v in verdicts.values() if len(v) == 1), len(verdicts)


def main() -> None:
    """Parse arguments, score every phrasing and print per-style hit rates and consistency."""
    never_crash_on_console_encoding()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-file", default=str(DEFAULT_EVAL))
    parser.add_argument("--corpus", nargs="*", default=None,
                        help="restrict to these corpora (default: every corpus in the file)")
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--mode", choices=["hybrid", "dense"], default=config.RETRIEVAL_MODE)
    args = parser.parse_args()

    items = load_eval(Path(args.eval_file))
    if args.corpus:
        items = [i for i in items if i["corpus"] in args.corpus]
    corpora: dict[str, tuple] = {}
    rows = score_items(items, args.k, args.mode, get_embedder(), corpora)

    print(f"{'group':<26} {'style':<15} {'hit':>4} {'rr':>5}  question")
    for r in rows:
        print(f"{r['group']:<26} {r['style']:<15} {r['hit']:>4} {r['rr']:>5.2f}  "
              f"{r['question'][:40]}")
    print("-" * 80)
    modes = {c: (args.mode if bm25 is not None else "dense") for c, (_, bm25) in corpora.items()}
    print(f"k={args.k} mode={modes} groups={consistency(rows)[1]} items={len(rows)}")
    for style, (hits, n) in hits_by_style(rows).items():
        print(f"hit@{args.k} {style:<16} {format_rate(hits, n)}")
    print(f"phrasing consistency    {format_rate(*consistency(rows))}")


if __name__ == "__main__":
    main()
