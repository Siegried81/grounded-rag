"""Run a retrieval evaluation against a built index and print quality metrics.

Makes retrieval quality measurable. Relevance is judged at the source level: a
retrieved chunk is relevant if its `source` is listed in the question's
`relevant_sources`, because chunk ids are not stable across re-indexing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/run_eval.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from rag.embed import get_embedder  # noqa: E402
from rag.metrics import hit_rate_at_k, mrr, recall_at_k  # noqa: E402
from rag.retrieve import retrieve  # noqa: E402
from rag.store import VectorStore  # noqa: E402


def load_eval(path: Path) -> list[dict]:
    """Read the eval JSONL (one {question, relevant_sources} object per line)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def main() -> None:
    """Parse arguments, evaluate every question and print per-question and aggregate metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="ai_act")
    parser.add_argument("--eval-file", default="eval/ai_act_eval.jsonl")
    parser.add_argument("--k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    store = VectorStore.load(config.index_path(args.corpus))
    embedder = get_embedder()
    items = load_eval(Path(args.eval_file))

    print(f"{'question':<60} {'hit@k':>6} {'rec@k':>6} {'rr':>6}")
    hits, recalls, rrs = [], [], []
    for item in items:
        relevant = set(item["relevant_sources"])
        results = retrieve(item["question"], store, embedder, top_k=args.k)
        # Source-level judgement: rank positions are labelled with the chunk's source.
        sources = [r.chunk.source for r in results]
        h = hit_rate_at_k(sources, relevant, args.k)
        rec = recall_at_k(sources, relevant, args.k)
        rr = mrr(sources, relevant)
        hits.append(h)
        recalls.append(rec)
        rrs.append(rr)
        print(f"{item['question'][:58]:<60} {h:>6.2f} {rec:>6.2f} {rr:>6.2f}")

    n = len(items) or 1
    print("-" * 80)
    print(f"questions={len(items)} k={args.k}")
    print(
        f"recall@{args.k}={sum(recalls) / n:.3f}  "
        f"hit_rate@{args.k}={sum(hits) / n:.3f}  MRR={sum(rrs) / n:.3f}"
    )


if __name__ == "__main__":
    main()
