"""Run a retrieval evaluation against a built index and print quality metrics.

Makes retrieval quality measurable. Relevance is judged at the source level: a
retrieved chunk is relevant if its `source` is listed in the question's
`relevant_sources`, because chunk ids are not stable across re-indexing.

The BM25 index is loaded exactly as `cli.py` and `app.py` do, so by default the
metrics describe the retrieval the app actually runs (`config.RETRIEVAL_MODE`,
hybrid unless overridden). `--mode dense` measures the dense channel alone.

Defaults target the `*_sections` corpora (several documents per corpus), whose
eval file is `eval/<corpus>_eval.jsonl`; the one-file `ai_act` corpus gives a
hit rate of 1.0 by construction and says nothing about ranking. hit_rate@k is a
0/1 outcome per question, so it is printed with a 95% Wilson interval
(rag/stats.py); recall@k and MRR are not binomial and get none.
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
from rag.lexical import BM25Index  # noqa: E402
from rag.metrics import hit_rate_at_k, mrr, recall_at_k  # noqa: E402
from rag.retrieve import retrieve  # noqa: E402
from rag.stats import wilson  # noqa: E402
from rag.store import VectorStore  # noqa: E402

DEFAULT_CORPUS = "ai_act_sections"


def load_eval(path: Path) -> list[dict]:
    """Read the eval JSONL (one {question, relevant_sources} object per line)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def default_eval_file(corpus: str) -> Path:
    """The retrieval eval file that goes with a corpus: eval/<corpus>_eval.jsonl."""
    return config.ROOT / "eval" / f"{corpus}_eval.jsonl"


def load_corpus(corpus: str):
    """Load the vector store and optional BM25 index for a corpus, as the app does.

    An older index without bm25.json is dense-only, exactly like the app.
    """
    store = VectorStore.load(config.index_path(corpus))
    try:
        bm25 = BM25Index.load(config.bm25_path(corpus))
    except FileNotFoundError:
        bm25 = None
    return store, bm25


def format_interval(successes: int, n: int) -> str:
    """"[lo-hi]" for a binomial rate, three decimals, or "[n/a]" without observations."""
    ci = wilson(successes, n)
    return "[n/a]" if ci is None else f"[{ci[0]:.3f}-{ci[1]:.3f}]"


def main() -> None:
    """Parse arguments, evaluate every question and print per-question and aggregate metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--eval-file", default=None,
                        help="defaults to eval/<corpus>_eval.jsonl")
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--mode", choices=["hybrid", "dense"], default=config.RETRIEVAL_MODE)
    args = parser.parse_args()

    store, bm25 = load_corpus(args.corpus)
    embedder = get_embedder()
    eval_file = Path(args.eval_file) if args.eval_file else default_eval_file(args.corpus)
    items = load_eval(eval_file)

    print(f"{'question':<60} {'hit@k':>6} {'rec@k':>6} {'rr':>6}")
    hits, recalls, rrs = [], [], []
    for item in items:
        relevant = set(item["relevant_sources"])
        results = retrieve(
            item["question"], store, embedder, top_k=args.k, bm25=bm25, mode=args.mode
        )
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
    effective = args.mode if bm25 is not None else "dense"
    print(f"questions={len(items)} k={args.k} mode={effective} corpus={args.corpus}")
    n_hits = int(sum(hits))
    print(
        f"recall@{args.k}={sum(recalls) / n:.3f}  "
        f"hit_rate@{args.k}={sum(hits) / n:.3f} "
        f"(95% CI {format_interval(n_hits, len(items))}, n={len(items)})  "
        f"MRR={sum(rrs) / n:.3f}"
    )


if __name__ == "__main__":
    main()
