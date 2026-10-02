# Grounded RAG — Technical Deep Dive

This document explains how the engine is built, why each decision was made, and
what it deliberately does not do. The README is the business-first overview; this
is the engineering reference.

---

## 1. Design goals

1. **Grounded or silent.** Every factual sentence is tied to a retrieved passage,
   or the system refuses. There is no "best effort" ungrounded answer.
2. **Transparent, not magic.** No vector-DB black box and no embedding framework.
   Each stage is one small module you can read in a sitting.
3. **Swappable backends.** Embedding and generation are behind thin seams, so you
   move between local Ollama, a hosted model, or a test fake without touching the
   pipeline.
4. **Offline-testable.** The whole test suite runs with no network, no API key and
   no model download — a hard requirement, not a nicety.
5. **Right-sized.** It is a focused engine (~15 modules), not a platform. Where a
   production system would need more (a real vector DB, a reranker), the seam is
   named but the complexity is not added "just in case".

---

## 2. Data model (`rag/types.py`)

Three tiny types are the contract every module builds against:

- `Chunk(id, text, source, ordinal, corpus, meta)` — one retrievable passage.
  `source` + `ordinal` make a citation traceable; `meta` carries extras (e.g. a
  PDF `page`) without widening the type. `id = f"{corpus}:{source}:{ordinal}"`.
- `Retrieved(chunk, score)` — a chunk paired with its relevance score.
- `Embedder` (a `Protocol`) — anything with a `model` name and
  `embed(texts) -> list[list[float]]`. The pipeline depends only on this protocol,
  never on a concrete model, which is what makes the backend swappable and the
  tests able to inject a deterministic fake.

Keeping the contract in one dependency-free module is what lets the other modules
stay decoupled and independently testable.

---

## 3. Ingestion (`rag/ingest.py`)

Ingestion is the offline half of the pipeline and produces two reproducible
artifacts per corpus: a dense vector store and a BM25 keyword index.

```text
files in data/<corpus>/   →  chunk  →  embed (cached)  →  VectorStore (+ save)
                                               └────────→  BM25Index   (+ save)
```

- **File handling.** `.txt`/`.md` are read as UTF-8; `.pdf` is read **page by
  page** (via `pypdf`) so page numbers survive into `meta["page"]`. `README.md`
  is skipped (it documents the folder, it is not evidence). Order is sorted by
  filename for determinism.
- **Empty corpus → error.** An empty index would make *every* question refuse,
  silently. `ingest_corpus` raises instead.

### 3.1 Chunking (`rag/smart_chunk.py`, fallback `rag/chunk.py`)

The default `smart_chunk_text` packs **whole sentences** into windows of up to
`CHUNK_SIZE` characters. When a window is full, the next one begins with the
trailing sentences that fit inside `CHUNK_OVERLAP` characters (sentence-level
overlap, not a cut-in-half character overlap). A single sentence longer than
`CHUNK_SIZE` becomes its own chunk — a word is never split.

Why: fixed character windows (the original `chunk.py`, kept as a `CHUNKER=char`
fallback) cut sentences mid-word, which both degrades the embedding of a passage
and makes a cited quote unreadable. The sentence splitter is a pragmatic,
dependency-free heuristic (breaks on `.!?` + whitespace, and blank lines); it can
over-split on abbreviations like "e.g.", which only shifts a boundary slightly and
is an acceptable trade for having no NLP dependency.

PDFs go through `chunk_pages`, which chunks each page independently (so a chunk
never spans a page) while keeping globally increasing ordinals — that is what lets
a citation say "p.7".

### 3.2 Embedding cache (`rag/cache.py`)

Embedding is the slow/expensive stage. `EmbeddingCache` is a JSON map keyed by
`sha256(model \0 text)`. `embed_with_cache` sends only cache misses to the
embedder (deduplicated, one batch) and returns vectors in input order. Keying by
model guarantees a cached vector from one model is never served for another
(different embedding spaces are not comparable). Re-ingesting an unchanged corpus
therefore embeds nothing.

---

## 4. Indexes

### 4.1 Dense store (`rag/store.py`)

A brute-force, vectorised cosine search over an in-memory `float32` matrix,
persisted as `vectors.npy` + `store.json` (chunks in the same row order). Zero-norm
vectors score 0 instead of NaN so they rank last. `vectors_for_ids` exposes stored
vectors so a reranker (MMR) can measure passage-to-passage similarity without
re-embedding.

Why not a vector DB: at thousands of chunks, cosine over a NumPy matrix is
milliseconds, has no dependency, and the index is two inspectable files. A real
corpus would swap this module for FAISS/pgvector behind the same `search` seam.

### 4.2 Keyword index (`rag/lexical.py`)

A persisted BM25 (`rank_bm25`, Okapi) index over the same chunks, with a shared
tokeniser (lowercase, split on non-alphanumerics) so queries and documents are
normalised identically. BM25 catches exact terms — identifiers, acronyms, article
numbers — that dense embeddings blur. Chunks that share no term with the query
score 0 and are dropped, since they are noise on a keyword channel.

---

## 5. Retrieval (`rag/retrieve.py`)

This is where the pieces combine, and where the refusal contract lives.

```text
query
  → dense = store.search(q, CANDIDATE_K)          # cosine candidate pool
  → dense_pass = [r for r in dense if r.score >= SCORE_THRESHOLD]
  → if not dense_pass: return []                   # REFUSE (the whole contract)
  → if hybrid and bm25:                            # fuse keyword channel
        fused = RRF( dense_pass ids , bm25.search(q, CANDIDATE_K) ids )
        ranked = fused mapped back to Retrieved
    else:
        ranked = dense_pass
  → if USE_MMR: ranked = MMR(q, ranked vectors, MMR_LAMBDA)   # diversify
  → return ranked[:TOP_K]
```

**The refusal gate is anchored to the dense cosine score, always.** Hybrid fusion
and MMR only reorder and diversify passages that are *already* semantically
relevant — they can never admit an answer that dense retrieval would have refused.
This is deliberate: BM25 scores and RRF scores live on scales where "is this
relevant enough to answer at all?" is not a meaningful threshold, so the semantic
cosine score remains the single, stable gate.

- **RRF (`fusion.reciprocal_rank_fusion`).** Merges the dense and keyword rankings
  by summing `1/(k+rank)`. It uses ranks only, so the two incompatible score scales
  combine without normalisation.
- **MMR (`fusion.mmr`).** Greedily selects passages maximising
  `λ·sim(query) − (1−λ)·max sim(already-selected)`, dropping near-duplicates. It is
  skipped gracefully if the store cannot supply vectors (e.g. a test fake).

Knobs (all in `config.py`): `RETRIEVAL_MODE` (`hybrid`|`dense`), `CANDIDATE_K`,
`TOP_K`, `SCORE_THRESHOLD`, `USE_MMR`, `MMR_LAMBDA`.

---

## 6. Generation (`rag/llm.py`, `rag/answer.py`)

- **`llm.complete`** is a provider-agnostic, single-shot chat call. It tries
  providers in priority order (`groq → openrouter → ollama` by default), skips any
  whose API key is missing, falls through on failure, and raises `LLMError` only if
  all fail. Config is read at call time, so tests can monkeypatch keys. There is no
  agent loop and no tool use — one structured call per question.
- **`answer.answer_question`** enforces the contract:
  - empty retrieval → returns `REFUSAL_MESSAGE` **without calling the LLM** (the
    safest way to avoid an ungrounded answer is to never give the model the chance);
  - otherwise it builds a prompt of numbered, source-labelled context blocks and a
    system prompt that tells the model to answer *only* from the sources, cite them
    inline as `[S1]`, and say it does not know if they are insufficient.

---

## 7. Verification (`rag/verify.py`)

Turns "cite-or-refuse" from a claim into something checked, on every answer, with
no extra LLM call. `verify_answer` returns a `VerificationReport`:

- `invalid_citations` — `[S#]` markers pointing outside the source list (a
  hallucinated reference). Any invalid citation makes the answer **not ok**.
- `uncited_sentences` — claim-like sentences (more than ~4 content words) carrying
  no citation.
- `grounding_score` — fraction of the answer's content words that also appear in
  the union of the **cited** sources. `ok` requires it to clear
  `VERIFY_MIN_GROUNDING`.
- The exact refusal message verifies as fully ok (it asserts nothing).

**What it is and isn't.** The grounding score is a cheap *lexical* faithfulness
proxy: it catches an answer drifting away from its sources, not a subtle misreading
that reuses the same words. It is a guardrail and a signal, not an entailment
checker — an NLI/LLM-judge pass is named in the roadmap as the stronger version.

---

## 8. Evaluation & observability

- **`scripts/run_eval.py` + `rag/metrics.py`.** Runs retrieval over a labelled
  question set and reports `recall@k`, `hit_rate@k`, `MRR`. Relevance is judged at
  the **source level**: a retrieved chunk counts if its `source` is in the
  question's `relevant_sources`. Caveat: the shipped `eval/ai_act_eval.jsonl` has a
  single-document corpus, so source-level recall is 0/1 per question — it becomes
  genuinely informative with multi-document corpora (e.g. several filings).
- **`rag/logging_utils.py`.** Appends one JSON line per query (UTC timestamp,
  corpus, #retrieved, top score, refused). Logging never raises — it must not be
  able to break answering.

---

## 9. Testing strategy

81 tests, all offline, enforced by construction:

- the embedder under test is a **deterministic fake** (hash-seeded, L2-normalised);
- every HTTP call (Ollama/Groq/OpenRouter embeddings and chat) is **monkeypatched**;
- modules with cross-dependencies are tested with in-file fakes so each suite runs
  independently of the others.

Beyond unit tests, an end-to-end path (ingest → hybrid retrieve → answer → verify,
with the LLM mocked) confirms the modules actually interoperate and that the
refusal branch makes no LLM call.

---

## 10. Known limitations

- **Lexical grounding only.** No semantic entailment check (see §7).
- **Single-document sample corpora.** Retrieval metrics need multi-document sets to
  be fully meaningful (§8).
- **Heuristic sentence splitter.** Over-splits on abbreviations; acceptable for
  chunk boundaries, not a linguistic tokenizer.
- **Brute-force dense search.** Fine to ~10⁴–10⁵ chunks; beyond that, swap in an
  ANN index behind `store.search`.
- **One-shot generation.** No multi-hop / query decomposition for questions that
  need evidence spread across many passages.

---

## 11. Roadmap

1. **Cross-encoder reranker** over the fused candidate pool before `TOP_K`, for a
   sharper final ordering than MMR alone.
2. **LLM/NLI faithfulness check** as an optional, stronger verifier on top of the
   lexical proxy.
3. **Real-filing eval sets** (multiple 10-Ks) with multi-source relevance labels.
4. **Answer streaming** in the CLI and Streamlit UI.
5. **ANN store** (FAISS/pgvector) behind the existing `search` seam for larger
   corpora.
6. **Incremental ingest** keyed on file hashes (the embedding cache already makes
   re-embedding free; this would skip unchanged files end to end).
