# Architecture (overview)

A one-page map of the engine. The full reference — every decision, trade-off and
limitation — is in [`technical_deep_dive.md`](technical_deep_dive.md).

## Pipeline

**Ingestion (offline, `rag/ingest.py`)**

1. **Chunk** (`rag/smart_chunk.py`): documents are split into sentence-aware
   passages (`chunk.py` is a fixed-character fallback). PDFs are chunked page by
   page so a citation can name a page. Each `Chunk` records its source and ordinal.
2. **Embed** (`rag/embed.py` + `rag/cache.py`): chunk texts become vectors via an
   `Embedder` (Ollama / hosted / fake). An on-disk cache means unchanged passages
   are never re-embedded.
3. **Index**: vectors go into a NumPy cosine store (`rag/store.py`) and a BM25
   keyword index (`rag/lexical.py`), both saved under `index/<corpus>`.

**Query (`cli.py`, `app.py`)**

4. **Retrieve** (`rag/retrieve.py`): the question is embedded; dense cosine and BM25
   rankings are fused (RRF) and diversified (MMR). See "Cite or refuse" below.
5. **Answer** (`rag/answer.py` + `rag/llm.py`): surviving passages go to the LLM in
   one structured call; it must answer only from them and cite `[S1]`, `[S2]`, …
6. **Verify** (`rag/verify.py`): every `[S#]` is checked to be real and a lexical
   grounding score is computed, on every answer.

## Cite or refuse

An answer is only allowed if it is grounded. The refusal gate is anchored to the
**dense cosine** score: if no passage reaches `SCORE_THRESHOLD` (default 0.35), the
assistant returns `REFUSAL_MESSAGE` instead of answering — regardless of retrieval
mode, since hybrid fusion and MMR only reorder already-relevant passages. The
threshold is the one setting that changes what an answer means (too low → confident
but ungrounded text; too high → answerable questions refused) and is tuned per
embedding model in `config.py`.

## Why a NumPy store, not a vector DB

A corpus here is hundreds to a few thousand chunks; brute-force cosine is a few
milliseconds and needs no server, deploys with the repo, and is trivially
inspectable and testable. The `VectorStore` interface (`add`, `search`, `save`,
`load`) is small enough to re-implement on FAISS/pgvector for a million-chunk
corpus without touching the rest of the pipeline.
