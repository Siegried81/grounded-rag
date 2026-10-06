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

**Query (`cli.py`, `app.py` Streamlit, `api/main.py` FastAPI + `web/` React)**

4. **Retrieve** (`rag/retrieve.py`): the question is embedded; dense cosine and BM25
   rankings are fused (RRF) and diversified (MMR), both implemented in
   `rag/fusion.py`. See "Cite or refuse" below.
5. **Answer** (`rag/answer.py` + `rag/llm.py`): surviving passages go to the LLM in
   one structured call, each wrapped in a `<source>` fence that marks it as
   untrusted data (prompt-injection mitigation); the model must answer only from
   them and cite `[S1]`, `[S2]`, … Alternative markers some models emit (`【S1】`)
   are normalised to `[S1]`. The LLM client rotates Groq keys, waits for an
   announced rate-limit reset, falls back across providers and caches responses.
6. **Verify** (`rag/verify.py`): every `[S#]` is checked to be real and a lexical
   grounding score is computed, on every answer.

**Evaluation (`scripts/`)**: four suites — retrieval metrics (`run_eval.py`,
scored by `rag/metrics.py`), answer-level metrics (`run_answer_eval.py`,
`rag/answer_metrics.py`), an adversarial red-team suite (`run_redteam.py`) and a
phrasing-robustness check (`run_phrasing_eval.py`). The first and last need only
local embeddings; the other two make one live LLM call per question. Every rate is
printed with a 95% Wilson interval from `rag/stats.py`, and each live run writes
its per-question rows to `logs/`.

## Cite or refuse

An answer is only allowed if it is grounded. The refusal gate is anchored to the
**dense cosine** score: if no passage reaches `SCORE_THRESHOLD` (default 0.35), the
assistant returns `REFUSAL_MESSAGE` instead of answering — regardless of retrieval
mode, since hybrid fusion and MMR only reorder already-relevant passages. The
threshold is the one setting that changes what an answer means (too low → confident
but ungrounded text; too high → answerable questions refused) and is tuned per
embedding model in `config.py`.

**What the gate does not do.** Over 61 live questions it has never fired. The
unanswerable eval questions score 0.599–0.888 top cosine and the answerable ones
0.611–0.910, so no threshold separates them: cosine measures whether a passage is
*about* the question, and an unanswerable question about the corpus's own subject
still is. What refuses in practice is the model following the "answer only from the
sources" instruction. Both live reports therefore print `refusals: gate / model`
next to every refusal metric, so a refusal score is never read as evidence about
the threshold.

**And what verification does not do.** `rag/verify.py` checks that a citation
exists, points inside the retrieved set, and shares vocabulary with the answer —
none of which is a check on truth. The worked example is `aa-03`, in
[`technical_deep_dive.md`](technical_deep_dive.md) §8.2: a fine quoted correctly
from the wrong article's scope, `verify_ok: true`. Read it
before trusting a green verification badge; it is why a source card's filename
opens the whole cited document, not just the excerpt.

## Why a NumPy store, not a vector DB

A corpus here is hundreds to a few thousand chunks; brute-force cosine is a few
milliseconds and needs no server, deploys with the repo, and is trivially
inspectable and testable. The `VectorStore` interface (`add`, `search`, `save`,
`load`) is small enough to re-implement on FAISS/pgvector for a million-chunk
corpus without touching the rest of the pipeline.
