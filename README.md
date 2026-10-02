# Grounded RAG — a cite-or-refuse document assistant

A small, domain-agnostic Retrieval-Augmented Generation engine: drop documents in,
ask questions in natural language, and get an answer **grounded in retrieved
passages with inline citations** — or an honest refusal when nothing relevant was
found.

It is built to be transparent: no vector-DB black box, no embedding framework. Every
step — chunk, embed, store, retrieve, generate, verify — is one small, readable
module. The same engine is proven on two very different corpora:

- **EU AI Act** — *"What are the obligations for high-risk AI systems?"*
- **SEC 10-K filings** — *"What risk factors does the company report?"*

> **Deep dive:** the full architecture, every design decision and the measurement
> choices are in [`docs/technical_deep_dive.md`](docs/technical_deep_dive.md).

## The one rule it will not break

If no retrieved passage clears the relevance threshold, the assistant **refuses to
answer** rather than inventing one. Every factual sentence it produces points back
to a numbered source `[S1]`, `[S2]`… that you can expand and read, and `verify.py`
checks — on every answer — that each citation is real. An answer with no sources is
treated as a failure, not a result.

## How it works

```text
documents (PDF / txt)
   → smart_chunk.py  pack whole sentences into passages (page-aware for PDFs)
   → cache.py        skip re-embedding unchanged passages
   → embed.py        embed each passage (Ollama / hosted / fake)
   → store.py        persist vectors + metadata (NumPy, cosine)
   → lexical.py      build a BM25 keyword index alongside
   ──────────────────────────────────────────────────────────────────────
question
   → embed.py        embed the question with the same model
   → retrieve.py     dense cosine + BM25, fused (RRF), diversified (MMR)
                     — refuse if no passage clears the cosine threshold
   → answer.py       build a grounded prompt, call the LLM, enforce citations
   → verify.py       check every [S#] is real and measure lexical grounding
```

Each question is a **single structured LLM call** — no agent loop, no multi-step
autonomous tool use.

## What makes it more than a toy

- **Hybrid retrieval.** Dense embeddings miss exact terms (identifiers, acronyms,
  article numbers); a BM25 keyword channel catches them, and the two rankings are
  merged with Reciprocal Rank Fusion. MMR then drops near-duplicate passages so the
  top-k is diverse. The refusal gate stays anchored to the dense cosine score, so
  hybrid mode never lowers the bar for answering. Toggle with `RETRIEVAL_MODE`.
- **Sentence-aware chunking.** Passages end on sentence boundaries instead of
  mid-word, improving both embedding quality and how a cited quote reads. PDFs are
  chunked page-by-page so a citation can point to a page.
- **Answer verification.** `verify.py` parses the `[S#]` markers, flags any pointing
  at a non-existent source (a hallucinated reference), flags claim-like sentences
  with no citation, and computes a lexical grounding score — a cheap, offline
  faithfulness proxy that runs on every answer.
- **Embedding cache.** Re-ingesting an unchanged corpus re-embeds nothing.
- **Measurable retrieval.** `scripts/run_eval.py` scores retrieval against a
  labelled question set (recall@k, hit-rate@k, MRR) so changes are judged by
  numbers, not vibes.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows (PowerShell)  ·  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                # then set your LLM provider + key
```

### 1. Add documents

Drop files into the corpus folders (a short EU AI Act primer ships as a sample):

```text
data/ai_act/      *.txt, *.md, *.pdf
data/filings/     a 10-K as .txt or .pdf  (see data/filings/README.md)
```

### 2. Build the index

```bash
python cli.py ingest --corpus ai_act
python cli.py ingest --corpus filings
```

### 3. Ask

```bash
python cli.py ask --corpus ai_act "What are the obligations for high-risk AI systems?"
```

…or launch the chat UI (use `python -m streamlit` — it avoids a Windows app-control
block on the `streamlit.exe` launcher):

```bash
python -m streamlit run app.py
```

### 4. Evaluate retrieval

```bash
python scripts/run_eval.py --corpus ai_act --eval-file eval/ai_act_eval.jsonl
```

## Providers (all free tiers, no provider locked in)

| Role       | Default                            | Fallbacks                          | Set in `.env`    |
| ---------- | ---------------------------------- | ---------------------------------- | ---------------- |
| Embeddings | Ollama `nomic-embed-text` (local)  | hosted OpenAI-compatible endpoint  | `EMBED_PROVIDER` |
| Generation | Groq                               | OpenRouter, Ollama (local)         | `LLM_PROVIDER`   |

With Ollama for both, the whole thing runs offline and free:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

## Repository layout

```text
rag/
  types.py        shared dataclasses + the Embedder protocol (the contract)
  smart_chunk.py  sentence-aware, page-aware chunking   (chunk.py = char fallback)
  cache.py        on-disk embedding cache
  embed.py        Ollama / hosted / fake embedders + factory
  store.py        NumPy cosine vector store (save/load)
  lexical.py      BM25 keyword index
  fusion.py       Reciprocal Rank Fusion + MMR
  retrieve.py     hybrid retrieval + refusal gate
  llm.py          provider-agnostic single-shot chat client
  answer.py       grounded prompt + cite-or-refuse
  verify.py       citation & grounding verification
  metrics.py      recall@k / hit-rate@k / MRR
  logging_utils.py structured query logging (JSONL)
  ingest.py       build the indexes for a corpus
cli.py            ingest / ask
app.py            Streamlit chat UI
scripts/run_eval.py   retrieval evaluation
eval/             labelled question sets
tests/            81 offline tests (no network, no keys)
docs/             architecture.md · technical_deep_dive.md
```

## Tests

```bash
python -m pytest -q
```

All 81 tests run **offline**: the embedder is a deterministic fake and every HTTP
call is mocked. No network, no API key, no model download required.

## Why these design choices

- **NumPy store, not a vector DB.** At this scale a persisted cosine search is a few
  lines, has no extra dependency, and every step is inspectable. A production corpus
  would swap `store.py` for FAISS/pgvector behind the same interface — adding that
  here would be abstraction "just in case".
- **Pluggable backends.** The engine never imports a concrete model or vendor; it
  depends on a tiny `Embedder` protocol and duck-typed stores, which is what lets
  the tests inject fakes and lets you move from local Ollama to a hosted model
  without touching the pipeline.
- **Cite-or-refuse, then verify.** Grounding is only worth something if the system
  admits when it has nothing to ground on, and if the grounding is checked rather
  than asserted. The threshold and the grounding floor are both explicit and
  configurable.

Known limitations and the roadmap (cross-encoder reranking, real-filing eval sets,
answer streaming) live in [`docs/technical_deep_dive.md`](docs/technical_deep_dive.md).
