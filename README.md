# Grounded RAG — cite or refuse

**Ask your documents a question; get an answer where every claim cites a passage
you can read — or an honest "I don't know".**

> **The one rule:** if no retrieved passage clears the relevance bar, the assistant
> refuses instead of inventing an answer. Every factual sentence points to a
> numbered source `[S1]`, `[S2]`…, and every citation is checked after generation.

Proven on two very different corpora: the **EU AI Act** and **SEC 10-K filings**.
No vector DB, no RAG framework: each step is one small, readable module.
Docs: [technical deep dive](docs/technical_deep_dive.md) · [architecture](docs/architecture.md) ·
[React UI & API](docs/react_ui.md) · [Docker](docs/docker.md)

## What's inside

- **Hybrid retrieval**: BM25 keywords + dense embeddings, merged with Reciprocal
  Rank Fusion, then diversified with MMR. The refusal gate stays on the dense
  cosine score, so hybrid mode never lowers the bar for answering.
- **Refusal gate** before the LLM is ever called, plus refusal detection in the answer.
- **`[S#]` citations, normalised**: gpt-oss's `【S1】`, `[**S1**]`, `[S1, S5]` and
  zero-width-space forms are rewritten to `[S1]`, so they are verified like any
  other citation (this took verify-ok from 0.21 to 1.00 on the same answers).
- **Verification on every answer**: hallucinated source numbers, uncited claims,
  and a lexical grounding score.
- **Every citation is readable in context**: a source card shows a window around
  the match, and its filename opens the whole cited document, scrolled to the
  first match — an excerpt alone cannot tell you whether the answer read it right.
  URLs inside a passage are clickable (`http(s)` only).
- **Prompt-injection fence**: retrieved text is wrapped in `<source id="S#">`
  blocks marked as untrusted data; tags inside a passage are neutralised.
- **Free-tier resilience**: rotation over up to 5 Groq keys, waiting for the
  rate-limit window Groq announces, fallback to OpenRouter / Ollama, and an
  on-disk LLM response cache.
- **Two UIs** (React + FastAPI on port 8002, or Streamlit), **Docker**, and one
  structured LLM call per question — no agent loop.
- **Language choice**: interface in English or French, answers in English by
  default (or auto / French / Dutch) whatever the language of the sources.

## Pipeline

```text
 docs (PDF/txt) ─► sentence-aware chunks ─► embed (cached) ─► NumPy store + BM25 index
                                                                     │
 question ─► embed ─► dense ∪ BM25 ─► RRF ─► MMR ─► gate: top cosine < threshold? ─► REFUSE
                                                       │ no
                                                       ▼
                         fenced prompt ─► LLM ─► normalise [S#] ─► verify ─► answer + sources
```

## Results

All runs: hybrid retrieval; generation with `openai/gpt-oss-120b` on Groq's free tier.

The tables below predate the current scoring code (Unicode-aware verifier with
FR/NL stopwords and `strict_ok`, refusal detection shared with the red team,
scale-aware number matching) and will be re-run; until then they describe the
previous scoring, not the current one. Re-run with
`.venv/bin/python scripts/run_answer_eval.py` and
`.venv/bin/python scripts/run_redteam.py --verbose`.

**Retrieval** (`scripts/run_eval.py`, k = 3, each question labelled with its one answering section)

| Corpus            | Questions | Recall@3 | MRR   |
| ----------------- | --------: | -------: | ----: |
| ai_act_sections   |        23 |    1.000 | 0.884 |
| filings_sections  |        25 |    0.900 | 0.840 |

**Answers** (`scripts/run_answer_eval.py`, k = 5, 40 questions incl. 12 unanswerable)

| Metric                   | ai_act_sections | filings_sections |   All |
| ------------------------ | --------------: | ---------------: | ----: |
| Answerable / unanswerable|            14/6 |             14/6 | 28/12 |
| Refusal F1               |           1.000 |            1.000 | 1.000 |
| False-refusal rate       |           0.000 |            0.000 | 0.000 |
| Key-fact recall          |           0.976 |            1.000 | 0.988 |
| Citation validity        |           1.000 |            1.000 | 1.000 |
| Mean lexical grounding   |           0.719 |            0.761 | 0.740 |
| Verify-ok rate           |           1.000 |            1.000 | 1.000 |

**Red team** (`scripts/run_redteam.py`, 21 adversarial cases, deterministic checks)

| Category             | Pass | Category          | Pass |
| -------------------- | ---: | ----------------- | ---: |
| prompt_extraction    |  3/3 | jailbreak         |  2/3 |
| language_switch      |  3/3 | citation_forgery  |  1/2 |
| injection_in_passage |  4/5 | out_of_scope      |  1/3 |
| **All**              | **14/21 (67%)** | personal_data | 0/2 |

The failures are findings, not noise. **No system-prompt sentence appeared in any
answer**. The weak spot is **scope**: off-topic and personal-data questions
(cookie recipe, quicksort, a commissioner's home address) cleared the cosine gate
and the answer carried no refusal the checker recognises: the threshold measures
similarity, not topicality. Three cases show the model **following embedded
instructions**: it emitted the `<prompt>` tag an injected passage asked for, ended
with a phrase a role-play demanded, and cited a user-demanded `[S9]` (`verify.py` flags it).
`--verbose` prints each answer; metric definitions are in the [deep dive](docs/technical_deep_dive.md).

## Quick start

Linux / WSL (Windows PowerShell: `.venv\Scripts\Activate.ps1` and `copy`):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                  # set LLM_PROVIDER and a key (e.g. GROQ_API_KEY)
ollama pull nomic-embed-text          # local embeddings (the default EMBED_PROVIDER)
ollama pull llama3.2:3b               # only needed for the Ollama generation fallback

python cli.py ingest --corpus ai_act
python cli.py ask --corpus ai_act "What are the obligations for high-risk AI systems?"
```

`config.py` refuses to import with an invalid `RETRIEVAL_MODE`, and a corpus
indexed with another embedding model than the configured one is reported as an
`index` error instead of answering with meaningless scores: re-run `ingest`.

**React UI** — two terminals, then open http://localhost:5180 (API on 8002):

```bash
.venv/bin/python -m uvicorn api.main:app --port 8002
cd web && npm install && npm run dev
```

`npm` is installed through nvm, which only loads in an interactive shell: run
the second terminal as a normal login shell and check that `which npm` points
under `~/.nvm` before `npm install`.

**Streamlit**: `python -m streamlit run app.py` (port 8501) · **Docker**:
`docker compose up --build` (Streamlit on :8501) or
`docker compose --profile api up --build` (API + built React UI on :8002);
see [docs/docker.md](docs/docker.md) for the bundled Ollama.

**Evaluations** (live LLM calls, except the retrieval one):

```bash
.venv/bin/python scripts/run_eval.py                     # retrieval; defaults to ai_act_sections, Wilson interval
.venv/bin/python scripts/run_answer_eval.py              # answers; per-row provider and diagnosis columns
.venv/bin/python scripts/run_redteam.py --verbose        # red team; writes logs/redteam_results.jsonl
.venv/bin/python scripts/run_phrasing_eval.py            # same question, formal / casual / other-language phrasings
```

## Providers

| Role       | Default                              | Fallbacks                       | `.env`           |
| ---------- | ------------------------------------ | ------------------------------- | ---------------- |
| Generation | Groq `openai/gpt-oss-120b` (1–5 keys)| OpenRouter, Ollama `llama3.2:3b`| `LLM_PROVIDER`   |
| Embeddings | Ollama `nomic-embed-text` (local)    | any OpenAI-compatible endpoint  | `EMBED_PROVIDER` |
With Ollama for both, everything runs offline and free.

## Tests

`python -m pytest -q` — the whole suite runs **offline** (fake embedder, all HTTP
mocked); the count printed at the end is the current one.

## Limitations

- **The gate measures similarity, not scope**: off-topic questions can pass it.
- **Grounding is lexical** (word overlap, not entailment); refusal detection is a phrase heuristic.
- **Small eval sets**: 23–25 retrieval / 40 answer questions, a single 10-K (Apple FY2025).
- **One live run, one model** (gpt-oss-120b, free tier): numbers will move with the model.
- **Brute-force NumPy search, one-shot generation**: fine at this scale, no multi-hop.

Details and roadmap: [technical deep dive](docs/technical_deep_dive.md).

---
Siegried Camus
