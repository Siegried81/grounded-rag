# React UI

A second front end, next to the Streamlit app (`app.py`, unchanged). It has two parts:

- `api/main.py`: a FastAPI server that runs the same pipeline as `app.py`
  (`retrieve` -> `answer_question` -> `verify_answer`, the same per-corpus caching
  and the same explanations from `rag/ui_helpers.py`) and returns JSON.
- `web/`: a Vite + React 18 single page (plain CSS, no UI kit) that calls that API.

Both UIs read the same `.env` and indexes, so they behave identically.

## Prerequisites

- The Python venv with `requirements.txt` installed (it now includes `fastapi`,
  `uvicorn` and `httpx`).
- Node.js 18+ and npm, for the React app only.
- At least one built index (`python cli.py ingest --corpus ai_act`).

## Development (two processes, hot reload)

From the repository root:

```bash
# 1. API on port 8002
.venv/bin/python -m uvicorn api.main:app --port 8002 --reload      # Linux / WSL / macOS
# .venv/Scripts/python -m uvicorn api.main:app --port 8002 --reload  # Windows

# 2. In another terminal: the React dev server on port 5180
cd web
npm install        # first time only
npm run dev
```

Open http://localhost:5180. Vite proxies every `/api` request to
`http://127.0.0.1:8002`, and the API also allows CORS from `localhost:5180`.
`npm` comes from nvm, which only loads in an interactive shell: `which npm`
should point under `~/.nvm`.

## Production (one process)

```bash
cd web && npm install && npm run build && cd ..   # writes web/dist
.venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8002
```

When `web/dist` exists at startup, the API serves the built UI from `/`, so
http://localhost:8002 is the whole app. `/api/*` routes always take precedence.
Rebuild and restart after changing anything in `web/src`.

With Docker: build the UI on the host first (the image has no Node toolchain),
then `docker compose --profile api up --build` and open http://localhost:8002.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness check (`{"status": "ok"}`). |
| GET | `/api/config[?corpus=NAME]` | Badges (LLM provider and model, embedding provider, default retrieval mode, refusal threshold, grounding bar), `llm_key_configured` (a boolean only, never a key), pipeline steps and limitations. |
| GET | `/api/corpora` | Indexed corpora with documents, chunks, PDF pages, detected languages, file list, available modes and 4 example questions. |
| POST | `/api/ask` | `{corpus, question, mode?, top_k?, use_mmr?}` -> answer, refusal, sources, verification, latency and error. |

`/api/config` and `/api/corpora` work without any LLM key or a running Ollama:
they load the indexes but never embed or call a model.

`POST /api/ask` response fields:

- `answer`: the LLM text with inline `[S#]` markers, or `null`.
- `refused` and `refusal`: `{best_score, threshold, explanation, closest[]}`. The
  gate is the dense cosine threshold from `SCORE_THRESHOLD`; it is not a request
  parameter because changing it changes what a refusal means.
- `sources[]`: `{sid, source, page, location, score, excerpt, excerpt_html, cited}`.
  `excerpt_html` is HTML-escaped server-side and only adds `<mark>` tags.
- `verification`: `{ok, strict_ok, grounding_score, min_grounding, valid_citations,
  invalid_citations, uncited_sentences, explanation}`. `strict_ok` also requires
  that no claim-like sentence is left uncited. Verification runs outside the LLM
  error handling: a failure there is a bug to surface, not a provider outage.
- `latency`: `{retrieval_s, answer_s}`.
- `error`: `null`, or `{kind: "embedding" | "llm" | "index", message, hint}`.
  `index` means the corpus was indexed with another embedding model than the
  configured one (`IndexMismatchError`): re-run `ingest`. These failures return
  HTTP 200 with the passages still included (none for `index`), as in the
  Streamlit app. Invalid input returns 422, and an unknown corpus returns 404.

## What the page shows

- A header with the cite-or-refuse rule and badges for the configuration (an
  amber badge when no LLM key is set).
- A sidebar with the corpus selector and stats, the retrieval settings (mode,
  top_k, MMR), the read-only refusal threshold, About / How it works,
  Limitations, and Clear conversation. On narrow screens it becomes a drawer.
- An empty state with clickable example questions taken from the corpus eval set.
- The chat thread. Answers show `[S#]` chips: click one to scroll to its source
  card and highlight it. A citation to a source that does not exist is struck
  through in red.
- Source cards that are always visible, with file, page, score, a highlighted
  excerpt, and an accent on the cited ones.
- A verification badge with its explanation, invalid citations and uncited claims.
- A refusal panel with the best score against the threshold and the closest passages.
- An error banner with the actionable hint, latency chips and a loading state.
  A request that exceeds the client-side timeout is aborted and reported in the
  same banner, so a stalled provider never leaves the page spinning.
- An AI disclosure footer, always visible: answers are generated by a model from
  the indexed documents and can be wrong; check the cited sources.
- Light and dark themes (`prefers-color-scheme`), visible focus rings, labelled
  controls, and reduced motion when the OS asks for it.

## Tests

`tests/test_api.py` drives the API with FastAPI's `TestClient`. The index,
embedder and LLM are replaced by fakes, so it makes no network calls.
