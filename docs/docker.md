# Running with Docker

The image runs the Streamlit UI by default and the CLI on demand. Corpora are
bind-mounted read-only from `./data`; built indexes (with their embedding cache)
and logs live in named volumes. Secrets stay in `.env` on the host and are never
copied into the image.

## Prerequisites

- Docker Engine with Compose v2.24 or newer (needed for `path`/`required: false`
  on `env_file` and `required: false` on `depends_on`).
- On Windows with WSL: Docker Desktop, Settings, Resources, WSL integration,
  enabled for your distribution, or `docker` in WSL talks to nothing.
- A `.env` file copied from `.env.example` (optional: without it the app falls
  back to the defaults in `config.py`).

## 1. Build

```bash
docker compose build
```

## 2. Ingest a corpus (one-shot)

Ingestion needs the embedding backend (`EMBED_PROVIDER`, Ollama by default).
Run it once per corpus; the index lands in the `rag_index` volume.

```bash
# Embeddings from the bundled Ollama service
docker compose --profile local-llm up -d ollama
docker compose --profile local-llm exec ollama ollama pull nomic-embed-text
docker compose --profile tools --profile local-llm run --rm ingest --corpus ai_act_sections
docker compose --profile tools --profile local-llm run --rm ingest --corpus filings_sections
```

Arguments after `ingest` replace the default `--corpus ai_act_sections`, because
the service's entrypoint is `python cli.py ingest`. There is no longer an
`ai_act` corpus: it held the same text as `ai_act_sections` (see
`data/ai_act_sections/README.md`). `data/filings/` is a provenance folder holding
the raw SEC submission and is not meant to be ingested on its own.

## 3. Run the app

Open http://localhost:8501 once the container is healthy.

**Hosted LLM (Groq / OpenRouter), embeddings from Ollama on your host**

```bash
OLLAMA_URL_IN_DOCKER=http://host.docker.internal:11434 docker compose up -d app
```

**Everything local (bundled Ollama for embeddings and generation)**

```bash
docker compose --profile local-llm up -d
docker compose --profile local-llm exec ollama ollama pull nomic-embed-text
docker compose --profile local-llm exec ollama ollama pull llama3.2:3b
```

Set `LLM_PROVIDER=ollama` in `.env` to generate locally; otherwise Ollama is only
used as the last fallback.

An Ollama running inside WSL listens on `127.0.0.1` by default, which containers
cannot reach through `host.docker.internal`. Either use the bundled `ollama`
service (above), or start the WSL one with `OLLAMA_HOST=0.0.0.0`.

**API and React UI (profile `api`)**

```bash
cd web && npm install && npm run build && cd ..   # on the host: the image has no Node
docker compose build                              # copies web/dist into the image
docker compose --profile api up -d api
```

Open http://localhost:8002 (the API serves the built UI). Without `web/dist`
only the JSON API is served. The API loads each index once and keeps it in
memory, so after an `ingest` run restart it: `docker compose --profile api restart api`.

**Ask from the CLI inside the container**

```bash
docker compose run --rm app python cli.py ask --corpus ai_act_sections "Which AI practices are prohibited?"
```

## How the Ollama URL is wired

`.env` usually says `OLLAMA_URL=http://localhost:11434`, which is right on the
host but wrong in a container (`localhost` is the container itself). Compose
therefore overrides it:

| Where Ollama runs             | Value used                                   |
|-------------------------------|----------------------------------------------|
| Bundled `ollama` service      | `http://ollama:11434` (default, service DNS) |
| On the host machine           | `OLLAMA_URL_IN_DOCKER=http://host.docker.internal:11434` |
| Elsewhere                     | `OLLAMA_URL_IN_DOCKER=http://<host>:11434`   |

`extra_hosts: host.docker.internal:host-gateway` makes the host name resolve on
Linux too; Docker Desktop provides it natively.

## Why it is built this way

- **`python:3.12-slim`**: small, and every dependency ships as a wheel, so no
  build toolchain is needed.
- **Dependencies before code**: `requirements.txt` is installed in its own
  layer; editing Python files only rebuilds the final copy layer.
- **Non-root user (uid 10001)**: limits what a compromised dependency could do.
  `index/` and `logs/` are created and owned by that user so fresh named volumes
  inherit write access.
- **`.dockerignore`**: excludes `.env`, `.venv`, `index/`, `cache/`, `logs/`,
  `.git` and `__pycache__`. Keys never enter the image or the build context, and
  indexes are runtime state rather than build output.
- **Corpora mounted read-only**: `./data` is input. Mounting it `:ro` means neither
  the app nor ingestion can modify the documents answers are grounded on, and
  adding a document needs no rebuild, only a re-ingest.
- **Named volumes for `index` and `logs`**: indexes survive container rebuilds,
  and the embedding cache (`index/<corpus>/embed_cache.json`) comes with them, so
  re-ingesting an unchanged corpus re-embeds nothing.
- **`HEALTHCHECK` on `/_stcore/health`**: Streamlit's own liveness endpoint. It is
  probed with Python's `urllib` because the slim image has no `curl`.
- **Exec-form `CMD`**: Streamlit runs as PID 1 and receives `SIGTERM` directly, so
  `docker stop` is immediate instead of waiting for the kill timeout.
- **Ollama behind a profile**: most runs use a hosted LLM, so the multi-GB local
  model server only starts when asked for (`--profile local-llm`). The app's
  `depends_on ... condition: service_healthy` with `required: false` waits for
  Ollama when it is part of the run and is ignored otherwise.
- **Ollama port not published**: only the app and ingest containers talk to it,
  over the Compose network.
- **Ingest as a one-shot `tools` service**: building an index is an explicit,
  occasional step (as with `python cli.py ingest`), not something the web
  container should do at start-up.
