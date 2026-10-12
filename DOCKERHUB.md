# grounded-rag - cite or refuse

## Quick reference

- **Source code, Dockerfile, issues:** https://github.com/Siegried81/grounded-rag
- **Maintained by:** Siegried81
- **Also on:** GitHub Container Registry, `ghcr.io/siegried81/<image>`
- **Architecture:** `linux/amd64`

## Supported tags

- `latest`: the current `main` branch
- `X.Y.Z` and `X`: released versions (git tags `vX.Y.Z`), the ones to pin for a stable deployment
- `sha-xxxxxxx`: one specific commit

## What is it?

Retrieval-augmented question answering where every claim cites a passage you can read, or the assistant refuses. Hybrid BM25 + dense retrieval, citation verification, Streamlit and FastAPI UIs.

Images from this repository:

- `grounded-rag`: Streamlit UI on 8501; same image runs the API on 8002

## How to use this image

```bash
docker run -d -p 8501:8501 -e GROQ_API_KEY=... siegried/grounded-rag
# open http://localhost:8501

# build the index once (persisted in the named volume)
docker run --rm -v rag_index:/app/index -e GROQ_API_KEY=... siegried/grounded-rag \
  python cli.py ingest --corpus ai_act_sections
```

## Environment variables

None are required to start the UI. Secrets are **never** baked into the image: pass
them at run time with `-e NAME=value` or `--env-file .env` (see `.env.example` in the repo).

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `groq`, `openrouter` or `ollama` |
| `GROQ_API_KEY` | Groq key (free tier is enough) |
| `OPENROUTER_API_KEY` | OpenRouter key |
| `OLLAMA_URL` | Ollama server, e.g. `http://host.docker.internal:11434` |
| `EMBED_PROVIDER` | embedding backend |

## Volumes

- `/app/index`: vector + BM25 index (keep it in a named volume)
- `/app/logs`: query logs
- `/app/data`: your documents, mount read-only

## License

See the LICENSE file in the source repository, if any. As with every image, it is
your responsibility to check that your use complies with the licenses of the
software it contains.
