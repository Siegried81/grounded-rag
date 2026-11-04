# grounded-rag on Kubernetes

A learning port of `docker-compose.yml` to Kubernetes. The Streamlit app (8501),
the FastAPI API (8002), the one-shot index builder as a Job, and an Ollama
sidecar (11434).

**Is k8s needed here?** No. A single RAG app belongs on Docker Compose or a PaaS.
This port is to practise the objects: Deployment, Service, **Job**, PVC, Secret.

## What maps to what

| Compose | Kubernetes |
|---|---|
| `app` service (Streamlit 8501) | Deployment + Service `rag-app` |
| `api` service (uvicorn 8002) | Deployment + Service `rag-api` |
| `ingest` one-shot (`restart: no`) | **Job** `rag-ingest` |
| `ollama` service (11434) | Deployment + Service `rag-ollama` |
| `.env` file | Secret `rag-secrets` |
| named volume `rag_index` | PVC `rag-index` |
| named volume `rag_logs` | PVC `rag-logs` |
| named volume `ollama_models` | PVC `rag-ollama-models` |

The one-shot `ingest` becomes a **Job**, not a Deployment: a Job runs a task to
completion and stops, which is exactly the "build the index once" shape. A
Deployment would restart it forever.

Note: Compose mounted `./data` read-only for the corpora. Here the data is assumed
baked into the image; if you keep corpora outside, add a read-only PVC and mount
it at `/app/data`.

## Run it locally with kind

```bash
kind create cluster --name rag
docker build -t siegried/grounded-rag:latest ..
kind load docker-image siegried/grounded-rag:latest --name rag

kubectl create secret generic rag-secrets --from-env-file=../.env

# apply everything at once; the Job builds the index onto the shared rag-index
# volume. The app and api come up alongside it -- their health probes don't
# check the index -- so wait for the Job before sending traffic (queries error
# until the index exists)
kubectl apply -f grounded-rag.yaml
kubectl wait --for=condition=complete job/rag-ingest --timeout=600s
kubectl get pods

kubectl port-forward svc/rag-api 8002:8002   # http://localhost:8002
kubectl port-forward svc/rag-app 8501:8501   # http://localhost:8501

kind delete cluster --name rag
```

## Check the manifests without a cluster

```bash
kubectl apply --dry-run=client -f grounded-rag.yaml
```
