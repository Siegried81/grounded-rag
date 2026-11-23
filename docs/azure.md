# Running grounded-rag on Azure

The default setup runs on Groq, Ollama and an in-memory NumPy index. This page
covers the Azure path: Azure OpenAI for answers and embeddings, Azure AI Search
for dense retrieval, RAGAS scores with an Azure judge, and a meeting-action
extractor. Every piece is opt-in. With no Azure setting, nothing here runs and
no published number changes.

## What each setting switches on

| Setting | Effect | Code |
|---|---|---|
| `LLM_PROVIDER=azure_openai` | answers come from your chat deployment | `rag/llm.py` |
| `EMBED_PROVIDER=azure_openai` | passages and questions are embedded by your embeddings deployment | `rag/embed.py` |
| `VECTOR_INDEX=azure_search` | an Azure AI Search index proposes the candidate passages | `rag/azure_search_index.py` |
| `scripts/run_ragas_eval.py` | RAGAS faithfulness, answer relevancy and context precision | |
| `scripts/run_action_eval.py` | action extraction measured on `eval/actions/` | `rag/actions.py` |

Azure OpenAI is called through the resource's OpenAI-compatible v1 API
(`<endpoint>/openai/v1`). The "model" is your deployment name, so
`AZURE_OPENAI_DEPLOYMENT` is whatever you called it in the portal.

Two things stay the same whatever you switch on:

- **The refusal gate.** Azure AI Search only proposes passages. The store scores
  them with its own cosine, the same formula and vectors as the default path, so
  `SCORE_THRESHOLD` keeps its meaning. A passage the index misses is missing,
  but no score moves.
- **The embedding space.** The store records the embedding model with every
  vector. An index built with Ollama embeddings will not be queried with Azure
  embeddings: rebuild it (`python cli.py ingest --corpus ...`).

## Before you start: cost

1. Get credits. Azure for Students gives a starting credit without a card.
   Check the current amount and conditions on the sign-up page.
2. Set a budget alert first. In the portal: Cost Management, Budgets, Add. Pick
   a small monthly amount and an email alert at 50 % and 90 %.
3. Pick a small, non-reasoning chat model (a "mini" model). The client sends
   `temperature: 0`, which reasoning models refuse.
4. Azure AI Search has a free tier: three indexes, enough for the two corpora.
5. When you are done, delete the resource group. That stops every cost at once.

Azure OpenAI is not open on every student subscription. If the portal refuses
to create it, try Azure AI Foundry, which serves the same models, and use the
endpoint it gives you.

## Local run against Azure

Create the resources in the portal (or with the script below), then add to `.env`:

```bash
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT=chat
AZURE_OPENAI_EMBED_DEPLOYMENT=embed
AZURE_SEARCH_ENDPOINT=https://<service>.search.windows.net
AZURE_SEARCH_API_KEY=...          # an ADMIN key: the backend creates the index
```

Then:

```bash
EMBED_PROVIDER=azure_openai python cli.py ingest --corpus ai_act_sections
EMBED_PROVIDER=azure_openai python cli.py ingest --corpus filings_sections
EMBED_PROVIDER=azure_openai python cli.py ingest --corpus filings_statements
EMBED_PROVIDER=azure_openai VECTOR_INDEX=azure_search LLM_PROVIDER=azure_openai \
  streamlit run app.py
```

Ingesting with Azure embeddings writes over the local index of the same name.
Keep a copy of `index/` if you want to go back to Ollama without re-embedding.

## Measuring it

Run each evaluation with the provider pinned (`LLM_FALLBACK=false`) and the
cache off (`LLM_CACHE=0`), so one run measures one model. Report Azure as its
own row next to the Groq figures, never in their place.

```bash
# 1. answers, with the project's own metrics (key facts, citations, refusals)
LLM_PROVIDER=azure_openai LLM_FALLBACK=false LLM_CACHE=0 \
  python scripts/run_answer_eval.py --out logs/answer_eval_azure.jsonl

# 2. RAGAS on the same answers: a dry run first prints the number of judge calls
python scripts/run_ragas_eval.py --results logs/answer_eval_azure.jsonl
python scripts/run_ragas_eval.py --results logs/answer_eval_azure.jsonl --yes

# 3. action extraction on the six labelled meetings
LLM_PROVIDER=azure_openai LLM_FALLBACK=false LLM_CACHE=0 \
  python scripts/run_action_eval.py
```

RAGAS needs `ragas` and `openai` (`requirements-optional.txt`). Its dependencies
did not build on Python 3.14 in November 2026; use Python 3.12, as the Docker
image does.

How to read the RAGAS scores:

- They only cover answerable questions that were answered. RAGAS does not know
  "cite or refuse", so a correct refusal would score as a failure. Refusals stay
  measured by `run_answer_eval.py`.
- Context precision is the variant without a reference answer: the eval sets
  have key facts and gold files, not reference answers. Retrieval recall against
  the gold files is `scripts/run_eval.py`.
- A judge does not give the same score twice, so each row is scored three times
  and the summary shows the spread. A judge from the same deployment that wrote
  the answers grades its own work: `--judge-deployment` picks another one.

## Action extraction

`rag/actions.py` reads a meeting transcript and returns the actions: title,
owner, due date, the sentence it comes from, and a confidence. It is one LLM
call with a fixed JSON schema, then checks in plain Python:

- the quoted sentence must be in the transcript;
- the deadline words must be in the quoted sentence, and the weekday, day or
  month they name must match the date;
- the owner is matched to a directory (a JSON file here, Microsoft Graph in a
  company). The model never sees the directory. A name that matches nobody, or
  two people, gets no owner;
- anything that fails a check is flagged for a person to review, never fixed
  by guessing.

```bash
python scripts/extract_actions.py eval/actions/transcripts/m2_comite_controle_fr.txt \
  --meeting-date 2026-03-10
```

The six meetings in `eval/actions/` are invented. They mix French, Dutch and
English and include the hard cases on purpose: two people called Thomas, a
person missing from the directory, a cancelled task, a task already done, a
prompt injection, and a meeting with no action at all.

## Deploying on Azure Container Apps

`deploy/azure/provision.sh` creates the Azure OpenAI resource and its two
deployments, a free AI Search service, a Key Vault holding both keys, and a
Container Apps app running the published image. The app reads the keys from the
vault through its managed identity, so no key sits in an environment variable
or in the script output.

```bash
az login
bash deploy/azure/provision.sh
# ...
az group delete -n rg-grounded-rag --yes   # stop every cost
```

The app scales to zero. Each cold start builds the indexes again, like the
Render deployment, which takes a few minutes. The image must contain the Azure
code, so publish a new image after this branch is merged.
