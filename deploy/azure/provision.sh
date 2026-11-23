#!/usr/bin/env bash
# Provision the Azure path of grounded-rag and deploy the Streamlit UI on it.
#
# Creates, in one resource group:
#   1. an Azure OpenAI resource with a chat and an embeddings deployment,
#   2. an Azure AI Search service on the free tier,
#   3. a Key Vault holding the two keys,
#   4. a Container Apps app running the published image, which reads the keys
#      from the vault through its managed identity (no key in an env var or in
#      this script's output).
#
# Same start-up as render.yaml: both indexes are built when the container
# starts, then Streamlit comes up. With scale-to-zero, every cold start embeds
# the corpora again (about 900 passages, a fraction of a cent on a small
# embeddings model) and the Azure AI Search index is rebuilt only if the
# vectors changed (see rag/azure_search_index.py).
#
# Usage: az login, then  bash deploy/azure/provision.sh
# Every name below can be overridden from the environment. Model names and
# versions change over time: list what your region offers with
#   az cognitiveservices model list -l "$LOCATION" -o table
# Set a budget alert in the portal before running this (docs/azure.md).
set -euo pipefail

SUFFIX="${SUFFIX:-$RANDOM}"                     # keeps global names unique
LOCATION="${LOCATION:-swedencentral}"
RG="${RG:-rg-grounded-rag}"
OPENAI_NAME="${OPENAI_NAME:-aoai-grounded-rag-$SUFFIX}"
SEARCH_NAME="${SEARCH_NAME:-srch-grounded-rag-$SUFFIX}"
VAULT_NAME="${VAULT_NAME:-kv-grag-$SUFFIX}"
ENV_NAME="${ENV_NAME:-cae-grounded-rag}"
APP_NAME="${APP_NAME:-grounded-rag}"
IMAGE="${IMAGE:-docker.io/siegried/grounded-rag:latest}"
CHAT_MODEL="${CHAT_MODEL:-gpt-4.1-mini}"
CHAT_VERSION="${CHAT_VERSION:-2025-04-14}"
EMBED_MODEL="${EMBED_MODEL:-text-embedding-3-small}"
EMBED_VERSION="${EMBED_VERSION:-1}"
CAPACITY="${CAPACITY:-10}"                      # thousands of tokens per minute

echo "== resource group $RG in $LOCATION"
az group create -n "$RG" -l "$LOCATION" -o none

echo "== Azure OpenAI $OPENAI_NAME"
az cognitiveservices account create -g "$RG" -n "$OPENAI_NAME" -l "$LOCATION" \
  --kind OpenAI --sku S0 --custom-domain "$OPENAI_NAME" -o none
az cognitiveservices account deployment create -g "$RG" -n "$OPENAI_NAME" \
  --deployment-name chat --model-name "$CHAT_MODEL" --model-version "$CHAT_VERSION" \
  --model-format OpenAI --sku-name GlobalStandard --sku-capacity "$CAPACITY" -o none
az cognitiveservices account deployment create -g "$RG" -n "$OPENAI_NAME" \
  --deployment-name embed --model-name "$EMBED_MODEL" --model-version "$EMBED_VERSION" \
  --model-format OpenAI --sku-name GlobalStandard --sku-capacity "$CAPACITY" -o none
OPENAI_ENDPOINT=$(az cognitiveservices account show -g "$RG" -n "$OPENAI_NAME" \
  --query properties.endpoint -o tsv)

echo "== Azure AI Search $SEARCH_NAME (free tier)"
az search service create -g "$RG" -n "$SEARCH_NAME" -l "$LOCATION" --sku free -o none
SEARCH_ENDPOINT="https://$SEARCH_NAME.search.windows.net"

echo "== Key Vault $VAULT_NAME"
az keyvault create -g "$RG" -n "$VAULT_NAME" -l "$LOCATION" \
  --enable-rbac-authorization true -o none
VAULT_ID=$(az keyvault show -n "$VAULT_NAME" --query id -o tsv)
# Writing secrets needs a data-plane role, even for the vault's creator.
az role assignment create --assignee "$(az ad signed-in-user show --query id -o tsv)" \
  --role "Key Vault Secrets Officer" --scope "$VAULT_ID" -o none
sleep 30   # role assignments take a moment to apply
# Keys go from one az call straight into the vault, never printed.
az keyvault secret set --vault-name "$VAULT_NAME" -n azure-openai-key -o none \
  --value "$(az cognitiveservices account keys list -g "$RG" -n "$OPENAI_NAME" --query key1 -o tsv)"
az keyvault secret set --vault-name "$VAULT_NAME" -n azure-search-key -o none \
  --value "$(az search admin-key show -g "$RG" --service-name "$SEARCH_NAME" --query primaryKey -o tsv)"

echo "== Container Apps $APP_NAME"
az containerapp env create -g "$RG" -n "$ENV_NAME" -l "$LOCATION" -o none
az containerapp create -g "$RG" -n "$APP_NAME" --environment "$ENV_NAME" \
  --image "$IMAGE" --target-port 8501 --ingress external \
  --min-replicas 0 --max-replicas 1 --cpu 1 --memory 2Gi \
  --system-assigned -o none
PRINCIPAL=$(az containerapp show -g "$RG" -n "$APP_NAME" --query identity.principalId -o tsv)
az role assignment create --assignee-object-id "$PRINCIPAL" --assignee-principal-type ServicePrincipal \
  --role "Key Vault Secrets User" --scope "$VAULT_ID" -o none
sleep 30
VAULT_URI="https://$VAULT_NAME.vault.azure.net"
az containerapp secret set -g "$RG" -n "$APP_NAME" -o none --secrets \
  "azure-openai-key=keyvaultref:$VAULT_URI/secrets/azure-openai-key,identityref:system" \
  "azure-search-key=keyvaultref:$VAULT_URI/secrets/azure-search-key,identityref:system"
az containerapp update -g "$RG" -n "$APP_NAME" -o none \
  --command sh -c "python cli.py ingest --corpus ai_act_sections && python cli.py ingest --corpus filings_sections && python cli.py ingest --corpus filings_statements && exec python -m streamlit run app.py --server.port=8501 --server.address=0.0.0.0" \
  --set-env-vars \
    LLM_PROVIDER=azure_openai LLM_FALLBACK=false \
    EMBED_PROVIDER=azure_openai VECTOR_INDEX=azure_search \
    AZURE_OPENAI_ENDPOINT="$OPENAI_ENDPOINT" \
    AZURE_OPENAI_DEPLOYMENT=chat AZURE_OPENAI_EMBED_DEPLOYMENT=embed \
    AZURE_SEARCH_ENDPOINT="$SEARCH_ENDPOINT" \
    AZURE_OPENAI_API_KEY=secretref:azure-openai-key \
    AZURE_SEARCH_API_KEY=secretref:azure-search-key

echo "== done"
echo "app:    https://$(az containerapp show -g "$RG" -n "$APP_NAME" --query properties.configuration.ingress.fqdn -o tsv)"
echo "delete everything: az group delete -n $RG --yes"
