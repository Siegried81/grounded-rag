"""Central configuration: all tunables and provider settings in one place.

Values are read from the environment (via a .env file) with sensible defaults so
the project runs out of the box with local Ollama. Keeping every knob here — chunk
sizes, the retrieval threshold, provider endpoints — means no module hardcodes a
magic number, and the one setting that changes what an answer means (the relevance
threshold below which the assistant refuses) is visible and documented.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"       # one subfolder per corpus, e.g. data/ai_act
INDEX_DIR = ROOT / "index"     # persisted vector stores, one subfolder per corpus

# --- Chunking ----------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))        # characters per passage
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))  # characters shared between neighbours

# --- Retrieval ---------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "5"))
# Minimum cosine similarity for a passage to count as evidence. Below this for
# every candidate, the assistant refuses rather than answering ungrounded. This
# gate is always applied to the DENSE cosine score, so it keeps the same meaning
# whether retrieval runs in dense or hybrid mode.
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.35"))
# "hybrid" fuses dense (cosine) and BM25 (keyword) rankings; "dense" uses vectors
# only. Hybrid catches exact terms (identifiers, acronyms) that embeddings miss.
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid").strip().lower()  # hybrid | dense
# retrieve() treats anything that is not "dense" as hybrid, so a typo would run
# hybrid while the UIs display the typo. Fail at start-up instead.
if RETRIEVAL_MODE not in ("hybrid", "dense"):
    raise ValueError(f"RETRIEVAL_MODE must be 'hybrid' or 'dense', got {RETRIEVAL_MODE!r}")
# Size of the candidate pool pulled from each channel before fusion/MMR/top-k.
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "20"))
# Maximal Marginal Relevance reorders the final set to drop near-duplicate
# passages; lambda 1.0 is pure relevance, 0.0 pure diversity.
USE_MMR = os.getenv("USE_MMR", "true").lower() == "true"
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.6"))

# --- Chunking / ingestion ----------------------------------------------------
# "smart" packs whole sentences (better passages & citations); "char" is the
# original fixed-character splitter, kept as a fallback.
CHUNKER = os.getenv("CHUNKER", "smart")  # smart | char
USE_EMBED_CACHE = os.getenv("USE_EMBED_CACHE", "true").lower() == "true"

# --- Verification & logging --------------------------------------------------
# Minimum lexical grounding for an answer to pass the (cheap, offline) faithfulness
# check. It is a proxy, not entailment, so the bar is deliberately modest.
VERIFY_MIN_GROUNDING = float(os.getenv("VERIFY_MIN_GROUNDING", "0.30"))
QUERY_LOG_PATH = ROOT / "logs" / "queries.jsonl"

# --- Answer language ---------------------------------------------------------
# Language the UIs ask answers to be written in. "auto" adds nothing to the
# prompt (the model usually follows the language of the question and sources);
# a named language adds one instruction sentence (see rag/answer.py). Only the
# UIs read this default: the evaluation scripts call answer_question without a
# language, so their prompts, and the LLM cache built on them, stay unchanged.
SUPPORTED_ANSWER_LANGUAGES = ("auto", "en", "fr", "nl")
DEFAULT_ANSWER_LANGUAGE = os.getenv("ANSWER_LANGUAGE", "en").strip().lower()
if DEFAULT_ANSWER_LANGUAGE not in SUPPORTED_ANSWER_LANGUAGES:
    DEFAULT_ANSWER_LANGUAGE = "en"
# Languages the interface texts (examples, explanations, limitations) exist in.
SUPPORTED_UI_LANGUAGES = ("en", "fr")

# --- Embedding backend -------------------------------------------------------
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "ollama")  # ollama | hosted | fake
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
HOSTED_EMBED_BASE_URL = os.getenv("HOSTED_EMBED_BASE_URL", "")
HOSTED_EMBED_MODEL = os.getenv("HOSTED_EMBED_MODEL", "")
HOSTED_EMBED_API_KEY = os.getenv("HOSTED_EMBED_API_KEY", "")

# --- Generation backend ------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # groq | openrouter | ollama
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Up to five Groq keys (GROQ_API_KEY, GROQ_API_KEY_2 .. _5). The client rotates to
# the next key when one is rate-limited or fails, which multiplies the free-tier
# quota across keys before falling back to another provider.
GROQ_API_KEYS = [
    k
    for k in (
        GROQ_API_KEY,
        os.getenv("GROQ_API_KEY_2", ""),
        os.getenv("GROQ_API_KEY_3", ""),
        os.getenv("GROQ_API_KEY_4", ""),
        os.getenv("GROQ_API_KEY_5", ""),
    )
    if k
]
# Groq retired its Llama 3.x models; gpt-oss-120b is the strongest model it serves.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2:3b")

# Groq's free tier limits tokens per minute per key. When every key answers 429,
# wait for the window Groq announces (capped here) and retry once before falling
# back to the next provider: a short wait keeps answers on the configured model
# instead of silently switching providers mid-evaluation. 0 = fall back at once.
GROQ_MAX_WAIT_S = float(os.getenv("GROQ_MAX_WAIT_S", "65"))

# Successful completions are cached on disk by an exact hash of the request
# (provider order, models, system prompt, prompt). Re-asking the same question
# over the same passages, or re-running an evaluation, then costs no quota.
LLM_CACHE = os.getenv("LLM_CACHE", "1") == "1"
LLM_CACHE_DIR = INDEX_DIR / "llm_cache"

# Ordered fallback chain used by rag/llm.py. First provider is the configured one.
LLM_FALLBACK_ORDER = [
    p for p in [LLM_PROVIDER, "groq", "openrouter", "ollama"]
    if p and p not in ("",)
]


def corpus_dir(corpus: str) -> Path:
    """Folder holding the raw documents for a corpus (e.g. data/ai_act)."""
    return DATA_DIR / corpus


def index_path(corpus: str) -> Path:
    """Folder where the built vector store for a corpus is persisted."""
    return INDEX_DIR / corpus


def bm25_path(corpus: str) -> Path:
    """File holding the BM25 keyword index for a corpus, next to its vectors."""
    return index_path(corpus) / "bm25.json"


def embed_cache_path(corpus: str) -> Path:
    """File holding the embedding cache for a corpus, next to its index."""
    return index_path(corpus) / "embed_cache.json"
