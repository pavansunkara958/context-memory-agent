"""Configuration.

One principle drives this file: **the expensive thing should be the optional
thing.** Embedding, indexing, retrieval, reranking, context optimisation and
the entire evaluation suite run locally with no API key. Only free-text answer
generation can call a hosted model, and `PROVIDER=offline` replaces even that
with an extractive answerer.

That is not only about cost. It means retrieval quality can be measured
deterministically and repeatably — the same query returns the same ranking
every run, so an ablation table means something. A pipeline whose metrics
depend on a sampled model response cannot be ablated honestly.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


# --- Generation -------------------------------------------------------------
PROVIDER = os.getenv("PROVIDER", "offline").lower()
MODEL_GENERATE = os.getenv("MODEL_GENERATE", "gpt-5-mini")

GATEWAY_BASE_URL = os.getenv("GATEWAY_BASE_URL", "")
GATEWAY_AUTH_MODE = os.getenv("GATEWAY_AUTH_MODE", "bearer").lower()
GATEWAY_HEADER_NAME = os.getenv("GATEWAY_HEADER_NAME", "X-Api-Key")
GATEWAY_KEY = os.getenv("GATEWAY_KEY", "")
GATEWAY_MODEL = os.getenv("GATEWAY_MODEL", "")

# --- Local models -----------------------------------------------------------
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "auto").lower()
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
USE_RERANKER = os.getenv("USE_RERANKER", "1") == "1"

# --- Chunking ---------------------------------------------------------------
# 110 tokens, not 180. At 180 every document in this corpus fits in a single
# chunk, so overlap, ordinals and near-duplicate removal are all dead code that
# still passes its unit tests. Sizing chunks below the typical document length
# is what makes the retrieval granularity real.
CHUNK_TOKENS = _int("CHUNK_TOKENS", 110)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 25)

# --- Retrieval --------------------------------------------------------------
RETRIEVE_K = _int("RETRIEVE_K", 20)      # candidates from each retriever
RERANK_K = _int("RERANK_K", 6)           # survivors after reranking
SIM_FLOOR = _float("SIM_FLOOR", 0.25)    # dense similarity floor
RRF_K = _int("RRF_K", 60)                # reciprocal rank fusion constant

# BM25 returns a long tail of documents sharing a single common token. A hit is
# only treated as a real lexical match if it scores at least this fraction of
# the best BM25 score for the same query — relative, not absolute, because BM25
# scores are not comparable across queries.
SPARSE_FLOOR_RATIO = _float("SPARSE_FLOOR_RATIO", 0.35)

# --- Context optimisation ---------------------------------------------------
DEDUP_THRESHOLD = _float("DEDUP_THRESHOLD", 0.92)
CONTEXT_BUDGET_TOKENS = _int("CONTEXT_BUDGET_TOKENS", 2000)
WORKING_MEMORY_TOKENS = _int("WORKING_MEMORY_TOKENS", 800)
SESSION_COMPACT_AFTER = _int("SESSION_COMPACT_AFTER", 8)

# --- Storage ----------------------------------------------------------------
DATA_DIR = ROOT / os.getenv("DATA_DIR", "data")
CHROMA_DIR = DATA_DIR / "chroma"
SESSION_DB = DATA_DIR / "sessions.sqlite3"
NUMPY_STORE_DIR = DATA_DIR / "numpy_store"


def ensure_dirs() -> None:
    for d in (DATA_DIR, CHROMA_DIR, NUMPY_STORE_DIR, ROOT / "logs"):
        d.mkdir(parents=True, exist_ok=True)


def generation_client():
    """Return (client, model) for answer generation, or (None, None) offline."""
    if PROVIDER == "offline":
        return None, None

    from openai import OpenAI

    if PROVIDER == "gateway":
        if not GATEWAY_BASE_URL or not GATEWAY_KEY:
            raise SystemExit(
                "\nPROVIDER=gateway needs GATEWAY_BASE_URL and GATEWAY_KEY in .env.\n"
            )
        if GATEWAY_AUTH_MODE == "header":
            # A gateway that authenticates on a custom header may forward an
            # Authorization header verbatim to the upstream provider, which
            # then rejects it. Remove it on the way out.
            import httpx

            def _strip_auth(request: "httpx.Request") -> None:
                request.headers.pop("authorization", None)

            http_client = httpx.Client(
                timeout=httpx.Timeout(120.0, connect=15.0),
                event_hooks={"request": [_strip_auth]},
            )
            client = OpenAI(
                base_url=GATEWAY_BASE_URL,
                api_key="unused",  # stripped before the wire
                default_headers={
                    GATEWAY_HEADER_NAME: GATEWAY_KEY,
                    "Accept-Encoding": "identity",
                },
                http_client=http_client,
            )
        else:
            client = OpenAI(base_url=GATEWAY_BASE_URL, api_key=GATEWAY_KEY)
        return client, (GATEWAY_MODEL or MODEL_GENERATE)

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("\nPROVIDER=openai needs OPENAI_API_KEY in .env.\n")
    return OpenAI(), MODEL_GENERATE


def describe() -> str:
    gen = PROVIDER if PROVIDER != "gateway" else f"gateway:{GATEWAY_MODEL}"
    return (f"provider={gen}  embed={EMBED_BACKEND}  "
            f"rerank={'on' if USE_RERANKER else 'off'}")
