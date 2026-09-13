"""Embeddings and reranking — both local, both free.

Two backends, chosen at runtime:

  sentence-transformers   all-MiniLM-L6-v2, 384 dimensions. Real semantic
                          similarity. ~90MB downloaded once, then offline.
  hash                    A deterministic hashing bag-of-words embedder with
                          no model and no download.

The hash backend is not a toy. It makes the entire pipeline — indexing,
retrieval, fusion, dedup, evaluation — runnable and testable in a CI container
with no network, which is what lets `scripts/mock_test.py` assert real
behaviour in under a second. It is lexical, so it scores worse on paraphrase
queries, and the evaluation harness reports which backend produced a number so
the two are never silently compared.

Reranking is a cross-encoder: it reads (query, chunk) *together* rather than
comparing two independently-produced vectors, which is why it resolves cases
bi-encoder similarity gets wrong. It is the expensive step, so it only ever
sees the top candidates, never the corpus.
"""
from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache

import numpy as np

from .config import EMBED_BACKEND, EMBED_MODEL, RERANK_MODEL

_TOKEN_RE = re.compile(r"[a-z0-9]+")
HASH_DIM = 512


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Hash backend
# ---------------------------------------------------------------------------
def _hash_vector(text: str, dim: int = HASH_DIM) -> np.ndarray:
    """Deterministic bag-of-words embedding with sublinear term weighting."""
    vec = np.zeros(dim, dtype=np.float32)
    counts: dict[str, int] = {}
    for tok in tokenize(text):
        counts[tok] = counts.get(tok, 0) + 1
    for tok, n in counts.items():
        h = hashlib.blake2b(tok.encode(), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "big") % dim
        sign = 1.0 if h[4] & 1 else -1.0
        vec[idx] += sign * (1.0 + math.log(n))
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _sentence_transformer():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def backend_name() -> str:
    """Which embedding backend is actually in use."""
    if EMBED_BACKEND == "hash":
        return "hash"
    try:
        _sentence_transformer()
        return "sentence-transformers"
    except Exception:
        # No network, no model cache, or the package is absent. The pipeline
        # still runs; it just retrieves lexically.
        return "hash"


def embed(texts: list[str]) -> np.ndarray:
    """Embed a batch. Returns an L2-normalised (n, dim) float32 array."""
    if not texts:
        return np.zeros((0, HASH_DIM), dtype=np.float32)

    if backend_name() == "sentence-transformers":
        model = _sentence_transformer()
        vecs = model.encode(texts, normalize_embeddings=True,
                            show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float32)

    return np.vstack([_hash_vector(t) for t in texts])


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity. Inputs are normalised, so this is a dot product —
    but normalise defensively, because a zero vector would otherwise return nan
    and poison every downstream comparison silently."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL)


@lru_cache(maxsize=1)
def reranker_name() -> str:
    try:
        _cross_encoder()
        return "cross-encoder"
    except Exception:
        return "lexical-overlap"


def rerank_scores(query: str, passages: list[str]) -> list[float]:
    """Score each passage against the query. Higher is better.

    The cross-encoder attends over the query and passage jointly. The fallback
    is weighted term overlap with an IDF-ish penalty on common words — crude,
    but monotonic in the right direction, which is all the ordering needs.
    """
    if not passages:
        return []

    if reranker_name() == "cross-encoder":
        model = _cross_encoder()
        pairs = [(query, p) for p in passages]
        return [float(s) for s in model.predict(pairs, show_progress_bar=False)]

    q_terms = set(tokenize(query))
    if not q_terms:
        return [0.0] * len(passages)

    # Document frequency over the candidate set only — cheap, and enough to
    # stop ubiquitous words from dominating the overlap score.
    df: dict[str, int] = {}
    tokenised = [set(tokenize(p)) for p in passages]
    for terms in tokenised:
        for t in terms & q_terms:
            df[t] = df.get(t, 0) + 1

    n = len(passages)
    scores = []
    for terms in tokenised:
        score = 0.0
        for t in terms & q_terms:
            score += math.log(1 + n / (1 + df.get(t, 0)))
        scores.append(score / (len(q_terms) ** 0.5))
    return scores
