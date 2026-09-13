"""Token counting.

Every budget in this system — working memory, assembled context, compaction
triggers — is expressed in tokens rather than characters or message counts,
because tokens are what the model actually pays for.

tiktoken when available; otherwise a word-based estimate. The estimate is
deliberately slightly pessimistic: a budget that overshoots truncates a prompt
in production, while one that undershoots merely wastes a little room.
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text))
    # ~1.3 tokens per whitespace word, rounded up.
    return int(len(text.split()) * 1.3) + 1


def truncate_to_tokens(text: str, budget: int) -> str:
    """Cut text to fit a token budget, on a word boundary."""
    if budget <= 0:
        return ""
    if count_tokens(text) <= budget:
        return text
    words = text.split()
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(" ".join(words[:mid])) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo])
