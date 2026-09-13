"""Structure-aware chunking.

Two decisions do most of the work here.

**Split on paragraph boundaries, not a fixed token stride.** A chunk that
begins mid-sentence retrieves badly, because its embedding is an average of
half an idea and the start of another. Packing whole paragraphs up to a budget
keeps each chunk about one thing.

**Prepend the document's identity to every chunk.** A chunk reading "the buffer
now applies backpressure instead of discarding the oldest records" is nearly
useless on its own — backpressure in *what*, since *when*? Carrying
"REL-5.0 · Helios firmware 5.0 release notes (release, 2026-02-26)" into the
embedded text means the chunk answers version-qualified questions, and the
citation is already attached when it reaches the model.

Overlap exists for the case a paragraph split lands mid-procedure: the tail of
the previous chunk is repeated so a step and its warning stay together.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import CHUNK_OVERLAP, CHUNK_TOKENS
from .corpus import Document
from .tokens import count_tokens


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    kind: str
    date: str
    text: str              # the body alone
    ordinal: int           # position within the document
    embed_text: str = ""   # header + body; what actually gets embedded
    meta: dict = field(default_factory=dict)

    def citation(self) -> str:
        return f"{self.doc_id}#{self.ordinal}"

    def header(self) -> str:
        return f"{self.doc_id} · {self.title} ({self.kind}, {self.date})"


def _normalise(text: str) -> str:
    """Collapse the source's indentation without losing paragraph breaks."""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return "\n\n".join(" ".join(p.split()) for p in paragraphs)


def _paragraphs(text: str) -> list[str]:
    return [p for p in _normalise(text).split("\n\n") if p.strip()]


def _tail_tokens(text: str, n: int) -> str:
    """Last ~n tokens of text, on a word boundary — the overlap carried forward."""
    if n <= 0:
        return ""
    words = text.split()
    out: list[str] = []
    for w in reversed(words):
        out.insert(0, w)
        if count_tokens(" ".join(out)) >= n:
            break
    return " ".join(out)


def chunk_document(doc: Document,
                   max_tokens: int = CHUNK_TOKENS,
                   overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    paragraphs = _paragraphs(doc.text)
    chunks: list[Chunk] = []
    buffer: list[str] = []
    carry = ""

    def flush() -> None:
        nonlocal buffer, carry
        if not buffer:
            return
        body = " ".join(buffer).strip()
        full = f"{carry} {body}".strip() if carry else body
        ordinal = len(chunks)
        c = Chunk(
            chunk_id=f"{doc.doc_id}::{ordinal}",
            doc_id=doc.doc_id,
            title=doc.title,
            kind=doc.kind,
            date=doc.date,
            text=full,
            ordinal=ordinal,
        )
        c.embed_text = f"{c.header()}\n{full}"
        c.meta = {"doc_id": doc.doc_id, "title": doc.title, "kind": doc.kind,
                  "date": doc.date, "ordinal": ordinal}
        chunks.append(c)
        carry = _tail_tokens(body, overlap)
        buffer = []

    for para in paragraphs:
        candidate = " ".join(buffer + [para])
        if buffer and count_tokens(candidate) > max_tokens:
            flush()
        # A single paragraph over budget becomes its own chunk rather than
        # being cut mid-sentence. Slightly oversized beats incoherent.
        buffer.append(para)
        if count_tokens(" ".join(buffer)) >= max_tokens:
            flush()

    flush()
    return chunks


def chunk_corpus(documents: list[Document]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunk_document(doc))
    return chunks


def chunk_stats(chunks: list[Chunk]) -> dict:
    sizes = [count_tokens(c.text) for c in chunks]
    return {
        "chunks": len(chunks),
        "mean_tokens": round(sum(sizes) / len(sizes), 1) if sizes else 0,
        "min_tokens": min(sizes) if sizes else 0,
        "max_tokens": max(sizes) if sizes else 0,
    }
