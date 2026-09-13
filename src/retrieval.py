"""The retrieval pipeline: hybrid search, fusion, reranking, attribution.

    query
      ├─ dense (embeddings)  → top-k by cosine
      └─ sparse (BM25)       → top-k by term weight
                ↓
      reciprocal rank fusion
                ↓
      cross-encoder rerank (top-k only)
                ↓
      similarity floor + source attribution

**Why both retrievers.** They fail differently. Dense search understands that
"my gateway went quiet" is about uplinks, but will happily return a chunk about
the wrong firmware version because version strings barely move an embedding.
BM25 nails "REL-5.0" and "exit code 3" — exact tokens — and is helpless on
paraphrase. Running both and fusing covers each one's blind spot, and the
ablation in evaluate.py measures exactly how much that is worth rather than
asserting it.

**Why RRF rather than weighted score blending.** Cosine similarity and BM25
scores are not on the same scale, and BM25's scale shifts with corpus
statistics. Any fixed weighting is a constant that was tuned once and silently
rots. Reciprocal rank fusion uses only *rank position*, so it needs no
normalisation and no tuning:

    score(d) = Σ over retrievers  1 / (K + rank(d))

**Why rerank last and on few candidates.** The cross-encoder reads query and
passage together, which is why it is accurate and why it costs. Running it over
40 fused candidates is affordable; running it over the corpus is not. Retrieve
broad, rerank narrow.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from .chunking import Chunk, chunk_corpus
from .config import RERANK_K, RETRIEVE_K, RRF_K, SIM_FLOOR, USE_RERANKER
from .corpus import DOCUMENTS
from .embeddings import backend_name, embed, embed_one, rerank_scores, tokenize
from .vectorstore import BaseStore, open_store

KNOWLEDGE_COLLECTION = "knowledge"


@dataclass
class Retrieved:
    """A chunk on its way to the context window, with its provenance intact."""
    chunk_id: str
    doc_id: str
    title: str
    kind: str
    date: str
    text: str
    ordinal: int
    dense_score: float = 0.0
    bm25_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float | None = None
    final_score: float = 0.0
    retrievers: list[str] = field(default_factory=list)

    def citation(self) -> str:
        return f"{self.doc_id}#{self.ordinal}"

    def attribution(self) -> str:
        return f"[{self.citation()}] {self.title} ({self.kind}, {self.date})"


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
class BM25:
    """Okapi BM25. Small enough to keep in memory, explicit enough to reason
    about — and having it in-process means the sparse half of the hybrid has no
    extra service dependency."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5,
                 b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = corpus_tokens
        self.n = len(corpus_tokens)
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = sum(self.doc_len) / self.n if self.n else 0.0
        self.tf: list[dict[str, int]] = []
        df: dict[str, int] = defaultdict(int)
        for toks in corpus_tokens:
            counts: dict[str, int] = defaultdict(int)
            for t in toks:
                counts[t] += 1
            self.tf.append(dict(counts))
            for t in counts:
                df[t] += 1
        self.idf = {
            t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()
        }

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = [0.0] * self.n
        for i in range(self.n):
            dl = self.doc_len[i] or 1
            s = 0.0
            for term in q:
                f = self.tf[i].get(term)
                if not f:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += idf * (f * (self.k1 + 1)) / denom
            out[i] = s
        return out


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------
class KnowledgeIndex:
    def __init__(self, store: BaseStore | None = None):
        self.store = store or open_store(KNOWLEDGE_COLLECTION)
        self._chunks: list[Chunk] = []
        self._bm25: BM25 | None = None
        self._by_id: dict[str, Chunk] = {}

    # -- build ------------------------------------------------------------
    def build(self, documents=None, reset: bool = True) -> dict:
        documents = documents if documents is not None else DOCUMENTS
        chunks = chunk_corpus(documents)
        if reset:
            self.store.reset()
        vectors = embed([c.embed_text for c in chunks])
        self.store.add(
            ids=[c.chunk_id for c in chunks],
            texts=[c.text for c in chunks],
            embeddings=vectors,
            metadatas=[c.meta for c in chunks],
        )
        self._install(chunks)
        return {"documents": len(documents), "chunks": len(chunks),
                "store": self.store.name, "embedder": backend_name()}

    def load(self) -> int:
        """Rehydrate the in-memory BM25 index from whatever is persisted."""
        rows = self.store.get_all()
        chunks = [
            Chunk(chunk_id=h.id, doc_id=h.meta.get("doc_id", ""),
                  title=h.meta.get("title", ""), kind=h.meta.get("kind", ""),
                  date=h.meta.get("date", ""), text=h.text,
                  ordinal=int(h.meta.get("ordinal", 0)), meta=h.meta)
            for h in rows
        ]
        for c in chunks:
            c.embed_text = f"{c.header()}\n{c.text}"
        self._install(chunks)
        return len(chunks)

    def _install(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._by_id = {c.chunk_id: c for c in chunks}
        self._bm25 = BM25([tokenize(c.embed_text) for c in chunks])

    @property
    def size(self) -> int:
        return len(self._chunks)

    # -- search -----------------------------------------------------------
    def search(self, query: str, *, k: int = RERANK_K,
               candidates: int = RETRIEVE_K, use_dense: bool = True,
               use_sparse: bool = True, use_rerank: bool | None = None,
               sim_floor: float = SIM_FLOOR,
               where: dict | None = None) -> list[Retrieved]:
        if not self._chunks:
            self.load()
        if not self._chunks:
            return []

        use_rerank = USE_RERANKER if use_rerank is None else use_rerank
        ranked: dict[str, Retrieved] = {}
        rank_lists: list[list[str]] = []

        # -- dense --------------------------------------------------------
        if use_dense:
            hits = self.store.query(embed_one(query), k=candidates, where=where)
            order = []
            for h in hits:
                r = self._as_retrieved(h.id)
                if r is None:
                    continue
                r.dense_score = h.score
                r.retrievers.append("dense")
                ranked[r.chunk_id] = r
                order.append(r.chunk_id)
            rank_lists.append(order)

        # -- sparse -------------------------------------------------------
        if use_sparse and self._bm25 is not None:
            scores = self._bm25.scores(query)
            idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:candidates]
            order = []
            for i in idx:
                if scores[i] <= 0:
                    continue
                c = self._chunks[i]
                if where and not all(c.meta.get(f) == v for f, v in where.items()):
                    continue
                r = ranked.get(c.chunk_id) or self._as_retrieved(c.chunk_id)
                if r is None:
                    continue
                r.bm25_score = scores[i]
                if "sparse" not in r.retrievers:
                    r.retrievers.append("sparse")
                ranked[r.chunk_id] = r
                order.append(r.chunk_id)
            rank_lists.append(order)

        if not ranked:
            return []

        # -- reciprocal rank fusion ---------------------------------------
        for order in rank_lists:
            for position, cid in enumerate(order):
                ranked[cid].fused_score += 1.0 / (RRF_K + position + 1)

        pool = sorted(ranked.values(), key=lambda r: -r.fused_score)
        pool = pool[: max(k * 4, candidates)]

        # -- rerank -------------------------------------------------------
        if use_rerank and pool:
            scores = rerank_scores(query, [r.text for r in pool])
            for r, s in zip(pool, scores):
                r.rerank_score = s
            pool.sort(key=lambda r: -(r.rerank_score or 0.0))
            for r in pool:
                r.final_score = r.rerank_score or 0.0
        else:
            for r in pool:
                r.final_score = r.fused_score

        # -- similarity floor ---------------------------------------------
        # Applied on the *dense* score, not the rerank score, because the floor
        # asks "is this chunk about the query at all" — a scale-free question
        # the bi-encoder answers consistently. Cross-encoder scores are logits
        # whose range shifts with the model, so a fixed threshold on them would
        # mean something different every time the reranker changed.
        kept = [r for r in pool if (not use_dense) or r.dense_score >= sim_floor
                or "sparse" in r.retrievers]
        return kept[:k]

    def _as_retrieved(self, chunk_id: str) -> Retrieved | None:
        c = self._by_id.get(chunk_id)
        if c is None:
            return None
        return Retrieved(chunk_id=c.chunk_id, doc_id=c.doc_id, title=c.title,
                         kind=c.kind, date=c.date, text=c.text, ordinal=c.ordinal)


_INDEX: KnowledgeIndex | None = None


def get_index() -> KnowledgeIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = KnowledgeIndex()
        _INDEX.load()
    return _INDEX
