"""Long-term storage: a persistent vector index.

Chroma is the primary backend, as the brief suggests. Behind it sits a
numpy-backed store with the same interface.

The fallback is not hedging for its own sake. A vector database is the one
dependency in this system that can fail for environmental reasons — a native
wheel that will not build, a sqlite version mismatch, an API rename between
minor versions — and none of those have anything to do with whether the
retrieval design is correct. Putting both behind one interface means the
pipeline, the evaluation harness and the tests are all portable, and swapping
to Pinecone or Weaviate later is one class rather than a refactor.

Exact search over a few hundred chunks is instant; the numpy store would only
become the wrong choice somewhere north of a hundred thousand vectors, where
approximate nearest neighbour starts to matter.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Hit:
    id: str
    text: str
    meta: dict
    score: float          # cosine similarity, higher is better


class BaseStore:
    name = "base"

    def add(self, ids, texts, embeddings, metadatas) -> None: ...
    def query(self, embedding, k: int, where: dict | None = None) -> list[Hit]: ...
    def count(self) -> int: ...
    def reset(self) -> None: ...
    def get_all(self) -> list[Hit]: ...


# ---------------------------------------------------------------------------
# numpy backend
# ---------------------------------------------------------------------------
class NumpyStore(BaseStore):
    name = "numpy"

    def __init__(self, path: Path, collection: str):
        self.dir = Path(path) / collection
        self.dir.mkdir(parents=True, exist_ok=True)
        self.vec_path = self.dir / "vectors.npy"
        self.meta_path = self.dir / "records.json"
        self._vectors: np.ndarray | None = None
        self._records: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self.vec_path.exists() and self.meta_path.exists():
            self._vectors = np.load(self.vec_path)
            self._records = json.loads(self.meta_path.read_text())
        else:
            self._vectors, self._records = None, []

    def _save(self) -> None:
        if self._vectors is not None:
            np.save(self.vec_path, self._vectors)
        self.meta_path.write_text(json.dumps(self._records, indent=1))

    def add(self, ids, texts, embeddings, metadatas) -> None:
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        existing = {r["id"]: i for i, r in enumerate(self._records)}
        new_vecs, new_recs = [], []
        for i, _id in enumerate(ids):
            rec = {"id": _id, "text": texts[i], "meta": metadatas[i]}
            if _id in existing:                       # upsert
                self._records[existing[_id]] = rec
                if self._vectors is not None:
                    self._vectors[existing[_id]] = arr[i]
            else:
                new_recs.append(rec)
                new_vecs.append(arr[i])
        if new_vecs:
            block = np.vstack(new_vecs)
            self._vectors = block if self._vectors is None else \
                np.vstack([self._vectors, block])
            self._records.extend(new_recs)
        self._save()

    def query(self, embedding, k: int, where: dict | None = None) -> list[Hit]:
        if self._vectors is None or not self._records:
            return []
        q = np.asarray(embedding, dtype=np.float32).reshape(-1)
        qn = np.linalg.norm(q) or 1.0
        mat = self._vectors
        norms = np.linalg.norm(mat, axis=1)
        norms[norms == 0] = 1.0
        sims = (mat @ q) / (norms * qn)

        idx = np.argsort(-sims)
        hits: list[Hit] = []
        for i in idx:
            rec = self._records[int(i)]
            if where and not all(rec["meta"].get(f) == v for f, v in where.items()):
                continue
            hits.append(Hit(rec["id"], rec["text"], rec["meta"], float(sims[int(i)])))
            if len(hits) >= k:
                break
        return hits

    def count(self) -> int:
        return len(self._records)

    def reset(self) -> None:
        self._vectors, self._records = None, []
        for p in (self.vec_path, self.meta_path):
            if p.exists():
                p.unlink()

    def get_all(self) -> list[Hit]:
        return [Hit(r["id"], r["text"], r["meta"], 1.0) for r in self._records]


# ---------------------------------------------------------------------------
# Chroma backend
# ---------------------------------------------------------------------------
class ChromaStore(BaseStore):
    name = "chroma"

    def __init__(self, path: Path, collection: str):
        import chromadb

        self.client = chromadb.PersistentClient(path=str(path))
        self.collection_name = collection
        self.collection = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids, texts, embeddings, metadatas) -> None:
        self.collection.upsert(
            ids=list(ids),
            documents=list(texts),
            embeddings=[list(map(float, e)) for e in embeddings],
            metadatas=list(metadatas),
        )

    def query(self, embedding, k: int, where: dict | None = None) -> list[Hit]:
        if self.count() == 0:
            return []
        res = self.collection.query(
            query_embeddings=[list(map(float, np.asarray(embedding).reshape(-1)))],
            n_results=min(k, self.count()),
            where=where or None,
        )
        hits: list[Hit] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for i, _id in enumerate(ids):
            # Chroma returns cosine DISTANCE; similarity is 1 - distance.
            hits.append(Hit(_id, docs[i], metas[i] or {}, 1.0 - float(dists[i])))
        return hits

    def count(self) -> int:
        return int(self.collection.count())

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"},
        )

    def get_all(self) -> list[Hit]:
        if self.count() == 0:
            return []
        res = self.collection.get()
        return [
            Hit(_id, res["documents"][i], res["metadatas"][i] or {}, 1.0)
            for i, _id in enumerate(res["ids"])
        ]


# ---------------------------------------------------------------------------
def open_store(collection: str, prefer: str = "auto") -> BaseStore:
    """Open a collection, preferring Chroma and degrading gracefully."""
    from .config import CHROMA_DIR, NUMPY_STORE_DIR, ensure_dirs

    ensure_dirs()
    if prefer in ("auto", "chroma"):
        try:
            return ChromaStore(CHROMA_DIR, collection)
        except Exception:
            if prefer == "chroma":
                raise
    return NumpyStore(NUMPY_STORE_DIR, collection)
