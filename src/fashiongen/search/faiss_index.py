"""
faiss_index.py
==============
Approximate-nearest-neighbour search over the CLIP style-embedding catalog.

Primary backend: FAISS HNSW (Hierarchical Navigable Small World) - the graph
index that gives sub-millisecond-ish query latency at 100K-1M+ vectors with
high recall. Because embeddings are L2-normalised, inner-product search is
equivalent to cosine similarity, so we use METRIC_INNER_PRODUCT.

Fallback backend: scikit-learn NearestNeighbors (exact / brute or kd/ball
tree). This keeps the whole search stack runnable and testable on machines
without FAISS installed - the public API is identical, so nothing downstream
changes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import faiss
    _HAS_FAISS = True
except Exception:                              # pragma: no cover
    _HAS_FAISS = False


def _l2norm(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x.astype(np.float32))
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


@dataclass
class SearchResult:
    ids: np.ndarray            # (Q, k) catalog ids
    scores: np.ndarray         # (Q, k) cosine similarities
    latency_ms: float          # mean per-query latency


class StyleIndex:
    """HNSW ANN index over garment style embeddings (FAISS or sklearn)."""

    def __init__(self, dim: int, space: str = "cosine",
                 hnsw_M: int = 32, ef_construction: int = 200, ef_search: int = 64):
        self.dim = dim
        self.space = space
        self.hnsw_M = hnsw_M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.backend = "faiss" if _HAS_FAISS else "sklearn"
        self._index = None
        self._ids = None
        self._vecs = None                      # kept for sklearn / debugging

    # ------------------------------------------------------------------ #
    def build(self, embeddings: np.ndarray, ids=None):
        emb = _l2norm(embeddings)
        n = emb.shape[0]
        self._ids = (np.asarray(ids) if ids is not None
                     else np.arange(n).astype(str))
        if self.backend == "faiss":
            index = faiss.IndexHNSWFlat(self.dim, self.hnsw_M,
                                        faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search
            index.add(emb)
            self._index = index
        else:
            from sklearn.neighbors import NearestNeighbors
            # cosine on normalised vectors == inner product; use brute for recall
            nn = NearestNeighbors(metric="cosine", algorithm="brute")
            nn.fit(emb)
            self._index = nn
            self._vecs = emb
        return self

    # ------------------------------------------------------------------ #
    def search(self, queries: np.ndarray, k: int = 10) -> SearchResult:
        q = _l2norm(np.atleast_2d(queries))
        t0 = time.perf_counter()
        if self.backend == "faiss":
            scores, idx = self._index.search(q, k)         # inner product
        else:
            dist, idx = self._index.kneighbors(q, n_neighbors=k)
            scores = 1.0 - dist                            # cosine distance -> sim
        latency_ms = (time.perf_counter() - t0) / len(q) * 1000.0
        return SearchResult(ids=self._ids[idx], scores=scores, latency_ms=latency_ms)

    def set_ef_search(self, ef: int):
        """Tune recall/latency trade-off at query time (FAISS only)."""
        self.ef_search = ef
        if self.backend == "faiss":
            self._index.hnsw.efSearch = ef

    # ------------------------------------------------------------------ #
    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path) + ".ids.npy", self._ids)
        meta = dict(dim=self.dim, backend=self.backend, hnsw_M=self.hnsw_M,
                    ef_construction=self.ef_construction, ef_search=self.ef_search)
        if self.backend == "faiss":
            faiss.write_index(self._index, str(path) + ".faiss")
        else:
            np.save(str(path) + ".vecs.npy", self._vecs)
        import json
        Path(str(path) + ".meta.json").write_text(__import__("json").dumps(meta))

    @classmethod
    def load(cls, path: str) -> "StyleIndex":
        import json
        meta = json.loads(Path(str(path) + ".meta.json").read_text())
        obj = cls(dim=meta["dim"], hnsw_M=meta["hnsw_M"],
                  ef_construction=meta["ef_construction"],
                  ef_search=meta["ef_search"])
        obj._ids = np.load(str(path) + ".ids.npy", allow_pickle=True)
        if meta["backend"] == "faiss" and _HAS_FAISS:
            obj._index = faiss.read_index(str(path) + ".faiss")
            obj.backend = "faiss"
        else:
            from sklearn.neighbors import NearestNeighbors
            vecs = np.load(str(path) + ".vecs.npy")
            nn = NearestNeighbors(metric="cosine", algorithm="brute").fit(vecs)
            obj._index, obj._vecs, obj.backend = nn, vecs, "sklearn"
        return obj
