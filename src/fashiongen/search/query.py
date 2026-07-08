"""
query.py
========
User-facing search service. Wraps a StyleIndex plus (optionally) a ClipEncoder
so callers can search the garment catalog by free text or by an example image.

    svc = SearchService.load("artifacts/index/style", encoder=ClipEncoder())
    svc.search_text("floral midi dress with puff sleeves", k=12)
    svc.search_image("query.jpg", k=12)

If no encoder is attached (e.g. embeddings were precomputed elsewhere), the
raw-vector search path is still available.
"""
from __future__ import annotations

import numpy as np

from .faiss_index import StyleIndex, SearchResult


class SearchService:
    def __init__(self, index: StyleIndex, encoder=None):
        self.index = index
        self.encoder = encoder

    @classmethod
    def load(cls, index_path: str, encoder=None):
        return cls(StyleIndex.load(index_path), encoder=encoder)

    def search_text(self, query: str, k: int = 12) -> SearchResult:
        if self.encoder is None:
            raise RuntimeError("attach a ClipEncoder to search by text")
        q = self.encoder.encode_texts([query])
        return self.index.search(q, k=k)

    def search_image(self, image, k: int = 12) -> SearchResult:
        if self.encoder is None:
            raise RuntimeError("attach a ClipEncoder to search by image")
        q = self.encoder.encode_images([image])
        return self.index.search(q, k=k)

    def search_vector(self, vec: np.ndarray, k: int = 12) -> SearchResult:
        return self.index.search(np.atleast_2d(vec), k=k)
