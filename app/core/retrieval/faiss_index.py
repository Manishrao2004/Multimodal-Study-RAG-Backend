"""Dense retrieval: sentence-embeddings in a FAISS index (matches EduRAG's
FAISS + BGE-small choice, but the embedding model is swappable via config —
any sentence-transformers model id works)."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.models.schemas import Chunk

_model_cache: dict[str, SentenceTransformer] = {}
_model_lock = Lock()


def get_embedding_model(model_name: str) -> SentenceTransformer:
    with _model_lock:
        if model_name not in _model_cache:
            _model_cache[model_name] = SentenceTransformer(model_name)
        return _model_cache[model_name]


class FaissIndex:
    def __init__(self, model_name: str):
        self._model_name = model_name
        self._chunk_ids: list[str] = []
        self._index: faiss.Index | None = None

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            self._chunk_ids = []
            self._index = None
            return
        model = get_embedding_model(self._model_name)
        embeddings = model.encode(
            [c.text for c in chunks], normalize_embeddings=True, show_progress_bar=False
        )
        embeddings = np.asarray(embeddings, dtype="float32")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized inner product
        index.add(embeddings)
        self._index = index
        self._chunk_ids = [c.chunk_id for c in chunks]

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._index is None:
            return []
        model = get_embedding_model(self._model_name)
        query_vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        query_vec = np.asarray(query_vec, dtype="float32")
        scores, indices = self._index.search(query_vec, min(top_k, len(self._chunk_ids)))
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self._chunk_ids[idx], float(score)))
        return results

    def save(self, path: Path) -> None:
        if self._index is not None:
            faiss.write_index(self._index, str(path))

    @property
    def is_built(self) -> bool:
        return self._index is not None
