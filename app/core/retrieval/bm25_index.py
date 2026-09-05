"""Lexical retrieval: Okapi BM25 over NLTK-tokenized chunk text (matches
EduRAG's lexical index choice)."""

from __future__ import annotations

import re

import nltk
from rank_bm25 import BM25Okapi

from app.models.schemas import Chunk

_STOPWORDS: set[str] | None = None


def _ensure_nltk_data() -> None:
    for pkg, path in [
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("stopwords", "corpora/stopwords"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


def get_stopwords() -> set[str]:
    global _STOPWORDS
    if _STOPWORDS is None:
        _ensure_nltk_data()
        from nltk.corpus import stopwords

        _STOPWORDS = set(stopwords.words("english"))
    return _STOPWORDS


def tokenize(text: str) -> list[str]:
    _ensure_nltk_data()
    from nltk.tokenize import word_tokenize

    tokens = word_tokenize(text.lower())
    return [t for t in tokens if re.search(r"[a-z0-9]", t)]


class BM25Index:
    def __init__(self):
        self._chunk_ids: list[str] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self._chunk_ids = [c.chunk_id for c in chunks]
        tokenized = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self._chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return [(cid, score) for cid, score in ranked[:top_k] if score > 0]

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None
