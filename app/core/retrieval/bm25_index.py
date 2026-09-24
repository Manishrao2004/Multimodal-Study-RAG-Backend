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
    """Lowercased word tokens, punctuation dropped. Stopwords are kept — the
    grounding and ROUGE-L/token-F1 metrics need the full token stream. Use
    `tokenize_for_retrieval` for the BM25 path instead."""
    _ensure_nltk_data()
    from nltk.tokenize import word_tokenize

    tokens = word_tokenize(text.lower())
    return [t for t in tokens if re.search(r"[a-z0-9]", t)]


def tokenize_for_retrieval(text: str) -> list[str]:
    """BM25 tokens with stopwords removed.

    Left in, stopwords make BM25 score any chunk sharing only a word like "the"
    against the query. Those junk candidates carry a real RRF rank and can push
    genuine matches out of the fused candidate pool, so they are dropped before
    indexing and before scoring.
    """
    stopwords = get_stopwords()
    return [t for t in tokenize(text) if t not in stopwords]


class BM25Index:
    def __init__(self):
        self._chunk_ids: list[str] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self._chunk_ids = [c.chunk_id for c in chunks]
        tokenized = [tokenize_for_retrieval(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        query_tokens = tokenize_for_retrieval(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(zip(self._chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return [(cid, score) for cid, score in ranked[:top_k] if score > 0]

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None
