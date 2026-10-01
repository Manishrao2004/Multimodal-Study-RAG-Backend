"""Lexical retrieval: deterministic Okapi BM25 tokenization.

Tokenization and stopword loading deliberately use package-bundled resources;
the API must never need to download language data during application startup.
"""

from __future__ import annotations

from difflib import get_close_matches
import re

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from app.models.schemas import Chunk

_STOPWORDS = frozenset(ENGLISH_STOP_WORDS)


def get_stopwords() -> frozenset[str]:
    return _STOPWORDS


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens, punctuation dropped. Stopwords are kept — the
    grounding and ROUGE-L/token-F1 metrics need the full token stream. Use
    `tokenize_for_retrieval` for the BM25 path instead."""
    return re.findall(r"(?u)\b\w+\b", text.lower())


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
        self._vocabulary: set[str] = set()

    def build(self, chunks: list[Chunk]) -> None:
        self._chunk_ids = [c.chunk_id for c in chunks]
        tokenized = [tokenize_for_retrieval(c.text) for c in chunks]
        self._vocabulary = {token for chunk_tokens in tokenized for token in chunk_tokens}
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def _expand_typos(self, query_tokens: list[str]) -> list[str]:
        """Append conservative corpus-vocabulary corrections for misspellings.

        The original query terms are retained. A correction is added only for a
        reasonably long unknown token with one close indexed match, making this
        a corpus-driven recall fallback rather than a domain-specific synonym
        list.
        """
        expanded = list(query_tokens)
        for token in query_tokens:
            if len(token) < 5 or token in self._vocabulary:
                continue
            match = get_close_matches(token, self._vocabulary, n=1, cutoff=0.84)
            if match:
                expanded.append(match[0])
        return expanded

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        query_tokens = self._expand_typos(tokenize_for_retrieval(query))
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(zip(self._chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return [(cid, score) for cid, score in ranked[:top_k] if score > 0]

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None
