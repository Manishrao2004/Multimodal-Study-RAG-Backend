"""Deterministic token grounding — Eq. (8) + Algorithm 1 Stage 4 in the EduRAG
paper. Hierarchical lexical matching cascade, no model calls:

  exact match in context      -> 1.0
  stem match in context       -> 0.9
  substring match in context  -> 0.8
  otherwise (ungrounded)      -> 0.2

grounding_ratio is computed over content tokens only (stopwords excluded).
"""

from __future__ import annotations

from app.core.retrieval.bm25_index import get_stopwords, tokenize
from app.models.schemas import TokenGrounding

EXACT_SCORE = 1.0
STEM_SCORE = 0.9
SUBSTRING_SCORE = 0.8
UNGROUNDED_SCORE = 0.2


def _stemmer():
    from nltk.stem import PorterStemmer

    return PorterStemmer()


def compute_token_grounding(answer_text: str, context_text: str) -> list[TokenGrounding]:
    stopwords = get_stopwords()
    stemmer = _stemmer()

    answer_tokens = tokenize(answer_text)
    context_tokens = tokenize(context_text)
    context_token_set = set(context_tokens)
    context_stem_set = {stemmer.stem(t) for t in context_tokens}
    context_joined = " ".join(context_tokens)

    results: list[TokenGrounding] = []
    for token in answer_tokens:
        if token in stopwords:
            continue  # excluded from grounding_ratio, but we still skip scoring stopwords here
        if token in context_token_set:
            results.append(TokenGrounding(token=token, score=EXACT_SCORE, match_type="exact"))
        elif stemmer.stem(token) in context_stem_set:
            results.append(TokenGrounding(token=token, score=STEM_SCORE, match_type="stem"))
        elif token in context_joined:
            results.append(TokenGrounding(token=token, score=SUBSTRING_SCORE, match_type="substring"))
        else:
            results.append(TokenGrounding(token=token, score=UNGROUNDED_SCORE, match_type="ungrounded"))
    return results


def grounding_ratio(token_scores: list[TokenGrounding], threshold: float) -> float:
    if not token_scores:
        return 0.0
    grounded = sum(1 for t in token_scores if t.score >= threshold)
    return grounded / len(token_scores)
