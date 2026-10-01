"""Cheap, deterministic evidence routing for adaptive study answers.

This is intentionally conservative. It does not claim to prove answerability;
it estimates whether the retrieved set is suitable for a strict source answer,
needs labelled background expansion, or is unrelated to the question.
"""

from __future__ import annotations

import re

from app.models.schemas import Chunk, EvidenceSufficiency, RetrievalMode

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "about", "all", "an", "and", "are", "as", "at", "be", "by",
    "can", "do", "for", "from", "give", "how", "i", "in", "is", "it",
    "me", "my", "of", "on", "or", "please", "tell", "that", "the",
    "this", "to", "what", "when", "where", "which", "who", "why", "with",
}
def _content_tokens(text: str) -> set[str]:
    return {token for token in _TOKEN.findall(text.lower()) if token not in _STOPWORDS}


def assess_evidence_sufficiency(
    query: str,
    ranked: list[tuple[Chunk, float]],
    mode: RetrievalMode,
) -> EvidenceSufficiency:
    if not ranked:
        return EvidenceSufficiency.insufficient

    query_tokens = _content_tokens(query)
    evidence_tokens = _content_tokens(" ".join(chunk.text for chunk, _ in ranked[:5]))
    lexical_coverage = (
        len(query_tokens & evidence_tokens) / len(query_tokens) if query_tokens else 0.0
    )
    top_score = max(float(score) for _, score in ranked)

    # Only full-mode scores are calibrated sigmoid cross-encoder probabilities;
    # BM25, dense and RRF scores use incomparable scales.
    semantic_strong = mode == RetrievalMode.full and top_score >= 0.62
    semantic_related = mode == RetrievalMode.full and top_score >= 0.25

    if lexical_coverage >= 0.50 or (lexical_coverage >= 0.25 and semantic_strong):
        result = EvidenceSufficiency.strong
    elif lexical_coverage >= 0.15 or semantic_related:
        result = EvidenceSufficiency.partial
    else:
        result = EvidenceSufficiency.insufficient

    return result
