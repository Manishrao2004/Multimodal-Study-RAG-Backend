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


def _outline_only(text: str) -> bool:
    """Recognize clear outlines without guessing whether normal prose is sufficient."""
    lines = [re.sub(r"^\s*(?:[-*#]+|\d+[.)])\s*", "", line).strip()
             for line in text.splitlines() if line.strip()]
    if not lines:
        return True
    for line in lines:
        list_intro = re.search(
            r"\b(types|kinds|categories|forms|stages|steps)\b.{0,60}\b(are|include)\b",
            line, re.IGNORECASE,
        )
        if list_intro:
            continue
        # An inline explanation or a sentence is substantive, even when short.
        parts = re.split(r"\s*[:\u2013\u2014]\s*|\s+-\s+", line, maxsplit=1)
        if len(parts) == 2 and len(_TOKEN.findall(parts[1])) >= 4:
            return False
        if len(_TOKEN.findall(line)) > 12 or re.search(r"[.!?]$", line):
            return False
    return True


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

    if result == EvidenceSufficiency.strong and re.search(
        r"\b(explain|describe|definition|meaning|elaborate|details?|understand)\b|\bwhat is\b",
        query, re.IGNORECASE,
    ):
        related = [chunk for chunk, _ in ranked[:5]
                   if query_tokens & _content_tokens(chunk.text + " " + (chunk.section_path or ""))]
        if related and all(_outline_only(chunk.text) for chunk in related):
            return EvidenceSufficiency.partial
    return result
