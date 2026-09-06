"""Deterministic near-duplicate detection for generated quiz/flashcard items.

An LLM asked for N items about one topic frequently rephrases the same
question twice (same fact, different wording of the prompt). This is
normalized-exact-text matching only — lowercase, strip punctuation, collapse
whitespace — not a semantic/embedding model; it catches copies and trivial
rewording but will miss genuine paraphrases. That is an intentional scope
limit: a heavyweight dedup model is not worth adding for a handful of quiz
items per request.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TypeVar

_WHITESPACE_OR_PUNCT = re.compile(r"[^a-z0-9]+")

T = TypeVar("T")


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — for exact
    near-duplicate comparison only, not a similarity score."""
    return _WHITESPACE_OR_PUNCT.sub(" ", text.lower()).strip()


def dedupe(items: list[T], key: Callable[[T], str]) -> tuple[list[T], int]:
    """Drop items whose normalized `key(item)` was already seen, preserving
    first-seen order. Returns (kept_items, duplicate_count)."""
    seen: set[str] = set()
    kept: list[T] = []
    duplicate_count = 0
    for item in items:
        normalized = normalize_text(key(item))
        if normalized in seen:
            duplicate_count += 1
            continue
        seen.add(normalized)
        kept.append(item)
    return kept, duplicate_count
