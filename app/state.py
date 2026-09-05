"""Process-wide singletons: the knowledge base and its retrieval index.
FastAPI's dependency system pulls these from `app.state` (set in main.py's
lifespan) rather than module-level globals, so tests can substitute their own."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.core.retrieval.pipeline import RetrievalIndex
from app.db.knowledge_base import KnowledgeBase


@dataclass
class AppState:
    settings: Settings
    kb: KnowledgeBase
    retrieval: RetrievalIndex


def build_app_state() -> AppState:
    from app.config import get_settings

    settings = get_settings()
    kb = KnowledgeBase(settings.kb_db_path)
    retrieval = RetrievalIndex(kb, settings)
    if kb.count() > 0:
        retrieval.rebuild()
    return AppState(settings=settings, kb=kb, retrieval=retrieval)
