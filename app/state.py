"""Process-wide singletons: the knowledge base and its retrieval index.
FastAPI's dependency system pulls these from `app.state` (set in main.py's
lifespan) rather than module-level globals, so tests can substitute their own."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import Settings
from app.core.retrieval.pipeline import RetrievalIndex
from app.db.knowledge_base import KnowledgeBase
from app.db.conversations import ConversationStore


@dataclass
class AppState:
    settings: Settings
    kb: KnowledgeBase
    retrieval: RetrievalIndex
    conversations: ConversationStore = field(init=False)

    def __post_init__(self):
        self.conversations = ConversationStore(self.settings.kb_db_path.parent / "conversations.sqlite3")


def build_app_state() -> AppState:
    from app.config import get_settings

    settings = get_settings()
    kb = KnowledgeBase(settings.kb_db_path)
    retrieval = RetrievalIndex(kb, settings)
    if kb.count() > 0:
        retrieval.rebuild()
    return AppState(settings=settings, kb=kb, retrieval=retrieval)
