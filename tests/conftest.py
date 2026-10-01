from __future__ import annotations

import pytest

from app.config import Settings
from app.core.retrieval.pipeline import RetrievalIndex
from app.db.knowledge_base import KnowledgeBase
from app.models.schemas import Chunk, ChunkType
from app.state import AppState


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Isolated storage per test, and every network-backed provider disabled so
    the suite never depends on an API key or a running Ollama."""
    return Settings(
        data_dir=tmp_path,
        upload_dir=tmp_path / "uploads",
        image_dir=tmp_path / "images",
        kb_dir=tmp_path / "kb",
        kb_db_path=tmp_path / "kb" / "kb.sqlite3",
        llm_provider="none",
        hyde_provider="none",
        hyde_enabled=False,
        vlm_provider="none",
        asr_provider="none",
    )


def make_chunk(
    chunk_id: str,
    text: str,
    source_file: str = "notes.pdf",
    page: int | None = 1,
    chunk_type: ChunkType = ChunkType.text,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source_file=source_file,
        page_number=page,
        section_path=None,
        text=text,
        chunk_length=len(text),
        type=chunk_type,
    )


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        make_chunk(
            "c1",
            "Reciprocal Rank Fusion combines several ranked lists by summing the "
            "reciprocal of each document's rank, which avoids having to normalise "
            "scores across retrievers that use different scales.",
        ),
        make_chunk(
            "c2",
            "BM25 is a sparse lexical ranking function based on term frequency and "
            "inverse document frequency. It matches exact keywords and cannot "
            "recognise synonyms.",
        ),
        make_chunk(
            "c3",
            "A cross-encoder reranker feeds the query and document jointly through a "
            "transformer, letting every query token attend to every document token, "
            "which is far more precise than comparing separate embeddings.",
            source_file="slides.pptx",
            page=4,
        ),
    ]


@pytest.fixture
def kb(settings) -> KnowledgeBase:
    return KnowledgeBase(settings.kb_db_path)


@pytest.fixture
def populated_state(settings, kb, sample_chunks) -> AppState:
    kb.add_chunks(sample_chunks)
    index = RetrievalIndex(kb, settings)
    index.rebuild()
    return AppState(settings=settings, kb=kb, retrieval=index)
