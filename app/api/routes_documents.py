"""Knowledge-base inspection and management.

The demo needs to show what has been ingested and let a document be removed
without wiping the whole database — deleting a source drops its chunks and
rebuilds the indexes, since BM25 and FAISS are both built from the KB in one
pass and hold no per-document deletion path of their own.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_app_state
from app.models.schemas import DocumentSummary, KnowledgeBaseStats
from app.state import AppState

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=list[DocumentSummary])
async def list_documents(state: AppState = Depends(get_app_state)):
    return state.kb.list_documents()


@router.get("/stats", response_model=KnowledgeBaseStats)
async def stats(state: AppState = Depends(get_app_state)):
    return KnowledgeBaseStats(
        documents=len(state.kb.list_documents()),
        chunks=state.kb.count(),
        by_type=state.kb.count_by_type(),
        index_ready=state.retrieval.is_ready,
    )


@router.delete("/{source_file}")
async def delete_document(source_file: str, state: AppState = Depends(get_app_state)):
    removed = state.kb.delete_document(source_file)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"No document named '{source_file}'.")
    await state.retrieval.rebuild_async()
    return {"source_file": source_file, "chunks_removed": removed}
