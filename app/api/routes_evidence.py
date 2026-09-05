from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_app_state
from app.models.schemas import Chunk
from app.state import AppState

router = APIRouter(prefix="/evidence", tags=["evidence"])


@router.get("/{chunk_id}", response_model=Chunk)
async def get_evidence_chunk(chunk_id: str, state: AppState = Depends(get_app_state)):
    chunk = state.kb.get_chunk(chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail=f"Chunk '{chunk_id}' not found.")
    return chunk
