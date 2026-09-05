from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

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


@router.get("/{chunk_id}/image")
async def get_evidence_image(chunk_id: str, state: AppState = Depends(get_app_state)):
    """Serves the source figure behind a visual chunk, so the UI can show the
    diagram next to the caption that was actually retrieved."""
    chunk = state.kb.get_chunk(chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail=f"Chunk '{chunk_id}' not found.")
    if not chunk.image_ref:
        raise HTTPException(status_code=404, detail="This chunk has no associated image.")

    # image_ref comes from our own ingestion, but resolve-and-verify anyway so a
    # tampered database row can't read files outside the image directory.
    image_dir = state.settings.image_dir.resolve()
    image_path = (image_dir / chunk.image_ref).resolve()
    if not image_path.is_file() or image_dir not in image_path.parents:
        raise HTTPException(status_code=404, detail="Image file is no longer available.")
    return FileResponse(image_path, media_type="image/png")
