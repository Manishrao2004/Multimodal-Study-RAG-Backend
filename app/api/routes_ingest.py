from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.api.deps import get_app_state
from app.core.generation.llm_client import LLMError
from app.core.ingestion.pipeline import SUPPORTED_SUFFIXES, ingest_file
from app.models.schemas import ChunkType, IngestResponse
from app.state import AppState

router = APIRouter(prefix="/ingest", tags=["ingest"])


def safe_filename(raw: str | None) -> str:
    """Uploaded filenames are attacker-controlled and are used to build a path
    under data/uploads — strip any directory component so '../../x' can't
    escape the upload directory."""
    name = Path(raw or "").name
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Missing or invalid filename.")
    return name


@router.post("", response_model=IngestResponse)
async def ingest_document(file: UploadFile, state: AppState = Depends(get_app_state)):
    filename = safe_filename(file.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_SUFFIXES)}",
        )

    dest_path = state.settings.upload_dir / filename
    dest_path.write_bytes(await file.read())

    try:
        result = await ingest_file(dest_path, state.settings)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse document: {exc}") from exc

    if not result.chunks:
        raise HTTPException(status_code=422, detail="No extractable content found in document.")

    state.kb.add_chunks(result.chunks)
    await state.retrieval.rebuild_async()

    chunks = result.chunks
    return IngestResponse(
        source_file=filename,
        chunks_added=len(chunks),
        text_chunks=sum(1 for c in chunks if c.type == ChunkType.text),
        table_chunks=sum(1 for c in chunks if c.type == ChunkType.table),
        visual_chunks=sum(1 for c in chunks if c.type == ChunkType.visual),
        audio_chunks=sum(1 for c in chunks if c.type == ChunkType.audio),
        figures_detected=result.figures_detected,
        figures_captioned=result.figures_captioned,
        warnings=result.warnings,
    )
