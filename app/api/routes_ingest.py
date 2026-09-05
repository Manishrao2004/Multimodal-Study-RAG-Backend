from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.api.deps import get_app_state
from app.core.ingestion.pipeline import parse_document
from app.models.schemas import ChunkType, IngestResponse
from app.state import AppState

router = APIRouter(prefix="/ingest", tags=["ingest"])

_SUPPORTED_SUFFIXES = {".pdf", ".docx", ".pptx", ".html", ".htm"}


@router.post("", response_model=IngestResponse)
async def ingest_document(file: UploadFile, state: AppState = Depends(get_app_state)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(_SUPPORTED_SUFFIXES)}",
        )

    dest_path = state.settings.upload_dir / file.filename
    contents = await file.read()
    dest_path.write_bytes(contents)

    chunks = parse_document(dest_path)
    if not chunks:
        raise HTTPException(status_code=422, detail="No extractable content found in document.")

    state.kb.add_chunks(chunks)
    state.retrieval.rebuild()

    return IngestResponse(
        source_file=file.filename,
        chunks_added=len(chunks),
        text_chunks=sum(1 for c in chunks if c.type == ChunkType.text),
        table_chunks=sum(1 for c in chunks if c.type == ChunkType.table),
        visual_chunks=sum(1 for c in chunks if c.type == ChunkType.visual),
    )
