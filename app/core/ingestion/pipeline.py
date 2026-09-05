"""Document ingestion: Docling parsing -> page/section-aware chunks.

Mirrors EduRAG's Tier-1 (structural parsing with Docling, chunk_id preserving
page + section metadata, text/table/visual chunk types). Visual captioning
(VLM) is wired in separately via app/core/ingestion/vlm_caption.py — this
module handles text and table content.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker

from app.models.schemas import Chunk, ChunkType

_converter = DocumentConverter()


def _make_chunk_id(source_file: str, index: int) -> str:
    digest = hashlib.sha1(f"{source_file}:{index}".encode()).hexdigest()[:10]
    return f"{Path(source_file).stem}_{index:04d}_{digest}"


def parse_document(file_path: Path) -> list[Chunk]:
    """Parse a single document (PDF/DOCX/PPTX/HTML) into text/table chunks."""
    result = _converter.convert(str(file_path))
    doc = result.document

    chunker = HierarchicalChunker()
    chunks: list[Chunk] = []
    for i, chunk in enumerate(chunker.chunk(doc)):
        text = chunk.text.strip()
        if not text:
            continue

        meta = chunk.meta
        page_number = None
        section_path = None
        is_table = False

        doc_items = getattr(meta, "doc_items", None) or []
        for item in doc_items:
            prov = getattr(item, "prov", None) or []
            if prov:
                page_number = getattr(prov[0], "page_no", None)
            label = str(getattr(item, "label", "")).lower()
            if "table" in label:
                is_table = True

        headings = getattr(meta, "headings", None)
        if headings:
            section_path = " > ".join(headings)

        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(file_path.name, i),
                source_file=file_path.name,
                page_number=page_number,
                section_path=section_path,
                text=text,
                chunk_length=len(text),
                type=ChunkType.table if is_table else ChunkType.text,
                image_ref=None,
            )
        )
    return chunks
