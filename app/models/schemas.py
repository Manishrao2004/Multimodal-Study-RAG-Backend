from enum import Enum

from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    text = "text"
    table = "table"
    visual = "visual"


class Chunk(BaseModel):
    chunk_id: str
    source_file: str
    page_number: int | None = None
    section_path: str | None = None
    text: str
    chunk_length: int
    type: ChunkType
    image_ref: str | None = None


class IngestResponse(BaseModel):
    source_file: str
    chunks_added: int
    text_chunks: int
    table_chunks: int
    visual_chunks: int


class AskRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class EvidenceItem(BaseModel):
    chunk: Chunk
    rerank_score: float
    attribution_score: float | None = None


class TokenGrounding(BaseModel):
    token: str
    score: float
    match_type: str  # exact | stem | substring | ungrounded


class Citation(BaseModel):
    marker: int
    chunk_id: str
    valid: bool


class AskResponse(BaseModel):
    answer: str
    evidence: list[EvidenceItem]
    token_grounding: list[TokenGrounding] | None = None
    citations: list[Citation] | None = None
    grounding_ratio: float | None = None
