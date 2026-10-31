from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.core.security import validate_query_text

# Max length is also enforced declaratively via Field(max_length=...) on each
# field below (kept in sync with Settings.security_max_query_length); this
# validator additionally rejects blank text and disallowed control
# characters, which a length bound alone can't catch — see app.core.security
# for why raw control chars in a query matter (log/terminal injection via a
# verbatim-logged query).
_MAX_QUERY_LENGTH = 4000


def _clean_query_text(value: str) -> str:
    return validate_query_text(value, max_length=_MAX_QUERY_LENGTH)


class ChunkType(str, Enum):
    text = "text"
    table = "table"
    visual = "visual"
    audio = "audio"


class Chunk(BaseModel):
    chunk_id: str
    source_file: str
    page_number: int | None = None
    section_path: str | None = None
    text: str
    chunk_length: int
    type: ChunkType
    image_ref: str | None = None
    visual_inspection_status: Literal["completed", "unavailable"] | None = None
    # Audio chunks carry their position in the recording instead of a page.
    timestamp_start: float | None = None
    timestamp_end: float | None = None

    def locator(self) -> str:
        """Human-readable 'where did this come from', used in prompts and in
        the UI's citation line."""
        if self.type == ChunkType.audio and self.timestamp_start is not None:
            from app.core.ingestion.transcription import format_timestamp

            return f"{self.source_file} @ {format_timestamp(self.timestamp_start)}"
        if self.page_number is not None:
            return f"{self.source_file}, p.{self.page_number}"
        return self.source_file


class IngestResponse(BaseModel):
    source_file: str
    chunks_added: int
    text_chunks: int
    table_chunks: int
    visual_chunks: int
    audio_chunks: int = 0
    figures_detected: int = 0
    figures_captioned: int = 0
    warnings: list[str] = Field(default_factory=list)


class DocumentSummary(BaseModel):
    source_file: str
    chunk_count: int
    text_chunks: int
    table_chunks: int
    visual_chunks: int
    audio_chunks: int
    pages: int | None = None


class KnowledgeBaseStats(BaseModel):
    documents: int
    chunks: int
    by_type: dict[str, int]
    index_ready: bool


class RetrievalMode(str, Enum):
    """Ablation settings mirroring EduRAG's Table 3 (BM25-only -> hybrid ->
    full system), exposed at query time so the evaluation harness can run all
    three arms against one live index."""

    bm25_only = "bm25_only"
    dense_only = "dense_only"
    hybrid = "hybrid"  # BM25 + FAISS + RRF, no HyDE and no reranking
    full = "full"  # + HyDE expansion + cross-encoder reranking


class AnswerScope(str, Enum):
    """Student-facing knowledge policy, independent of retrieval ablations."""

    auto = "auto"
    sources_only = "sources_only"
    model_only = "model_only"


class AnswerBasis(str, Enum):
    sources = "sources"
    mixed = "mixed"
    model = "model"


class EvidenceSufficiency(str, Enum):
    strong = "strong"
    partial = "partial"
    insufficient = "insufficient"
    not_checked = "not_checked"


class EvidenceMethod(str, Enum):
    edurag = "edurag"
    adaptive = "adaptive"
    aspect = "aspect"


class EvidenceQuote(BaseModel):
    chunk_id: str = Field(min_length=1, max_length=500)
    quote: str = Field(min_length=1, max_length=1000)


class AspectEvidence(BaseModel):
    aspect_id: str
    aspect: str
    status: Literal["supported", "partial", "missing", "conflicting", "not_checked"]
    supports: list[EvidenceQuote] = Field(default_factory=list)
    conflicts: list[EvidenceQuote] = Field(default_factory=list)
    reason: str = ""


class RetrievalAttempt(BaseModel):
    aspect_ids: list[str] = Field(default_factory=list)
    query: str
    status: Literal["completed", "unavailable"]
    retrieved_ids: list[str] = Field(default_factory=list)
    added_ids: list[str] = Field(default_factory=list)
    elapsed_ms: float = 0


class VerifiedClaim(BaseModel):
    claim_id: str
    text: str
    provenance: Literal["source", "derived", "background"]
    status: Literal["supported", "unsupported", "contradicted", "not_checked"]
    citation_markers: list[int] = Field(default_factory=list)
    supports: list[EvidenceQuote] = Field(default_factory=list)
    reason: str = ""


class ClaimVerification(BaseModel):
    status: Literal["completed", "unavailable", "not_checked"] = "not_checked"
    claims: list[VerifiedClaim] = Field(default_factory=list)
    initial_claims: list[VerifiedClaim] = Field(default_factory=list)
    repair_attempted: bool = False
    withheld: bool = False
    fallback_used: bool = False
    fallback_kind: Literal["supported_claims", "source_excerpts"] | None = None
    released_claim_ids: list[str] = Field(default_factory=list)
    failure_type: str | None = None
    reason: str | None = None
    model_calls: int = 0


class EvidenceTrace(BaseModel):
    initial_chunk_ids: list[str] = Field(default_factory=list)
    final_chunk_ids: list[str] = Field(default_factory=list)
    initial_map: list[AspectEvidence] = Field(default_factory=list)
    retrieval_attempts: list[RetrievalAttempt] = Field(default_factory=list)
    assessment_calls: int = 0
    targeted_retrieval_enabled: bool = False
    claim_verification_enabled: bool = False


class ConversationTurn(BaseModel):
    """Recent client-side conversation context for a follow-up question.

    This is deliberately small and is never treated as evidence: it only helps
    resolve references such as "explain that more simply".
    """

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        return validate_query_text(value, max_length=8000)


class QuestionPlan(BaseModel):
    """Structured, model-produced interpretation of the student's request.

    It carries conversation resolution and response requirements through the
    pipeline without embedding topic-specific rules in generation prompts.
    The plan is organisational metadata, never factual evidence.
    """

    standalone_query: str = Field(min_length=1, max_length=_MAX_QUERY_LENGTH)
    referenced_items: list[str] = Field(default_factory=list, max_length=50)
    requested_aspects: list[str] = Field(
        default_factory=list, max_length=20,
        description="Only aspects explicitly requested by the student. A generic comparison is ['comparison'], without invented comparison criteria.",
    )
    response_format: str = Field(default="concise student-friendly explanation", max_length=500)
    verification_required: bool = False
    verification_reason: str | None = Field(default=None, max_length=500)


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=_MAX_QUERY_LENGTH)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: RetrievalMode = RetrievalMode.full
    detect_contradictions: bool = True
    comprehensive: bool = False
    history: list[ConversationTurn] = Field(default_factory=list, max_length=12)
    answer_scope: AnswerScope = AnswerScope.auto
    evidence_method: EvidenceMethod | None = None

    _validate_query = field_validator("query")(_clean_query_text)


class EvidenceItem(BaseModel):
    chunk: Chunk
    rerank_score: float
    attribution_score: float | None = None
    attribution_percent: float | None = None


class TokenGrounding(BaseModel):
    token: str
    score: float
    match_type: str  # exact | stem | substring | ungrounded


class Citation(BaseModel):
    marker: int
    chunk_id: str
    valid: bool


class Disagreement(BaseModel):
    """A surfaced cross-document conflict (plan Sec. 5.1)."""

    chunk_id_a: str
    chunk_id_b: str
    source_a: str
    source_b: str
    similarity: float
    claim_a: str
    claim_b: str
    verified: bool = False
    explanation: str | None = None


class AskResponse(BaseModel):
    answer: str
    evidence: list[EvidenceItem]
    token_grounding: list[TokenGrounding] | None = None
    citations: list[Citation] | None = None
    grounding_ratio: float | None = None
    citation_validity_rate: float | None = None
    disagreements: list[Disagreement] = Field(default_factory=list)
    mode: RetrievalMode = RetrievalMode.full
    latency_ms: float | None = None
    comprehensive: bool = False
    sources_considered: list[str] = Field(default_factory=list)
    sources_used: list[str] = Field(default_factory=list)
    answer_scope: AnswerScope = AnswerScope.auto
    answer_basis: AnswerBasis = AnswerBasis.sources
    evidence_sufficiency: EvidenceSufficiency = EvidenceSufficiency.not_checked
    provenance_notice: str | None = None
    evidence_check_status: Literal["completed", "unavailable", "not_checked"] = "not_checked"
    evidence_check_reason: str | None = None
    supported_aspects: list[str] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)
    evidence_method: EvidenceMethod = EvidenceMethod.adaptive
    evidence_map: list[AspectEvidence] = Field(default_factory=list)
    evidence_trace: EvidenceTrace = Field(default_factory=EvidenceTrace)
    claim_verification: ClaimVerification = Field(default_factory=ClaimVerification)
    model_calls: list[dict] = Field(default_factory=list)


class ScreenshotAskResponse(BaseModel):
    """Question answered from a pasted/captured image, optionally augmented
    with evidence retrieved from the persistent study-material knowledge base."""

    answer: str
    image_analysis: str
    image_width: int
    image_height: int
    evidence: list[EvidenceItem]
    citations: list[Citation]
    grounding_ratio: float
    citation_validity_rate: float
    used_knowledge_base: bool
    saved: bool = False
    saved_chunk_id: str | None = None


# --- Study-productivity layer (plan Sec. 5.2) -----------------------------


class SummarizeRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=_MAX_QUERY_LENGTH)
    top_k: int = Field(default=8, ge=1, le=20)
    source_file: str | None = None

    _validate_topic = field_validator("topic")(_clean_query_text)


class SummarizeResponse(BaseModel):
    summary: str
    evidence: list[EvidenceItem]
    citations: list[Citation]
    citation_validity_rate: float
    grounding_ratio: float


class CompareRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=_MAX_QUERY_LENGTH)
    source_a: str
    source_b: str
    top_k: int = Field(default=4, ge=1, le=10)

    _validate_topic = field_validator("topic")(_clean_query_text)


class CompareResponse(BaseModel):
    comparison: str
    evidence_a: list[EvidenceItem]
    evidence_b: list[EvidenceItem]
    citations: list[Citation]
    citation_validity_rate: float


class QuizFormat(str, Enum):
    mcq = "mcq"
    short_answer = "short_answer"
    flashcard = "flashcard"


class QuizRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=_MAX_QUERY_LENGTH)
    count: int = Field(default=5, ge=1, le=20)
    format: QuizFormat = QuizFormat.mcq
    top_k: int = Field(default=8, ge=1, le=20)
    source_file: str | None = None

    _validate_topic = field_validator("topic")(_clean_query_text)


class QuizItem(BaseModel):
    question: str
    options: list[str] | None = None  # MCQ only
    answer: str
    explanation: str | None = None
    citation_marker: int | None = None
    chunk_id: str | None = None
    grounded: bool = False


class QuizResponse(BaseModel):
    items: list[QuizItem]
    evidence: list[EvidenceItem]
    grounded_rate: float
    dropped: int = 0
