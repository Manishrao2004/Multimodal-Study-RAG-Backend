"""Central settings. All model choices are swappable via env vars — nothing here
is hardcoded to a specific vendor. Copy .env.example to .env and edit as needed.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Storage -------------------------------------------------------
    data_dir: Path = BACKEND_ROOT / "data"
    upload_dir: Path = BACKEND_ROOT / "data" / "uploads"
    image_dir: Path = BACKEND_ROOT / "data" / "images"
    kb_dir: Path = BACKEND_ROOT / "data" / "kb"
    kb_db_path: Path = BACKEND_ROOT / "data" / "kb" / "knowledge_base.sqlite3"
    faiss_index_path: Path = BACKEND_ROOT / "data" / "kb" / "faiss.index"

    # --- Chunking --------------------------------------------------------
    chunk_max_tokens: int = 512

    # --- Embeddings (dense retrieval) ------------------------------------
    # Any sentence-transformers-compatible model id works here.
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # --- Reranker ----------------------------------------------------------
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- LLM (answer generation + HyDE) -------------------------------------
    # "ollama"          -> talks to a local Ollama server (OLLAMA_BASE_URL)
    # "openai_compatible" -> talks to any OpenAI-compatible chat endpoint
    #                        (OpenAI, Groq, OpenRouter, LM Studio, vLLM, etc.)
    # "none"            -> disabled; useful for retrieval-only testing
    llm_provider: str = "ollama"
    llm_model: str = "gemma2:9b"
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: str = ""

    hyde_provider: str = "ollama"
    hyde_model: str = "deepseek-r1:1.5b"
    hyde_base_url: str = "http://localhost:11434"
    hyde_api_key: str = ""
    hyde_enabled: bool = True

    # --- Vision-language model (diagram captioning) -------------------------
    vlm_provider: str = "none"  # "ollama" | "openai_compatible" | "none"
    vlm_model: str = "llava-phi3"
    vlm_base_url: str = "http://localhost:11434"
    vlm_api_key: str = ""

    # Figure filtering before captioning (EduRAG Sec. III-A: size / aspect-ratio
    # filters drop icons, rules, and logos; perceptual hashing drops repeats of
    # the same figure across pages).
    vlm_min_width_px: int = 120
    vlm_min_height_px: int = 120
    vlm_min_area_px: int = 40_000
    vlm_max_aspect_ratio: float = 6.0
    vlm_phash_distance: int = 4  # Hamming distance under which two figures are "the same"
    vlm_max_images_per_doc: int = 40  # safety cap on captioning cost per upload

    # --- Audio transcription (lecture recordings -> text chunks) -------------
    # "openai_compatible" -> any /audio/transcriptions endpoint (Groq, OpenAI)
    # "faster_whisper"    -> local CPU/GPU transcription, no API calls
    # "none"              -> audio ingestion disabled
    asr_provider: str = "none"
    asr_model: str = "whisper-large-v3-turbo"
    asr_base_url: str = "https://api.groq.com/openai/v1"
    asr_api_key: str = ""
    asr_segment_target_chars: int = 1200  # transcript chunk size before splitting

    # --- Retrieval tuning ----------------------------------------------------
    bm25_top_k: int = 20
    faiss_top_k: int = 20
    rrf_k: int = 60
    rerank_top_k: int = 5

    # --- Attribution ---------------------------------------------------------
    attribution_alpha: float = 0.7  # weight on cross-encoder relevance
    attribution_beta: float = 0.3  # weight on lexical (Jaccard) overlap
    grounding_threshold: float = 0.8

    # --- Contradiction detection (project novel contribution, Sec. 5.1) -------
    # Band bounds calibrated by measuring bge-small-en-v1.5 cosine similarity on
    # constructed pairs:
    #     unrelated topics          0.47 - 0.52
    #     genuine contradictions     0.82 - 0.89
    #     paraphrases / agreement    0.91 - 1.00
    # Contradictions and paraphrases *overlap*, so cosine similarity alone
    # cannot tell "these disagree" from "these agree" — which is exactly why the
    # base paper's bare-threshold consensus check is weak. The band's only job is
    # to discard unrelated pairs and verbatim duplicates cheaply; the LLM
    # verification step below does the actual discrimination.
    consensus_related_min: float = 0.70
    consensus_agreement_max: float = 0.97
    consensus_max_pairs_verified: int = 6  # LLM verification budget per query

    # --- Security ----------------------------------------------------------------
    # Uploaded files and retrieved chunk text are both untrusted input.
    security_max_document_bytes: int = 25_000_000  # 25 MB per PDF/DOCX/PPTX/HTML
    security_max_audio_bytes: int = 100_000_000  # 100 MB per lecture recording
    security_max_screenshot_bytes: int = 10_000_000  # 10 MB per pasted/captured image
    security_max_screenshot_pixels: int = 20_000_000  # decompression-bomb guard
    screenshot_max_dimension_px: int = 4096  # resize very large captures before VLM calls
    security_max_query_length: int = 4000  # applies to /ask query and /study topic

    # --- Server ----------------------------------------------------------------
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.kb_dir.mkdir(parents=True, exist_ok=True)
    settings.image_dir.mkdir(parents=True, exist_ok=True)
    return settings
