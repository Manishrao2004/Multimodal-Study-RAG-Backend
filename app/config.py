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

    # --- Retrieval tuning ----------------------------------------------------
    bm25_top_k: int = 20
    faiss_top_k: int = 20
    rrf_k: int = 60
    rerank_top_k: int = 5

    # --- Attribution ---------------------------------------------------------
    attribution_alpha: float = 0.7  # weight on cross-encoder relevance
    attribution_beta: float = 0.3  # weight on lexical (Jaccard) overlap
    grounding_threshold: float = 0.8

    # --- Server ----------------------------------------------------------------
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.kb_dir.mkdir(parents=True, exist_ok=True)
    return settings
