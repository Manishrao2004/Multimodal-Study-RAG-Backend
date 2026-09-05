from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_ask import router as ask_router
from app.api.routes_documents import router as documents_router
from app.api.routes_evidence import router as evidence_router
from app.api.routes_ingest import router as ingest_router
from app.api.routes_study import router as study_router
from app.config import get_settings
from app.state import build_app_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.app_state = build_app_state()
    yield


app = FastAPI(
    title="AI Study Assistant — Backend",
    description=(
        "Multimodal, explainable RAG backend (EduRAG-derived pipeline).\n\n"
        "Ingest study material (PDF/DOCX/PPTX/HTML, plus diagrams and lecture "
        "audio), ask citation-grounded questions with token-level grounding, "
        "and generate summaries, comparisons and quizzes from the same evidence."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(ask_router)
app.include_router(evidence_router)
app.include_router(study_router)
app.include_router(documents_router)


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok"}


@app.get("/config", tags=["meta"])
async def config():
    """Which models are actually wired up. Misconfiguration (a provider left on
    'none', a model the API key can't reach) is the most common failure in this
    stack, so it is worth being able to see the active setup without reading
    the .env file. Secrets are reported only as present/absent."""
    s = get_settings()
    return {
        "llm": {
            "provider": s.llm_provider,
            "model": s.llm_model,
            "base_url": s.llm_base_url,
            "api_key_set": bool(s.llm_api_key),
        },
        "hyde": {
            "enabled": s.hyde_enabled,
            "provider": s.hyde_provider,
            "model": s.hyde_model,
            "api_key_set": bool(s.hyde_api_key),
        },
        "vlm": {
            "provider": s.vlm_provider,
            "model": s.vlm_model,
            "api_key_set": bool(s.vlm_api_key),
        },
        "asr": {
            "provider": s.asr_provider,
            "model": s.asr_model,
            "api_key_set": bool(s.asr_api_key),
        },
        "embedding_model": s.embedding_model,
        "reranker_model": s.reranker_model,
        "retrieval": {
            "bm25_top_k": s.bm25_top_k,
            "faiss_top_k": s.faiss_top_k,
            "rerank_top_k": s.rerank_top_k,
        },
    }
