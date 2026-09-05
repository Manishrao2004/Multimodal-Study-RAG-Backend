from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_ask import router as ask_router
from app.api.routes_evidence import router as evidence_router
from app.api.routes_ingest import router as ingest_router
from app.config import get_settings
from app.state import build_app_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.app_state = build_app_state()
    yield


app = FastAPI(
    title="AI Study Assistant — Backend",
    description="Multimodal, explainable RAG backend (EduRAG-derived pipeline).",
    version="0.1.0",
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


@app.get("/health")
async def health():
    return {"status": "ok"}
