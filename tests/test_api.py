"""API-level tests. The LLM is stubbed, so these exercise real retrieval,
attribution and grounding without any network access."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_app_state
from app.api.routes_ingest import safe_filename
from app.main import app
from app.models.schemas import ChunkType
from tests.conftest import make_chunk

STUB_ANSWER = (
    "Reciprocal Rank Fusion merges ranked lists by summing reciprocal ranks [1]. "
    "BM25 matches exact keywords only [2]."
)


@pytest.fixture
def client(populated_state, monkeypatch):
    async def fake_generate_answer(query, chunks, settings, disagreements=None):
        return STUB_ANSWER

    monkeypatch.setattr("app.api.routes_ask.generate_answer", fake_generate_answer)
    app.dependency_overrides[get_app_state] = lambda: populated_state
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestMetaEndpoints:
    def test_health(self, client):
        assert client.get("/health").json() == {"status": "ok"}

    def test_config_reports_models_without_leaking_the_key(self, client):
        payload = client.get("/config").json()
        assert "api_key_set" in payload["llm"]
        assert "api_key" not in payload["llm"]
        assert isinstance(payload["llm"]["api_key_set"], bool)


class TestAsk:
    def test_answer_carries_evidence_and_grounding(self, client):
        response = client.post("/ask", json={"query": "what is reciprocal rank fusion"})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == STUB_ANSWER
        assert body["evidence"]
        assert 0.0 <= body["grounding_ratio"] <= 1.0
        assert body["latency_ms"] > 0

    def test_citations_are_validated_against_retrieved_chunks(self, client):
        body = client.post("/ask", json={"query": "reciprocal rank fusion"}).json()
        markers = {c["marker"] for c in body["citations"]}
        assert markers == {1, 2}
        assert all(c["valid"] for c in body["citations"])
        assert body["citation_validity_rate"] == 1.0

    def test_evidence_attribution_percentages_sum_to_one(self, client):
        body = client.post("/ask", json={"query": "reranking", "top_k": 3}).json()
        total = sum(item["attribution_percent"] for item in body["evidence"])
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_top_k_is_honoured(self, client):
        body = client.post("/ask", json={"query": "retrieval", "top_k": 2}).json()
        assert len(body["evidence"]) <= 2

    def test_ablation_mode_is_echoed_back(self, client):
        body = client.post(
            "/ask", json={"query": "bm25 keywords", "mode": "bm25_only"}
        ).json()
        assert body["mode"] == "bm25_only"

    def test_unknown_mode_is_rejected(self, client):
        response = client.post("/ask", json={"query": "x", "mode": "telepathy"})
        assert response.status_code == 422

    def test_empty_knowledge_base_returns_409(self, settings, kb, monkeypatch):
        from app.core.retrieval.pipeline import RetrievalIndex
        from app.state import AppState

        empty = AppState(settings=settings, kb=kb, retrieval=RetrievalIndex(kb, settings))
        app.dependency_overrides[get_app_state] = lambda: empty
        with TestClient(app) as test_client:
            response = test_client.post("/ask", json={"query": "anything"})
        app.dependency_overrides.clear()
        assert response.status_code == 409


class TestDocuments:
    def test_listing_groups_chunks_by_source(self, client):
        docs = {d["source_file"]: d for d in client.get("/documents").json()}
        assert docs["notes.pdf"]["chunk_count"] == 2
        assert docs["slides.pptx"]["chunk_count"] == 1

    def test_stats_reports_totals_and_readiness(self, client):
        stats = client.get("/documents/stats").json()
        assert stats["documents"] == 2
        assert stats["chunks"] == 3
        assert stats["index_ready"] is True
        assert stats["by_type"]["text"] == 3

    def test_deleting_a_document_removes_it_from_the_index(self, client):
        assert client.delete("/documents/notes.pdf").json()["chunks_removed"] == 2
        assert client.get("/documents/stats").json()["chunks"] == 1

    def test_deleting_an_unknown_document_is_404(self, client):
        assert client.delete("/documents/nope.pdf").status_code == 404


class TestEvidence:
    def test_chunk_can_be_fetched_by_id(self, client):
        body = client.get("/evidence/c1").json()
        assert body["chunk_id"] == "c1"

    def test_unknown_chunk_is_404(self, client):
        assert client.get("/evidence/missing").status_code == 404

    def test_text_chunk_has_no_image(self, client):
        assert client.get("/evidence/c1/image").status_code == 404


class TestUploadFilenameSafety:
    def test_plain_filename_passes_through(self):
        assert safe_filename("lecture.pdf") == "lecture.pdf"

    @pytest.mark.parametrize(
        "hostile",
        ["../../etc/passwd", "..\\..\\windows\\system32\\cfg.pdf", "/abs/path/x.pdf"],
    )
    def test_directory_components_are_stripped(self, hostile):
        cleaned = safe_filename(hostile)
        assert "/" not in cleaned and "\\" not in cleaned
        assert not cleaned.startswith("..")

    def test_missing_filename_is_rejected(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            safe_filename("")


class TestIngestValidation:
    def test_unsupported_extension_is_rejected(self, client):
        response = client.post(
            "/ingest", files={"file": ("virus.exe", b"MZ", "application/octet-stream")}
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]


@pytest.mark.asyncio
class TestConsensus:
    """Plan Sec. 5.1 — the band prefilter must not fire on unrelated passages."""

    async def test_unrelated_passages_are_not_flagged_as_conflicting(self, settings):
        from app.core.attribution.consensus import find_disagreements

        chunks = [
            make_chunk("a", "Photosynthesis converts light energy into glucose in plants.", "book.pdf"),
            make_chunk("b", "The French Revolution began in 1789 with the storming of the Bastille.", "notes.pdf"),
        ]
        # llm_provider is 'none' in the fixture, so this returns the unverified
        # band candidates — an unrelated pair must produce none at all.
        assert await find_disagreements(chunks, settings) == []

    async def test_near_identical_passages_are_not_flagged(self, settings):
        from app.core.attribution.consensus import find_disagreements

        text = "BM25 ranks documents using term frequency and inverse document frequency."
        chunks = [
            make_chunk("a", text, "book.pdf"),
            make_chunk("b", text, "notes.pdf"),
        ]
        assert await find_disagreements(chunks, settings) == []

    async def test_same_document_pairs_are_never_compared(self, settings):
        from app.core.attribution.consensus import detect_disagreements

        chunks = [
            make_chunk("a", "The tree has 8 levels in this implementation.", "book.pdf"),
            make_chunk("b", "The tree has 12 levels in this implementation.", "book.pdf"),
        ]
        flagged = detect_disagreements(chunks, settings.embedding_model, 0.0, 1.0)
        assert flagged == []

    async def test_conflicting_cross_document_claims_enter_the_band(self, settings):
        from app.core.attribution.consensus import detect_disagreements

        chunks = [
            make_chunk("a", "A B-tree node in this course holds a maximum of 8 keys.", "book.pdf"),
            make_chunk("b", "In our syllabus a B-tree node holds at most 12 keys per node.", "notes.pdf"),
        ]
        flagged = detect_disagreements(
            chunks,
            settings.embedding_model,
            related_min=settings.consensus_related_min,
            agreement_max=settings.consensus_agreement_max,
        )
        assert len(flagged) == 1
        assert {flagged[0].source_a, flagged[0].source_b} == {"book.pdf", "notes.pdf"}


class TestChunkLocator:
    def test_page_locator_for_documents(self):
        assert make_chunk("a", "x", "book.pdf", page=42).locator() == "book.pdf, p.42"

    def test_audio_locator_uses_a_timestamp(self):
        chunk = make_chunk("a", "x", "lecture.mp3", page=None, chunk_type=ChunkType.audio)
        chunk.timestamp_start = 125.0
        assert chunk.locator() == "lecture.mp3 @ 2:05"

    def test_locator_falls_back_to_the_filename(self):
        assert make_chunk("a", "x", "notes.html", page=None).locator() == "notes.html"
