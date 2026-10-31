from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.api import routes_study as study
from app.api.deps import get_app_state
from app.models.schemas import Chunk, QuizItem


@pytest.fixture
def client(monkeypatch):
    chunks = {name: Chunk(chunk_id=name, source_file=f"{name}.pdf", text=f"Evidence for {name}", chunk_length=14, type="text") for name in ("a", "b")}
    class Retrieval:
        is_ready = True
        async def retrieve(self, topic, top_k, source_file=None):
            return [(chunks[source_file[0]] if source_file else chunks["a"], 0.9)]
    state = SimpleNamespace(retrieval=Retrieval(), settings=SimpleNamespace(grounding_threshold=.8),
                            kb=SimpleNamespace(list_documents=lambda: [SimpleNamespace(source_file=f"{name}.pdf") for name in chunks]))
    async def summary(*args): return "Summary [1]"
    async def comparison(*args): return "Comparison [1] [2]"
    async def quiz(*args): return [QuizItem(question="Question?", answer="Answer", citation_marker=1, chunk_id="a", grounded=True)], 1
    monkeypatch.setattr(study, "generate_summary", summary)
    monkeypatch.setattr(study, "generate_comparison", comparison)
    monkeypatch.setattr(study, "generate_quiz", quiz)
    monkeypatch.setattr(study, "compute_attribution", lambda *args: {"a": 1.0, "b": 1.0})
    monkeypatch.setattr(study, "compute_token_grounding", lambda *args: [])
    monkeypatch.setattr(study, "grounding_ratio", lambda *args: 0.0)
    app = FastAPI()
    app.include_router(study.router)
    app.dependency_overrides[get_app_state] = lambda: state
    return TestClient(app)


def test_summary_returns_retrieved_source_and_citation(client):
    response = client.post("/study/summarize", json={"topic": "test", "source_file": "a.pdf"})
    assert response.status_code == 200
    assert response.json()["evidence"][0]["chunk"]["chunk_id"] == "a"
    assert response.json()["citations"][0]["valid"] is True


def test_comparison_uses_both_sources_and_rejects_same_document(client):
    response = client.post("/study/compare", json={"topic": "test", "source_a": "a.pdf", "source_b": "b.pdf"})
    assert response.status_code == 200
    assert [c["chunk_id"] for c in response.json()["citations"]] == ["a", "b"]
    assert client.post("/study/compare", json={"topic": "test", "source_a": "a.pdf", "source_b": "a.pdf"}).status_code == 400


@pytest.mark.parametrize("format", ["mcq", "short_answer", "flashcard"])
def test_practice_reports_returned_and_omitted_items(client, format):
    response = client.post("/study/quiz", json={"topic": "test", "format": format})
    assert response.status_code == 200
    assert response.json()["grounded_rate"] == .5
    assert response.json()["dropped"] == 1
