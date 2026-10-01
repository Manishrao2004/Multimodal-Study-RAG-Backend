from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.deps import get_app_state
from app.core.security import SecurityValidationError, validate_upload
from app.core.vision.screenshot import SCREENSHOT_EXTENSIONS, normalize_screenshot
from app.main import app


def make_image_bytes(
    size: tuple[int, int] = (320, 180), image_format: str = "PNG"
) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(35, 90, 140)).save(buffer, format=image_format)
    return buffer.getvalue()


class TestScreenshotValidation:
    def test_png_upload_is_accepted(self):
        data = make_image_bytes()
        result = validate_upload(
            "capture.png",
            data,
            supported_extensions=SCREENSHOT_EXTENSIONS,
            max_size_bytes=1_000_000,
        )
        assert result.extension == "png"

    def test_jpeg_upload_is_accepted(self):
        data = make_image_bytes(image_format="JPEG")
        result = validate_upload(
            "capture.jpeg",
            data,
            supported_extensions=SCREENSHOT_EXTENSIONS,
            max_size_bytes=1_000_000,
        )
        assert result.extension == "jpeg"

    def test_mismatched_image_extension_is_rejected(self):
        with pytest.raises(SecurityValidationError):
            validate_upload(
                "capture.png",
                make_image_bytes(image_format="JPEG"),
                supported_extensions=SCREENSHOT_EXTENSIONS,
                max_size_bytes=1_000_000,
            )

    def test_corrupt_image_is_rejected_during_decode(self):
        with pytest.raises(SecurityValidationError):
            normalize_screenshot(
                b"\x89PNG\r\n\x1a\nnot-a-real-image",
                max_pixels=1_000_000,
                max_dimension=2048,
            )

    def test_excessive_pixel_count_is_rejected(self):
        with pytest.raises(SecurityValidationError, match="pixel limit"):
            normalize_screenshot(
                make_image_bytes(size=(100, 100)),
                max_pixels=9_999,
                max_dimension=2048,
            )

    def test_large_dimension_is_resized_before_model_call(self):
        normalized = normalize_screenshot(
            make_image_bytes(size=(800, 200)),
            max_pixels=1_000_000,
            max_dimension=400,
        )
        assert normalized.width == 400
        assert normalized.height == 100
        assert normalized.png_bytes.startswith(b"\x89PNG")


@pytest.fixture
def vision_client(populated_state, monkeypatch):
    populated_state.settings.vlm_provider = "openai_compatible"
    populated_state.settings.llm_provider = "openai_compatible"

    async def fake_analyze_screenshot(screenshot, question, settings):
        return (
            "Visible architecture diagram: BM25 and FAISS feed Reciprocal Rank "
            "Fusion, followed by a cross-encoder reranker."
        )

    async def fake_generate_answer(query, chunks, settings, disagreements=None):
        answer = "The screenshot shows BM25 and FAISS feeding Reciprocal Rank Fusion [1]."
        if len(chunks) > 1:
            answer += " The uploaded notes also describe reciprocal-rank fusion [2]."
        return answer

    def fake_attribution(query, answer, chunks, settings):
        return {chunk.chunk_id: 1.0 for chunk in chunks}

    monkeypatch.setattr("app.api.routes_vision.analyze_screenshot", fake_analyze_screenshot)
    monkeypatch.setattr("app.api.routes_vision.generate_answer", fake_generate_answer)
    monkeypatch.setattr("app.api.routes_vision.compute_attribution", fake_attribution)
    monkeypatch.setattr("app.main.build_app_state", lambda: populated_state)
    app.dependency_overrides[get_app_state] = lambda: populated_state
    with TestClient(app) as test_client:
        yield test_client, populated_state
    app.dependency_overrides.clear()


class TestScreenshotAskAPI:
    def test_disabled_vision_provider_returns_503(self, populated_state, monkeypatch):
        monkeypatch.setattr("app.main.build_app_state", lambda: populated_state)
        app.dependency_overrides[get_app_state] = lambda: populated_state
        with TestClient(app) as client:
            response = client.post(
                "/vision/ask",
                data={"question": "Explain this"},
                files={"image": ("capture.png", make_image_bytes(), "image/png")},
            )
        app.dependency_overrides.clear()
        assert response.status_code == 503

    def test_screenshot_only_answer_is_ephemeral(self, vision_client):
        client, state = vision_client
        original_count = state.kb.count()
        response = client.post(
            "/vision/ask",
            data={"question": "Explain this diagram", "use_knowledge_base": "false"},
            files={"image": ("capture.png", make_image_bytes(), "image/png")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["image_width"] == 320
        assert body["image_height"] == 180
        assert body["evidence"][0]["chunk"]["type"] == "visual"
        assert body["evidence"][0]["chunk"]["image_ref"] is None
        assert body["citations"][0]["valid"] is True
        assert body["used_knowledge_base"] is False
        assert body["saved"] is False
        assert state.kb.count() == original_count

    def test_screenshot_can_be_augmented_with_kb_evidence(self, vision_client, monkeypatch):
        client, state = vision_client

        async def fake_retrieve(query, top_k, mode):
            return [(state.kb.get_chunk("c1"), 0.9)]

        monkeypatch.setattr(state.retrieval, "retrieve", fake_retrieve)
        response = client.post(
            "/vision/ask",
            data={"question": "Connect this to my notes", "use_knowledge_base": "true"},
            files={"image": ("capture.png", make_image_bytes(), "image/png")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["used_knowledge_base"] is True
        assert len(body["evidence"]) == 2
        assert {citation["marker"] for citation in body["citations"]} == {1, 2}

    def test_saved_screenshot_becomes_fetchable_visual_evidence(self, vision_client, monkeypatch):
        client, state = vision_client

        async def fake_rebuild():
            return None

        monkeypatch.setattr(state.retrieval, "rebuild_async", fake_rebuild)
        response = client.post(
            "/vision/ask",
            data={
                "question": "Save and explain this",
                "use_knowledge_base": "false",
                "save_to_library": "true",
            },
            files={"image": ("capture.png", make_image_bytes(), "image/png")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["saved"] is True
        chunk_id = body["saved_chunk_id"]
        stored = state.kb.get_chunk(chunk_id)
        assert stored is not None
        assert stored.image_ref
        assert (state.settings.image_dir / stored.image_ref).is_file()
        assert client.get(f"/evidence/{chunk_id}").status_code == 200
        image_response = client.get(f"/evidence/{chunk_id}/image")
        assert image_response.status_code == 200
        assert image_response.headers["content-type"] == "image/png"

    def test_invalid_image_content_is_rejected(self, vision_client):
        client, _state = vision_client
        response = client.post(
            "/vision/ask",
            data={"question": "Explain this"},
            files={"image": ("capture.png", b"not an image", "image/png")},
        )
        assert response.status_code == 400
