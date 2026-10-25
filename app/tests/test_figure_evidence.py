import json

import pytest
from PIL import Image

from app.core.vision import evidence
from app.models.schemas import ChunkType
from app.tests.conftest import chunk


def prepare(settings, monkeypatch, response=None):
    settings.image_dir.mkdir()
    Image.new("RGB", (140, 140), "white").save(settings.image_dir / "figure.png")
    settings.vlm_provider = "openai_compatible"
    evidence._CACHE.clear()
    calls = []
    class Client:
        def __init__(self, config):
            assert config.max_retries == 0
            assert config.max_output_tokens == 1536
        async def chat(self, messages, **kwargs):
            calls.append(messages)
            return response or json.dumps(dict(observations=["Input A points to output B."],
                                               unreadable_details=[], sufficient_for_question=True))
    monkeypatch.setattr(evidence, "LLMClient", Client)
    return chunk("Generic caption", kind=ChunkType.visual, image_ref="figure.png"), calls


@pytest.mark.asyncio
async def test_actual_image_and_question_are_sent_and_cached_without_mutating_caption(settings, monkeypatch):
    c, calls = prepare(settings, monkeypatch)
    first = await evidence.inspect_figure(c, "Explain the arrow", settings)
    second = await evidence.inspect_figure(c, "Explain the arrow", settings)
    assert len(calls) == 1
    assert first.visual_inspection_status == second.visual_inspection_status == "completed"
    assert "Input A points to output B." in first.text
    assert c.text == "Generic caption" and c.visual_inspection_status is None
    assert first.image_ref == "figure.png" and first.chunk_id == c.chunk_id
    message = calls[0][0]["content"]
    assert "Explain the arrow" in message[0]["text"]
    assert message[1]["image_url"]["url"].startswith("data:image/png;base64,")
    await evidence.inspect_figure(c, "Read the output label", settings)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_changed_image_invalidates_the_cached_analysis(settings, monkeypatch):
    c, calls = prepare(settings, monkeypatch)
    await evidence.inspect_figure(c, "Explain the arrow", settings)
    Image.new("RGB", (140, 140), "black").save(settings.image_dir / "figure.png")
    await evidence.inspect_figure(c, "Explain the arrow", settings)
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("response", ["not JSON", "{}", '{"observations":[],"sufficient_for_question":true}',
    '{"observations":["A label is unreadable"],"unreadable_details":["output"],"sufficient_for_question":false}'])
async def test_bad_or_unreadable_analysis_cannot_establish_visual_support(settings, monkeypatch, response):
    c, calls = prepare(settings, monkeypatch, response)
    result = await evidence.inspect_figure(c, "Explain the diagram", settings)
    assert result.visual_inspection_status == "unavailable"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_one_figure_budget_and_disabled_mode(settings, monkeypatch):
    c, calls = prepare(settings, monkeypatch)
    other = c.model_copy(update={"chunk_id": "c2"})
    results = await evidence.inspect_retrieved_figures("Compare these diagrams", [(c, .9), (other, .8)], settings)
    assert len(calls) == 1
    assert results[0][0].visual_inspection_status == "completed"
    assert results[1][0].visual_inspection_status == "unavailable"
    settings.visual_query_enabled = False
    result = await evidence.inspect_figure(c, "New question", settings)
    assert result.visual_inspection_status == "unavailable" and len(calls) == 1


@pytest.mark.asyncio
async def test_path_escape_never_reads_an_external_image(settings, monkeypatch):
    c, calls = prepare(settings, monkeypatch)
    c.image_ref = "../outside.png"
    result = await evidence.inspect_figure(c, "Read the diagram", settings)
    assert result.visual_inspection_status == "unavailable" and not calls


@pytest.mark.asyncio
async def test_provider_error_is_not_retained_as_evidence(settings, monkeypatch):
    c, _ = prepare(settings, monkeypatch)
    class Client:
        def __init__(self, config):
            pass
        async def chat(self, *args, **kwargs):
            raise RuntimeError("secret provider response")
    monkeypatch.setattr(evidence, "LLMClient", Client)
    result = await evidence.inspect_figure(c, "Read the diagram", settings)
    assert result.visual_inspection_status == "unavailable"
    assert "secret" not in result.model_dump_json()


def test_graph_algorithm_question_is_not_a_request_to_read_a_specific_figure():
    assert not evidence.requires_visual_evidence("Explain graph algorithms")
    assert evidence.requires_visual_evidence("Explain the graph")


@pytest.mark.asyncio
async def test_text_question_does_not_inspect_an_incidental_lower_ranked_figure(settings, monkeypatch):
    c, calls = prepare(settings, monkeypatch)
    ranked = [(chunk(), .95), (c, .4)]
    result = await evidence.inspect_retrieved_figures("Define a transformer", ranked, settings)
    assert result is ranked and not calls
