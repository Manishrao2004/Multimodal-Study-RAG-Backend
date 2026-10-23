"""Caption failures must preserve successful figures and explain partial uploads."""
import asyncio

import pytest

from app.config import Settings
from app.core.ingestion import vlm_caption
from app.core.ingestion.figures import ExtractedFigure


def figure(index):
    return ExtractedFigure(index, 1, str(index).encode(), 400, 400, index)


def test_partial_failure_preserves_captions_and_redacts_provider_error(monkeypatch):
    async def caption(data, settings):
        if data == b"1":
            raise RuntimeError("secret-provider-response")
        return None if data == b"2" else "Readable diagram."

    monkeypatch.setattr(vlm_caption, "caption_image", caption)
    warnings = []
    result = asyncio.run(vlm_caption.caption_figures(
        [figure(i) for i in range(3)],
        Settings(vlm_provider="openai_compatible"), warnings=warnings,
    ))
    assert result == {0: "Readable diagram."}
    assert len(warnings) == 2
    assert "could not be captioned" in warnings[0]
    assert "skipped" in warnings[1]
    assert "secret-provider-response" not in " ".join(warnings)


def test_caption_concurrency_is_bounded(monkeypatch):
    active = peak = 0

    async def caption(data, settings):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return "Diagram."

    monkeypatch.setattr(vlm_caption, "caption_image", caption)
    result = asyncio.run(vlm_caption.caption_figures(
        [figure(i) for i in range(6)],
        Settings(vlm_provider="openai_compatible"), concurrency=2,
    ))
    assert len(result) == 6
    assert peak == 2


def test_invalid_concurrency_fails_instead_of_hanging():
    with pytest.raises(ValueError, match="at least 1"):
        asyncio.run(vlm_caption.caption_figures(
            [figure(0)], Settings(vlm_provider="openai_compatible"), concurrency=0,
        ))


def test_disabled_provider_does_not_call_model(monkeypatch):
    async def unexpected(*args):
        raise AssertionError("Disabled provider must not be invoked")

    monkeypatch.setattr(vlm_caption, "caption_image", unexpected)
    assert asyncio.run(vlm_caption.caption_figures(
        [figure(0)], Settings(vlm_provider="none"),
    )) == {}
