import pytest

from app.core.generation.llm_client import LLMClient, LLMConfig, LLMError, capture_llm_calls


@pytest.mark.asyncio
async def test_rate_limit_fallback_records_actual_model_without_payload(monkeypatch):
    async def post(self, url, payload, headers):
        if self.config.model == "primary":
            raise LLMError("rate limited", status_code=429)
        return {"choices": [{"message": {"content": "response"}}]}
    monkeypatch.setattr(LLMClient, "_post", post)
    fallback = LLMConfig(provider="openai_compatible", model="secondary", base_url="http://local")
    client = LLMClient(LLMConfig(provider="openai_compatible", model="primary", base_url="http://local", fallback=fallback))
    with capture_llm_calls() as calls:
        result = await client.chat([{"role": "user", "content": "private prompt"}])
    assert result == "response"
    assert [(c["model"], c["status"]) for c in calls] == [
        ("primary", "fallback_used"), ("primary", "rate_limited_fallback"), ("secondary", "ok")]
    assert "private prompt" not in str(calls)


@pytest.mark.asyncio
async def test_unrelated_error_text_cannot_trigger_fallback(monkeypatch):
    async def post(*args, **kwargs):
        raise LLMError("status 500; unrelated text 429", status_code=500)
    monkeypatch.setattr(LLMClient, "_post", post)
    fallback = LLMConfig(provider="openai_compatible", model="secondary", base_url="http://local")
    client = LLMClient(LLMConfig(provider="openai_compatible", model="primary", base_url="http://local", fallback=fallback))
    with capture_llm_calls() as calls, pytest.raises(LLMError):
        await client.chat([{"role": "user", "content": "test"}])
    assert len(calls) == 1 and calls[0]["http_status"] == 500


@pytest.mark.asyncio
async def test_failed_fallback_is_not_reported_as_success(monkeypatch):
    async def post(self, url, payload, headers):
        raise LLMError("unavailable", status_code=429 if self.config.model == "primary" else 404)
    monkeypatch.setattr(LLMClient, "_post", post)
    fallback = LLMConfig(provider="openai_compatible", model="secondary", base_url="http://local")
    client = LLMClient(LLMConfig(provider="openai_compatible", model="primary", base_url="http://local", fallback=fallback))
    with capture_llm_calls() as calls, pytest.raises(LLMError):
        await client.chat([{"role": "user", "content": "test"}])
    assert calls[0]["status"] == "fallback_failed"
    assert calls[-1]["status"] == "error" and calls[-1]["http_status"] == 404
