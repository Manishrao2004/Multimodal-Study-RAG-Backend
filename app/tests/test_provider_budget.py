import pytest

from app.core.generation.llm_client import LLMClient, LLMConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("provider,key", [("openai_compatible", "max_tokens"), ("ollama", "num_predict")])
async def test_explicit_token_budget_reaches_each_supported_provider(monkeypatch, provider, key):
    sent = []
    async def post(self, url, payload, headers):
        sent.append(payload)
        return {"message": {"content": "ok"}, "choices": [{"message": {"content": "ok"}}]}
    monkeypatch.setattr(LLMClient, "_post", post)
    client = LLMClient(LLMConfig(provider=provider, model="local-test", base_url="http://unused",
                                 max_retries=0, max_output_tokens=2048))
    assert await client.chat([{"role": "user", "content": "Test"}]) == "ok"
    payload = sent[0]["options"] if provider == "ollama" else sent[0]
    assert payload[key] == 2048
