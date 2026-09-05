"""Provider-agnostic chat-completion client.

Supports:
  - "ollama"           local Ollama server (http://localhost:11434/api/chat)
  - "openai_compatible" any OpenAI-compatible /chat/completions endpoint
                        (OpenAI, Groq, OpenRouter, LM Studio, vLLM, etc.)
  - "none"             disabled, raises if invoked

Nothing in the retrieval/attribution pipeline talks to a vendor SDK directly —
everything routes through `LLMClient.chat`, so switching providers is a config
change, not a code change.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class LLMConfig:
    provider: str
    model: str
    base_url: str
    api_key: str = ""
    timeout: float = 120.0


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    async def chat(self, messages: list[dict], temperature: float = 0.1) -> str:
        provider = self.config.provider
        if provider == "ollama":
            return await self._chat_ollama(messages, temperature)
        if provider == "openai_compatible":
            return await self._chat_openai_compatible(messages, temperature)
        if provider == "none":
            raise RuntimeError("LLM provider is set to 'none'; generation is disabled.")
        raise ValueError(f"Unknown LLM provider: {provider!r}")

    async def _chat_ollama(self, messages: list[dict], temperature: float) -> str:
        url = f"{self.config.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data["message"]["content"]

    async def _chat_openai_compatible(self, messages: list[dict], temperature: float) -> str:
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]
