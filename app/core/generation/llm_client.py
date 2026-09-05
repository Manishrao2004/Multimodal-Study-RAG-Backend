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

import asyncio
import re
from dataclasses import dataclass

import httpx

_THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think_tags(text: str) -> str:
    """Removes <think>...</think> reasoning blocks that reasoning models (e.g.
    DeepSeek-R1, Qwen3, gpt-oss) emit before their final answer. Applied to
    every generation path — an unstripped block otherwise leaks into answers,
    poisons token-grounding ratios, and confuses citation extraction."""
    cleaned = _THINK_TAG_PATTERN.sub("", text)
    # A truncated/unclosed reasoning block: keep only what follows the tag.
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>")[0]
    return cleaned.strip()


class LLMError(RuntimeError):
    """Raised when a provider call fails, with the provider's own message kept
    intact — a 404 'model not found' from the gateway is the single most common
    setup mistake and needs to survive up to the API layer."""


@dataclass
class LLMConfig:
    provider: str
    model: str
    base_url: str
    api_key: str = ""
    timeout: float = 120.0
    max_retries: int = 4
    backoff_base: float = 2.0


def parse_retry_after(response: httpx.Response) -> float | None:
    """How long a rate-limited provider wants us to wait.

    Prefers the standard Retry-After header; falls back to the wait Groq states
    in its error message ("Please try again in 6.32s"), which it sends without
    always setting the header.
    """
    header = response.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    match = re.search(r"try again in ([\d.]+)\s*s", response.text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


def build_vision_message(prompt: str, image_b64: str, provider: str, mime: str = "image/png") -> dict:
    """A single user message carrying one image, in whichever shape the provider
    expects. Ollama takes a parallel `images` list of bare base64; the OpenAI
    wire format takes a content array with a data: URL."""
    if provider == "ollama":
        return {"role": "user", "content": prompt, "images": [image_b64]}
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
        ],
    }


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    async def chat(self, messages: list[dict], temperature: float = 0.1) -> str:
        provider = self.config.provider
        if provider == "ollama":
            raw = await self._chat_ollama(messages, temperature)
        elif provider == "openai_compatible":
            raw = await self._chat_openai_compatible(messages, temperature)
        elif provider == "none":
            raise LLMError("LLM provider is set to 'none'; generation is disabled.")
        else:
            raise ValueError(f"Unknown LLM provider: {provider!r}")
        return strip_think_tags(raw)

    async def _chat_ollama(self, messages: list[dict], temperature: float) -> str:
        url = f"{self.config.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        data = await self._post(url, payload, headers={})
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
        data = await self._post(url, payload, headers)
        return data["choices"][0]["message"]["content"]

    async def _post(self, url: str, payload: dict, headers: dict) -> dict:
        """POST with retry on rate limits and transient server errors.

        Free API tiers enforce tight per-minute token budgets — Groq's is 8k TPM
        — which the evaluation harness blows through immediately when it runs a
        benchmark. Without this, a run silently loses a third of its queries and
        reports metrics over whatever survived.
        """
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                    resp = await client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    return resp.json()
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status = exc.response.status_code
                retryable = status == 429 or 500 <= status < 600
                if not retryable or attempt == self.config.max_retries:
                    raise LLMError(
                        f"{self.config.provider} call failed ({status}) for model "
                        f"'{self.config.model}': {exc.response.text[:400]}"
                    ) from exc
                delay = parse_retry_after(exc.response) or self.config.backoff_base**attempt
                # Cap so one absurd Retry-After can't stall a whole benchmark run.
                await asyncio.sleep(min(delay + 0.5, 30.0))
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt == self.config.max_retries:
                    raise LLMError(
                        f"Could not reach {self.config.provider} at "
                        f"{self.config.base_url}: {exc}"
                    ) from exc
                await asyncio.sleep(self.config.backoff_base**attempt)

        raise LLMError(f"{self.config.provider} call failed: {last_error}")
