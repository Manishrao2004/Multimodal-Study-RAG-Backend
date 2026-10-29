"""Minimal provider availability check without printing credentials or error bodies."""
import asyncio
import json

import httpx
from app.config import get_settings


async def main():
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(settings.llm_base_url.rstrip("/") + "/chat/completions",
                                         headers={"Authorization": "Bearer " + settings.llm_api_key},
                                         json={"model": settings.llm_model, "messages": [{"role": "user", "content": "Reply OK."}],
                                               "max_tokens": 64, "temperature": 0})
        error = response.json().get("error", {}) if response.headers.get("content-type", "").startswith("application/json") else {}
        message = error.get("message", "").lower()
        print(json.dumps({"status": response.status_code, "error_code": error.get("code"),
                          "daily_quota": "tokens per day" in message, "minute_quota": "tokens per minute" in message,
                          "retry_after": response.headers.get("retry-after")}))
    except Exception as exc:
        print(json.dumps({"error_type": type(exc).__name__}))


if __name__ == "__main__":
    asyncio.run(main())
