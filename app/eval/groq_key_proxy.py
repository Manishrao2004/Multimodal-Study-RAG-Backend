"""Local-only Groq proxy for a user's keys; tries another key on HTTP 429.

Start with uvicorn on 127.0.0.1. Reads LLM_API_KEY assignments (including
commented ones) from .env without logging or persisting credentials. Never
exposes credentials in responses. Intended only for local evaluation.
"""
from pathlib import Path
import re

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response

app = FastAPI()
keys = []
for line in (Path(__file__).resolve().parents[2] / ".env").read_text(encoding="utf-8").splitlines():
    match = re.match(r"^\s*#?\s*LLM_API_KEY\s*=\s*(.*?)\s*$", line)
    if match:
        value = match.group(1).strip().strip('"').strip("'")
        if value and value not in keys:
            keys.append(value)

next_key = 0


@app.post("/v1/chat/completions")
async def chat(request: Request):
    global next_key
    if not keys:
        return Response(status_code=503, content="No Groq keys configured")
    body = await request.body()
    start = next_key % len(keys)
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        for n in range(len(keys)):
            index = (start + n) % len(keys)
            try:
                result = await client.post("https://api.groq.com/openai/v1/chat/completions", content=body,
                    headers={"Authorization": "Bearer " + keys[index], "Content-Type": "application/json"})
            except httpx.HTTPError:
                return Response(status_code=503, content="Groq connection failed")
            if result.status_code != 429:
                next_key = index
                # Do not forward provider error bodies, which can contain account data.
                if result.status_code >= 400:
                    return Response(status_code=result.status_code, content="Groq request failed")
                return Response(content=result.content, status_code=result.status_code, media_type="application/json")
        next_key = (start + 1) % len(keys)
        return Response(status_code=429, content="All configured Groq keys rate-limited", headers={"Retry-After": "10"})
