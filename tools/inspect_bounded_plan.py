"""Diagnose a structured planner call with a bounded output budget."""
import asyncio
import json
import time

import httpx
from app.config import get_settings
from app.core.generation import orchestration
from app.eval.validate import CASES, load_cases
from app.models.schemas import ConversationTurn, QuestionPlan


async def main():
    captured = []
    class Capture:
        async def chat(self, messages, **kwargs):
            captured.extend(messages)
            raise RuntimeError("Prompt captured")
    original = orchestration._client
    orchestration._client = lambda settings: Capture()
    case = next(row for row in load_cases(CASES)["cases"] if row["id"] == "followup_comparison")
    settings = get_settings()
    try:
        await orchestration.plan_question(case["query"], [ConversationTurn(**turn) for turn in case["history"]], settings)
    except RuntimeError:
        pass
    finally:
        orchestration._client = original
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            result = await client.post(settings.llm_base_url.rstrip("/") + "/chat/completions",
                                       headers={"Authorization": "Bearer " + settings.llm_api_key},
                                       json={"model": settings.llm_model, "messages": captured,
                                             "max_tokens": 2048, "temperature": 0})
        output = {"status": result.status_code, "elapsed_ms": round((time.perf_counter() - started) * 1000)}
        if not result.is_success:
            error = result.json().get("error", {})
            message = error.get("message", "").lower()
            output.update(error_code=error.get("code"), error_type=error.get("type"),
                          daily_quota="tokens per day" in message or "tpd" in message,
                          minute_quota="tokens per minute" in message or "tpm" in message,
                          retry_after=result.headers.get("retry-after"))
        if result.is_success:
            choice = result.json()["choices"][0]
            output["finish_reason"] = choice.get("finish_reason")
            try:
                output["plan"] = QuestionPlan.model_validate_json(choice["message"]["content"]).model_dump()
            except Exception as exc:
                output["validation_error_type"] = type(exc).__name__
        print(json.dumps(output, indent=2))
    except Exception as exc:
        print(json.dumps({"error_type": type(exc).__name__}))


if __name__ == "__main__":
    asyncio.run(main())
