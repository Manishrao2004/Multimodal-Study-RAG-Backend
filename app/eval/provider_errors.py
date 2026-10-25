"""Redacted diagnostics for evaluator failures; never retain provider bodies."""
import re


def error_metadata(exc: Exception) -> dict:
    message = str(exc).lower()
    reason = "judge_or_transport_failure"
    if "tokens per day" in message or "tpd" in message:
        reason = "daily_token_quota"
    elif "requests per day" in message or "rpd" in message:
        reason = "daily_request_quota"
    elif "tokens per minute" in message or "tpm" in message:
        reason = "minute_token_quota"
    elif "rate limit" in message or "rate_limit" in message:
        reason = "rate_limit"
    result = {"error_type": type(exc).__name__, "error_reason": reason}
    for field in ("limit", "used", "requested"):
        match = re.search(r"\b" + field + r"\s*:?\s*([0-9]+(?:\.[0-9]+)?)", message)
        if match:
            result[field] = float(match.group(1))
    return result
