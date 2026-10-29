"""Reuse judgments only for identical completed answers and immutable test cases."""
import hashlib
import json


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def case_manifest(records: list[dict]) -> dict[str, str]:
    output_fields = {"response", "status", "elapsed_ms", "trace", "checks", "error_type",
                     "error_reason", "previous_attempts"}
    result = {r["id"]: digest({k: v for k, v in r.items() if k not in output_fields}) for r in records}
    if len(result) != len(records):
        raise ValueError("Duplicate case IDs")
    return result


def row_metadata(record: dict) -> dict:
    return {"record_sha256": digest(record), "input_status": record.get("status")}


def resume_rows(stored: dict, records: list[dict], config: dict, config_keys: tuple[str, ...]) -> tuple[dict, list[str]]:
    for key in config_keys:
        if stored.get(key) != config.get(key):
            raise ValueError(f"Cannot resume a different judge configuration: {key}")
    previous = {r["id"]: r for r in stored["rows"]}
    if len(previous) != len(stored["rows"]):
        raise ValueError("Duplicate cached case IDs")
    if stored.get("records_sha256") == digest(records):
        return previous, []
    if stored.get("case_manifest") != case_manifest(records):
        raise ValueError("Cannot resume changed test cases or reference labels")
    current = {r["id"]: r for r in records}
    recovered = []
    for case_id, old in list(previous.items()):
        if case_id not in current:
            raise ValueError("Cannot remove evaluated cases")
        if old.get("record_sha256") == digest(current[case_id]):
            continue
        # Recovering a failed handler produces a new answer, never a replacement
        # for a previously measured successful answer. Its old NA row has no scores.
        if old.get("input_status") == "error" and current[case_id].get("status") in {"error", "ok"} and not old.get("scores"):
            if current[case_id].get("status") == "ok":
                recovered.append(case_id)
            previous.pop(case_id)
        else:
            raise ValueError(f"Completed answer changed for {case_id}; use a fresh output")
    return previous, recovered
