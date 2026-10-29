from app.eval.provider_errors import error_metadata


def test_diagnostics_keep_quota_numbers_but_never_provider_identifiers():
    result = error_metadata(RuntimeError("Org org_private tokens per day (TPD): Limit 100, Used 99, Requested 10. key secret_value"))
    assert result["error_reason"] == "daily_token_quota"
    assert result["limit"] == 100
    assert result["requested"] == 10
    assert "org_private" not in str(result)
    assert "secret_value" not in str(result)


def test_unknown_provider_error_does_not_retain_message():
    result = error_metadata(ValueError("sensitive private response"))
    assert result == {"error_type": "ValueError", "error_reason": "judge_or_transport_failure"}
