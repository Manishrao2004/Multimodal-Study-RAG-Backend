import pytest

from app.eval.asqa_live_http import re_is_loopback, validate_server


def test_runner_accepts_only_local_http_endpoints():
    assert re_is_loopback("http://127.0.0.1:8000")
    assert not re_is_loopback("http://localhost.example.com")
    assert not re_is_loopback("http://user@localhost")


def test_runner_rejects_stale_pipeline_or_index():
    config = {"loaded_pipeline_sha256": "code", "retrieval": {"indexed_chunks": 2, "indexed_ids_sha256": "ids"}}
    kb = {"count": 2, "ids_sha256": "ids"}
    validate_server(config, kb, "code")
    with pytest.raises(ValueError):
        validate_server(config, kb, "new code")
    with pytest.raises(ValueError):
        validate_server(config, {**kb, "ids_sha256": "new ids"}, "code")
