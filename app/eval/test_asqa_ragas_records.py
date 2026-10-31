from app.eval.asqa_ragas_records import convert


def test_conversion_uses_actual_evidence_and_excludes_withheld():
    samples = [{"sample_id": "1", "annotations": [{"long_answer": "reference"}]}]
    response = {"answer_basis": "sources", "answer": "answer", "evidence": [{"chunk": {"text": "actual context"}}], "claim_verification": {}}
    base = {"sample_id": "1", "method": "aspect", "query": "question", "status": "ok", "response": response}
    result = convert([base], samples)[0]
    assert result["retrieved_contexts"] == ["actual context"]
    assert result["reference"] == "reference" and result["status"] == "ok"
    withheld = {**base, "response": {**response, "claim_verification": {"withheld": True}}}
    assert convert([withheld], samples)[0]["exclusion_reason"] == "source_answer_withheld"


def test_mixed_source_excerpt_is_explicitly_labelled():
    samples = [{"sample_id": "1", "annotations": [{"long_answer": "reference"}]}]
    response = {"answer_basis": "mixed", "answer": "## From your study material\n\n> excerpt\n\n## Additional explanation (general knowledge)\n\nbackground", "evidence": [{"chunk": {"text": "excerpt"}}], "claim_verification": {"withheld": True, "fallback_used": True}}
    row = {"sample_id": "1", "method": "aspect", "query": "question", "status": "ok", "response": response}
    result = convert([row], samples, source_sections=True)[0]
    assert result["status"] == "ok" and result["evaluated_answer_kind"] == "source_excerpts"
    assert result["response"]["answer"] == "> excerpt"


def test_source_only_excerpt_fallback_is_eligible_and_labelled():
    samples = [{"sample_id": "1", "annotations": [{"long_answer": "reference"}]}]
    response = {"answer_basis": "sources", "answer": "> excerpt", "evidence": [{"chunk": {"text": "excerpt"}}], "claim_verification": {"withheld": True, "fallback_used": True}}
    row = {"sample_id": "1", "method": "aspect", "query": "question", "status": "ok", "response": response}
    result = convert([row], samples, source_sections=True)[0]
    assert result["status"] == "ok" and result["evaluated_answer_kind"] == "source_excerpts"
