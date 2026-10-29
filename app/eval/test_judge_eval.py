import json

import pytest
from pydantic import ValidationError

from app.eval.judge_eval import Rubric, score_audit, provenance_sections, check_completion


def valid_audit():
    return dict(claims=[dict(text="Profit is INR 15000", quote="Profit is INR 15000", section_id=0, supported=True,
                            requires_citation=True, cited=True, citation_supported=True,
                            reason="The cited passage gives the same amount")],
                clarity=2, jargon_explained=2, instruction_following=2,
                regional_appropriateness=2, verification_caution=2,
                reason="Simple correct explanation")


def test_truncated_completion_is_rejected_even_when_json_looks_valid():
    data = {"choices": [{"finish_reason": "length", "message": {"content": json.dumps(valid_audit())}}]}
    with pytest.raises(ValueError, match="token budget"):
        check_completion(data)
    data["choices"][0]["finish_reason"] = "stop"
    assert check_completion(data) is data


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    async def chat(self, messages, temperature=0):
        self.calls.append(list(messages))
        return next(self.responses)


@pytest.mark.asyncio
async def test_single_json_fence_is_accepted_without_changing_verdicts():
    client = FakeClient(["```json\n" + json.dumps(valid_audit()) + "\n```"])
    audit, repairs = await score_audit(client, [])
    assert repairs == 0
    assert audit.model_dump() == valid_audit()


@pytest.mark.asyncio
async def test_schema_repair_retains_original_inputs_and_records_attempt():
    client = FakeClient(["not JSON", json.dumps(valid_audit())])
    audit, repairs = await score_audit(client, [{"role": "user", "content": "Original case"}])
    assert repairs == 1
    assert audit.claims[0].supported is True
    assert client.calls[1][0]["content"] == "Original case"
    assert "schema or consistency" in client.calls[1][-1]["content"]


@pytest.mark.asyncio
async def test_contradictory_annotations_do_not_become_perfect_scores():
    invalid = valid_audit()
    invalid["claims"][0]["supported"] = False
    client = FakeClient([json.dumps(invalid)] * 3)
    with pytest.raises(ValueError, match="cannot entail"):
        await score_audit(client, [])
    assert len(client.calls) == 3


def test_judge_verdicts_and_rubric_scores_are_strict():
    invalid = valid_audit()
    invalid["claims"][0]["supported"] = "true"
    with pytest.raises(ValidationError):
        Rubric.model_validate(invalid)
    invalid = valid_audit()
    invalid["clarity"] = 3
    with pytest.raises(ValidationError):
        Rubric.model_validate(invalid)


def test_explicit_background_has_separate_provenance():
    sections = provenance_sections("## From your study material\nFact.\n\n## Additional explanation (general knowledge)\nExample.", "mixed")
    assert sections[0]["provenance"] == "source"
    assert sections[1]["provenance"] == "background"
    assert "Example." in sections[1]["text"]
    with pytest.raises(ValueError, match="boundary"):
        provenance_sections("Unlabeled mixed content", "mixed")


@pytest.mark.asyncio
async def test_background_cannot_silently_be_counted_as_unsupported_source_claim():
    invalid = valid_audit()
    sections = [{"id": 0, "provenance": "background", "text": "Profit is INR 15000"}]
    invalid["claims"][0]["cited"] = False
    invalid["claims"][0]["citation_supported"] = False
    client = FakeClient([json.dumps(invalid)] * 3)
    with pytest.raises(ValueError, match="provenance"):
        await score_audit(client, [], sections)
