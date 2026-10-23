from app.core.retrieval.sufficiency import assess_evidence_sufficiency
from app.models.schemas import EvidenceSufficiency, RetrievalMode
from tests.conftest import make_chunk


def test_directly_covered_question_is_strong():
    ranked = [
        (
            make_chunk(
                "c1",
                "A sleeping partner contributes capital but does not participate in management.",
            ),
            0.91,
        )
    ]
    assert assess_evidence_sufficiency(
        "What is a sleeping partner?", ranked, RetrievalMode.full
    ) == EvidenceSufficiency.strong


def test_unrelated_evidence_is_insufficient():
    ranked = [(make_chunk("c1", "Photosynthesis converts light into chemical energy."), 0.05)]
    assert assess_evidence_sufficiency(
        "Explain partnership liability", ranked, RetrievalMode.full
    ) == EvidenceSufficiency.insufficient
