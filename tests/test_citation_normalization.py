from app.core.attribution.citation_validation import (
    extract_and_validate_citations,
    normalize_citation_markers,
)
from tests.conftest import make_chunk


def test_provider_bracket_variants_are_canonicalized_and_validated():
    answer = "First claim【1】, second claim〔2〕, repeated claim［1］."
    normalized = normalize_citation_markers(answer)
    assert normalized == "First claim[1], second claim[2], repeated claim[1]."
    citations = extract_and_validate_citations(answer, [make_chunk("a", "a"), make_chunk("b", "b")])
    assert [(citation.marker, citation.valid) for citation in citations] == [(1, True), (2, True)]
