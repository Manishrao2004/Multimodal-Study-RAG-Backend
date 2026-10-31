from app.core.attribution.citation_validation import normalize_citation_markers


def test_model_line_citation_maps_to_passage_marker():
    assert normalize_citation_markers("Supported claim【2†L1-L4】.") == "Supported claim[2]."


def test_ambiguous_or_unrelated_brackets_are_not_converted():
    assert normalize_citation_markers("【2†other】 【abc】") == "【2†other】 【abc】"
