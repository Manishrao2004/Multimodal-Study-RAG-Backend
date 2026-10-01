from app.core.retrieval.bm25_index import BM25Index
from tests.conftest import make_chunk


def test_single_passage_remains_searchable():
    index = BM25Index()
    index.build([make_chunk("only", "Recursion solves smaller instances of a problem.")])
    hits = index.search("recursion", top_k=5)
    assert len(hits) == 1
    assert hits[0][0] == "only"
    assert hits[0][1] > 0
    assert index.search("thermodynamics", top_k=5) == []


def test_common_terms_do_not_disappear_in_two_passages():
    index = BM25Index()
    index.build([
        make_chunk("a", "Recursion has a base case."),
        make_chunk("b", "Recursion reduces the input."),
    ])
    assert {identifier for identifier, _ in index.search("recursion", top_k=2)} == {"a", "b"}


def test_stopword_only_corpus_is_safe_and_returns_no_evidence():
    index = BM25Index()
    index.build([make_chunk("empty", "the and or is")])
    assert index.search("the", top_k=2) == []
    assert index.search("recursion", top_k=2) == []
