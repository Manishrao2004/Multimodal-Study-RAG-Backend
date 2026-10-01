from app.core.retrieval.bm25_index import BM25Index
from tests.conftest import make_chunk


def test_bm25_recovers_a_conservative_corpus_typo_match():
    index = BM25Index()
    index.build(
        [
            make_chunk("sole", "A sole proprietorship is owned by one person."),
            make_chunk("other", "A cooperative is owned by its members."),
        ]
    )

    hits = index.search("sole propietorship", top_k=2)

    assert hits[0][0] == "sole"
