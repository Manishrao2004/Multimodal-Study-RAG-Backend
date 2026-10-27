from app.models.schemas import ChunkType
from app.core.retrieval.context_expansion import expand_local_context, bound_context
from app.db.knowledge_base import KnowledgeBase
from app.tests.conftest import chunk


def test_expansion_keeps_document_boundaries_and_does_not_retrieve_again(settings):
    kb = KnowledgeBase(settings.kb_db_path)
    heading = chunk("Transformer", chunk_id="heading")
    nearby = chunk(chunk_id="nearby")
    distant = chunk(chunk_id="distant").model_copy(update={"page_number": 10})
    other = chunk(chunk_id="other").model_copy(update={"source_file": "other.pdf"})
    kb.add_chunks([heading, nearby, distant, other])
    expanded = expand_local_context([(heading, .9)], kb)
    assert [c.chunk_id for c, _ in expanded] == ["heading", "nearby"]
    assert expanded[1][1] == 0
    assert kb.get_chunk("heading").text == "Transformer"


def test_no_duplicate_neighbors_and_a_bounded_context(settings):
    kb = KnowledgeBase(settings.kb_db_path)
    chunks = [chunk(chunk_id=f"c{i}") for i in range(10)]
    kb.add_chunks(chunks)
    ranked = [(c, .9) for c in chunks[:3]]
    expanded = expand_local_context(ranked, kb, limit=4)
    ids = [c.chunk_id for c, _ in expanded]
    assert len(ids) <= 4 and len(ids) == len(set(ids))
    large = [(chunk("x" * 10000, chunk_id=str(i)), .8) for i in range(10)]
    bounded = bound_context(large)
    assert sum(len(c.text) for c, _ in bounded) <= 16000
    assert all(len(c.text) <= 4000 for c, _ in bounded)
    assert large[0][0].text == "x" * 10000


def test_same_page_figure_is_not_crowded_out_by_text_neighbors(settings):
    kb = KnowledgeBase(settings.kb_db_path)
    top = chunk("Functional organization is...", chunk_id="top")
    texts = [chunk(f"filler {i}", chunk_id=f"t{i}") for i in range(3)]
    figure = chunk("An organizational chart ...", chunk_id="fig").model_copy(
        update={"type": ChunkType.visual, "image_ref": "fig.png"})
    other_page = figure.model_copy(update={"chunk_id": "fig2", "page_number": 9})
    kb.add_chunks([top, *texts, figure, other_page])
    expanded = expand_local_context([(top, .9)], kb)
    ids = [c.chunk_id for c, _ in expanded]
    assert "fig" in ids and "fig2" not in ids
    assert ids.index("fig") == 1
