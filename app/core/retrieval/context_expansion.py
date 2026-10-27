"""Add a small amount of nearby source text without another retrieval/HyDE call."""
from app.models.schemas import ChunkType


def expand_local_context(ranked, kb, *, limit=8):
    selected = list(ranked[:limit])
    seen = {chunk.chunk_id for chunk, _ in selected}
    # A definition and the diagram that illustrates it share a page but rarely
    # share vocabulary, so retrieval alone misses the figure. Reserve a slot for
    # the same-page figure before generic neighbours can crowd it out.
    for chunk, _ in ranked[:2]:
        if chunk.page_number is None or len(selected) >= limit:
            continue
        for figure in kb.figures_on_page(chunk, limit=1):
            if figure.chunk_id not in seen:
                selected.append((figure, 0.0))
                seen.add(figure.chunk_id)
    for chunk, _ in ranked[:2]:
        if chunk.page_number is None or len(selected) >= limit:
            continue
        for nearby in kb.nearby_chunks(chunk, limit=2):
            if nearby.chunk_id in seen or nearby.type == ChunkType.audio:
                continue
            selected.append((nearby, 0.0))  # expanded context has no reranker score
            seen.add(nearby.chunk_id)
            if len(selected) >= limit:
                break
    return selected


def bound_context(ranked, *, max_chars=16000, max_chunks=8):
    """The checked context and generation context must be the same bounded set."""
    result = []
    remaining = max_chars
    for chunk, score in ranked[:max_chunks]:
        if remaining <= 0:
            break
        text = chunk.text[:min(4000, remaining)]
        if not text.strip():
            continue
        result.append((chunk.model_copy(update={"text": text, "chunk_length": len(text)}), score))
        remaining -= len(text)
    return result
