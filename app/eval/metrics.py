"""Evaluation metrics — plan Sec. 8, mirroring EduRAG's Table 2/3 design.

Retrieval:  MRR, Recall@k
Generation: BERTScore F1 (primary), ROUGE-L and Token F1 (secondary)
Trust:      Faithfulness, Citation Validity Rate, Token Grounding Ratio

BERTScore is the paper's primary generation metric. The real `bert-score`
package is used when installed; otherwise `bertscore_f1` falls back to cosine
similarity over the configured sentence-embedding model. The fallback is a
*different* measure — coarser, sentence-level rather than token-alignment —
so results carry `bertscore_backend` and the report labels which one produced
the numbers. Never present fallback values as BERTScore proper.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from app.core.retrieval.bm25_index import tokenize


# --- Retrieval -----------------------------------------------------------


def reciprocal_rank(retrieved_ids: list[str], relevant_id: str) -> float:
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id == relevant_id:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(all_retrieved: list[list[str]], relevant_ids: list[str]) -> float:
    if not all_retrieved:
        return 0.0
    return float(
        np.mean([reciprocal_rank(r, gold) for r, gold in zip(all_retrieved, relevant_ids)])
    )


def recall_at_k(all_retrieved: list[list[str]], relevant_ids: list[str], k: int) -> float:
    if not all_retrieved:
        return 0.0
    hits = [1.0 if gold in retrieved[:k] else 0.0 for retrieved, gold in zip(all_retrieved, relevant_ids)]
    return float(np.mean(hits))


# --- Lexical generation metrics -------------------------------------------


def token_f1(prediction: str, reference: str) -> float:
    """Bag-of-tokens F1, as used for extractive QA. Expected to be low for
    open-ended generated answers — the base paper reports the same effect and
    treats it as secondary."""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    # Row-wise DP keeps memory at O(min(len)) rather than O(n*m); long
    # transcript chunks make the full table wasteful.
    if len(b) < len(a):
        a, b = b, a
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b):
            if token_a == token_b:
                current.append(previous[j] + 1)
            else:
                current.append(max(current[j], previous[j + 1]))
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F-measure over the longest common subsequence."""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# --- Semantic generation metrics ------------------------------------------


def bertscore_available() -> bool:
    try:
        import bert_score  # noqa: F401
    except ImportError:
        return False
    return True


def bertscore_f1(predictions: list[str], references: list[str]) -> tuple[list[float], str]:
    """Returns (per-example F1, backend name).

    Backend is "bert-score" for the real metric, or "embedding-cosine" for the
    fallback — the caller must report which was used.
    """
    if not predictions:
        return [], "none"

    try:
        from bert_score import score as _bert_score

        _precision, _recall, f1 = _bert_score(
            predictions, references, lang="en", rescale_with_baseline=False, verbose=False
        )
        return [float(x) for x in f1], "bert-score"
    except ImportError:
        pass

    return embedding_similarity(predictions, references), "embedding-cosine"


def embedding_similarity(
    predictions: list[str], references: list[str], model_name: str = "BAAI/bge-small-en-v1.5"
) -> list[float]:
    """Cosine similarity between sentence embeddings of each pair."""
    from app.core.retrieval.faiss_index import get_embedding_model

    if not predictions:
        return []
    model = get_embedding_model(model_name)
    pred_vecs = np.asarray(
        model.encode(predictions, normalize_embeddings=True, show_progress_bar=False)
    )
    ref_vecs = np.asarray(
        model.encode(references, normalize_embeddings=True, show_progress_bar=False)
    )
    return [float(np.dot(p, r)) for p, r in zip(pred_vecs, ref_vecs)]


def faithfulness(answer: str, chunk_texts: list[str], model_name: str) -> float:
    """How closely the answer sticks to its retrieved evidence: similarity to
    the single best-matching chunk (EduRAG's deterministic stand-in for an
    LLM-judge faithfulness score)."""
    if not answer.strip() or not chunk_texts:
        return 0.0
    scores = embedding_similarity([answer] * len(chunk_texts), chunk_texts, model_name)
    return max(scores) if scores else 0.0
