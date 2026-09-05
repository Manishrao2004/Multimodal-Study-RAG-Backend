"""Evaluation runner — plan Sec. 8, reproducing EduRAG's Table 2 (end-to-end
metrics) and Table 3 (three-way ablation).

Retrieval evaluation runs against the live index in each `RetrievalMode`, so
the ablation arms differ only in the retrieval path and nothing else.
Generation evaluation runs only on the full pipeline, since that is the system
being reported.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np

from app.config import Settings
from app.core.attribution.citation_validation import (
    citation_validity_rate,
    extract_and_validate_citations,
)
from app.core.attribution.token_grounding import compute_token_grounding, grounding_ratio
from app.core.generation.answer import generate_answer
from app.core.retrieval.pipeline import RetrievalIndex
from app.eval.benchmark import BenchmarkItem
from app.eval.metrics import (
    bertscore_f1,
    faithfulness,
    mean_reciprocal_rank,
    recall_at_k,
    rouge_l,
    token_f1,
)
from app.models.schemas import RetrievalMode

# Retrieval must fetch deep enough to score Recall@5 meaningfully; the reported
# top-k for answering is separate and smaller.
EVAL_RETRIEVAL_DEPTH = 10


@dataclass
class RetrievalScores:
    mode: str
    queries: int
    mrr: float
    recall_at_1: float
    recall_at_5: float

    def as_row(self) -> dict:
        return {
            "mode": self.mode,
            "queries": self.queries,
            "MRR": round(self.mrr, 4),
            "Recall@1": round(self.recall_at_1, 4),
            "Recall@5": round(self.recall_at_5, 4),
        }


@dataclass
class GenerationScores:
    queries: int
    bertscore_f1: float
    bertscore_backend: str
    rouge_l: float
    token_f1: float
    faithfulness: float
    citation_validity_rate: float
    citation_coverage: float
    token_grounding_ratio: float
    failures: int = 0

    def as_row(self) -> dict:
        return {
            "queries": self.queries,
            "BERTScore F1": round(self.bertscore_f1, 4),
            "BERTScore backend": self.bertscore_backend,
            "ROUGE-L": round(self.rouge_l, 4),
            "Token F1": round(self.token_f1, 4),
            "Faithfulness": round(self.faithfulness, 4),
            "Citation Validity Rate": round(self.citation_validity_rate, 4),
            "Citation Coverage": round(self.citation_coverage, 4),
            "Token Grounding Ratio": round(self.token_grounding_ratio, 4),
            "Generation failures": self.failures,
        }


@dataclass
class EvaluationReport:
    ablation: list[RetrievalScores] = field(default_factory=list)
    generation: GenerationScores | None = None
    visual_retrieval: RetrievalScores | None = None


async def evaluate_retrieval(
    items: list[BenchmarkItem],
    index: RetrievalIndex,
    mode: RetrievalMode,
    depth: int = EVAL_RETRIEVAL_DEPTH,
) -> RetrievalScores:
    retrieved_ids: list[list[str]] = []
    gold_ids: list[str] = []

    for item in items:
        ranked = await index.retrieve(item.query, top_k=depth, mode=mode)
        retrieved_ids.append([chunk.chunk_id for chunk, _ in ranked])
        gold_ids.append(item.gold_chunk_id)

    return RetrievalScores(
        mode=mode.value,
        queries=len(items),
        mrr=mean_reciprocal_rank(retrieved_ids, gold_ids),
        recall_at_1=recall_at_k(retrieved_ids, gold_ids, 1),
        recall_at_5=recall_at_k(retrieved_ids, gold_ids, 5),
    )


async def run_ablation(
    items: list[BenchmarkItem],
    index: RetrievalIndex,
    modes: list[RetrievalMode] | None = None,
) -> list[RetrievalScores]:
    """EduRAG Table 3: BM25-only -> hybrid (no HyDE/rerank) -> full system."""
    modes = modes or [RetrievalMode.bm25_only, RetrievalMode.hybrid, RetrievalMode.full]
    return [await evaluate_retrieval(items, index, mode) for mode in modes]


async def evaluate_generation(
    items: list[BenchmarkItem],
    index: RetrievalIndex,
    settings: Settings,
    top_k: int = 5,
    concurrency: int = 3,
) -> GenerationScores:
    semaphore = asyncio.Semaphore(concurrency)
    predictions: list[str] = []
    references: list[str] = []
    faithfulness_scores: list[float] = []
    validity_scores: list[float] = []
    cited_answers: list[float] = []
    grounding_scores: list[float] = []
    failures = 0
    lock = asyncio.Lock()

    async def run_one(item: BenchmarkItem) -> None:
        nonlocal failures
        async with semaphore:
            ranked = await index.retrieve(item.query, top_k=top_k, mode=RetrievalMode.full)
            if not ranked:
                async with lock:
                    failures += 1
                return
            chunks = [c for c, _ in ranked]
            try:
                answer = await generate_answer(item.query, chunks, settings)
            except Exception:
                async with lock:
                    failures += 1
                return

            chunk_texts = [c.text for c in chunks]
            citations = extract_and_validate_citations(answer, chunks)
            tokens = compute_token_grounding(answer, "\n".join(chunk_texts))

            async with lock:
                predictions.append(answer)
                references.append(item.reference_answer)
                faithfulness_scores.append(
                    faithfulness(answer, chunk_texts, settings.embedding_model)
                )
                # Two distinct failures, kept apart. Citation Validity Rate is
                # "of the citations present, how many point at a real retrieved
                # chunk" — the paper's definition, and averaging uncited answers
                # in as 0.0 would silently redefine it. An answer that cites
                # nothing is a *coverage* failure instead, tracked separately.
                if citations:
                    validity_scores.append(citation_validity_rate(citations))
                cited_answers.append(1.0 if citations else 0.0)
                grounding_scores.append(grounding_ratio(tokens, settings.grounding_threshold))

    await asyncio.gather(*(run_one(i) for i in items))

    if not predictions:
        return GenerationScores(0, 0.0, "none", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, failures)

    bert_scores, backend = bertscore_f1(predictions, references)

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    return GenerationScores(
        queries=len(predictions),
        bertscore_f1=mean(bert_scores),
        bertscore_backend=backend,
        rouge_l=mean([rouge_l(p, r) for p, r in zip(predictions, references)]),
        token_f1=mean([token_f1(p, r) for p, r in zip(predictions, references)]),
        faithfulness=mean(faithfulness_scores),
        citation_validity_rate=mean(validity_scores),
        citation_coverage=mean(cited_answers),
        token_grounding_ratio=mean(grounding_scores),
        failures=failures,
    )
