"""Print one controlled evidence decision without touching the knowledge base."""
import argparse
import asyncio
import json

from app.config import get_settings
from app.core.generation.orchestration import evaluate_evidence, plan_question
from app.eval.validate import CASES, load_cases
from app.models.schemas import Chunk, ChunkType


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case_id")
    args = parser.parse_args()
    data = load_cases(CASES)
    case = next(c for c in data["cases"] if c["id"] == args.case_id)
    corpus = {c["id"]: c for c in data["corpus"]}
    settings = get_settings()
    plan = await plan_question(case["query"], [], settings)
    chunks = [Chunk(chunk_id=k, source_file=corpus[k]["source"], text=corpus[k]["text"],
                    chunk_length=len(corpus[k]["text"]), type=ChunkType.text) for k in case["contexts"]]
    decision = await evaluate_evidence(plan, chunks, settings)
    print(json.dumps({"plan": plan.model_dump(), "evidence_decision": decision.model_dump()}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
