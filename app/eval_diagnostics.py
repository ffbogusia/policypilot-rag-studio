from pathlib import Path
from typing import Any

from eval.run_eval import (
    build_evaluation_summary,
    load_golden_questions,
    run_evaluation,
)


def run_eval_diagnostics(
    golden_path: str | Path = "eval/golden_questions.jsonl",
    index_dir: str | Path = ".cache/vector_store",
    top_k: int = 5,
) -> dict[str, Any]:
    """Run local RAG evaluation and return a compact diagnostics payload."""

    cases = load_golden_questions(golden_path)
    results = run_evaluation(
        cases=cases,
        index_dir=index_dir,
        top_k=top_k,
    )

    summary = build_evaluation_summary(results)

    return {
        **summary,
        "golden_path": str(golden_path),
        "index_dir": str(index_dir),
        "top_k": top_k,
        "failed_questions": [
            {
                "id": result.id,
                "question": result.question,
                "expected_source": result.expected_source,
                "actual_sources": result.actual_sources,
                "grounding_status": result.grounding_status,
            }
            for result in results
            if not result.passed
        ],
    }