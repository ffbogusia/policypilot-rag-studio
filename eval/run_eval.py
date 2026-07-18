import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.answer_generator import answer_question
from rag.schemas import AnswerResult


DEFAULT_GOLDEN_QUESTIONS_PATH = Path("eval/golden_questions.jsonl")
DEFAULT_INDEX_DIR = Path(".cache/vector_store")
DEFAULT_REPORT_PATH = Path("eval/eval_report.md")


@dataclass(frozen=True)
class EvaluationCase:
    """One expected RAG behavior from the golden questions file."""

    id: str
    question: str
    expected_source: str | None
    expected_refusal: bool
    must_contain: list[str]


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation result for one question."""

    id: str
    question: str
    passed: bool
    expected_source: str | None
    actual_sources: list[str]
    expected_refusal: bool
    actual_refusal: bool
    source_match: bool
    refusal_match: bool
    must_contain_match: bool
    grounding_status: str
    answer_preview: str


def load_golden_questions(
    path: str | Path = DEFAULT_GOLDEN_QUESTIONS_PATH,
) -> list[EvaluationCase]:
    """Load evaluation cases from a JSONL file."""

    golden_path = Path(path)
    cases: list[EvaluationCase] = []

    for line_number, line in enumerate(golden_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue

        raw_case: dict[str, Any] = json.loads(line)

        cases.append(
            EvaluationCase(
                id=str(raw_case["id"]),
                question=str(raw_case["question"]),
                expected_source=raw_case["expected_source"],
                expected_refusal=bool(raw_case["expected_refusal"]),
                must_contain=list(raw_case.get("must_contain", [])),
            )
        )

    return cases


def _source_paths(answer_result: AnswerResult) -> list[str]:
    """Extract source paths from an AnswerResult."""

    return [
        str(source.get("source_path", ""))
        for source in answer_result.sources
    ]


def _source_matches(
    answer_result: AnswerResult,
    expected_source: str | None,
) -> bool:
    """Check whether one of the cited sources matches the expected file."""

    if expected_source is None:
        return True

    return any(
        source_path.endswith(expected_source)
        for source_path in _source_paths(answer_result)
    )


def _answer_contains_all(answer: str, required_phrases: list[str]) -> bool:
    """Check whether the answer contains all required phrases."""

    normalized_answer = answer.lower()

    return all(
        phrase.lower() in normalized_answer
        for phrase in required_phrases
    )


def evaluate_answer(
    case: EvaluationCase,
    answer_result: AnswerResult,
) -> EvaluationResult:
    """Compare one RAG answer with one golden evaluation case."""

    source_match = _source_matches(
        answer_result=answer_result,
        expected_source=case.expected_source,
    )
    refusal_match = answer_result.refusal == case.expected_refusal

    if case.expected_refusal:
        must_contain_match = True
    else:
        must_contain_match = _answer_contains_all(
            answer=answer_result.answer,
            required_phrases=case.must_contain,
        )

    passed = source_match and refusal_match and must_contain_match

    return EvaluationResult(
        id=case.id,
        question=case.question,
        passed=passed,
        expected_source=case.expected_source,
        actual_sources=_source_paths(answer_result),
        expected_refusal=case.expected_refusal,
        actual_refusal=answer_result.refusal,
        source_match=source_match,
        refusal_match=refusal_match,
        must_contain_match=must_contain_match,
        grounding_status=answer_result.grounding_status,
        answer_preview=answer_result.answer[:220],
    )


def run_evaluation(
    cases: list[EvaluationCase],
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    top_k: int = 5,
) -> list[EvaluationResult]:
    """Run the RAG pipeline for all evaluation cases."""

    results: list[EvaluationResult] = []

    for case in cases:
        answer_result = answer_question(
            question=case.question,
            index_dir=index_dir,
            top_k=top_k,
        )

        results.append(
            evaluate_answer(
                case=case,
                answer_result=answer_result,
            )
        )

    return results


def print_results(results: list[EvaluationResult]) -> None:
    """Print a simple CLI evaluation report."""

    passed_count = sum(result.passed for result in results)
    total_count = len(results)

    print()
    print(f"Evaluation summary: {passed_count}/{total_count} passed")
    print("=" * 72)

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        actual_sources = ", ".join(result.actual_sources) or "-"

        print(f"[{status}] {result.id}: {result.question}")
        print(f"  Expected refusal: {result.expected_refusal}")
        print(f"  Actual refusal:   {result.actual_refusal}")
        print(f"  Expected source:  {result.expected_source or '-'}")
        print(f"  Actual sources:   {actual_sources}")
        print(f"  Source match:     {result.source_match}")
        print(f"  Refusal match:    {result.refusal_match}")
        print(f"  Must contain:     {result.must_contain_match}")
        print(f"  Grounding:        {result.grounding_status}")
        print(f"  Answer preview:   {result.answer_preview}")
        print("-" * 72)

def build_markdown_report(results: list[EvaluationResult]) -> str:
    """Build a Markdown evaluation report."""

    passed_count = sum(result.passed for result in results)
    total_count = len(results)

    lines = [
        "# RAG Evaluation Report",
        "",
        f"**Summary:** {passed_count}/{total_count} passed",
        "",
        "This report was generated locally from golden questions.",
        "",
        "## Results",
        "",
        "| ID | Status | Question | Expected source | Actual refusal | Grounding |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        expected_source = result.expected_source or "-"
        question = result.question.replace("|", "\\|")

        lines.append(
            "| "
            f"{result.id} | "
            f"{status} | "
            f"{question} | "
            f"{expected_source} | "
            f"{result.actual_refusal} | "
            f"{result.grounding_status} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This evaluation uses synthetic policy documents only.",
            "- The default local hash embedding provider is deterministic but not semantically strong.",
            "- Failing cases should be inspected with the RAG Debugger before changing the pipeline.",
            "",
        ]
    )

    return "\n".join(lines)


def write_markdown_report(
    results: list[EvaluationResult],
    output_path: str | Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Write a Markdown evaluation report to disk."""

    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_markdown_report(results), encoding="utf-8")

    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local RAG evaluation.")
    parser.add_argument(
        "--golden",
        default=str(DEFAULT_GOLDEN_QUESTIONS_PATH),
        help="Path to golden questions JSONL file.",
    )
    parser.add_argument(
        "--index",
        default=str(DEFAULT_INDEX_DIR),
        help="Path to local vector index folder.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per question.",
    )
    parser.add_argument(
        "--report-out",
        default=str(DEFAULT_REPORT_PATH),
        help="Path where the Markdown evaluation report will be written.",
    )

    args = parser.parse_args()

    cases = load_golden_questions(args.golden)
    results = run_evaluation(
        cases=cases,
        index_dir=args.index,
        top_k=args.top_k,
    )

    print_results(results)

    report_path = write_markdown_report(
        results=results,
        output_path=args.report_out,
    )

    print(f"Markdown report written to: {report_path}")


if __name__ == "__main__":
    main()