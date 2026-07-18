from eval.run_eval import (
    EvaluationCase,
    _answer_contains_all,
    evaluate_answer,
    load_golden_questions,
    build_markdown_report,
    write_markdown_report,
)
from rag.schemas import AnswerResult


def test_load_golden_questions_from_jsonl(tmp_path) -> None:
    golden_file = tmp_path / "golden_questions.jsonl"
    golden_file.write_text(
        (
            '{"id":"q001","question":"Do users need MFA?",'
            '"expected_source":"mfa_policy.md",'
            '"expected_refusal":false,'
            '"must_contain":["MFA"]}\n'
        )
    )

    cases = load_golden_questions(golden_file)

    assert len(cases) == 1
    assert cases[0].id == "q001"
    assert cases[0].question == "Do users need MFA?"
    assert cases[0].expected_source == "mfa_policy.md"
    assert cases[0].expected_refusal is False
    assert cases[0].must_contain == ["MFA"]


def test_answer_contains_all_is_case_insensitive() -> None:
    answer = "Employees must use MFA for privileged access."

    assert _answer_contains_all(answer, ["mfa", "PRIVILEGED"])


def test_evaluate_answer_passes_when_source_and_terms_match() -> None:
    case = EvaluationCase(
        id="q001",
        question="Do privileged users need MFA?",
        expected_source="mfa_policy.md",
        expected_refusal=False,
        must_contain=["MFA"],
    )
    answer_result = AnswerResult(
        question=case.question,
        answer="Privileged users must use MFA.",
        cited_chunk_ids=["mfa_policy.md::chunk_001"],
        sources=[
            {
                "chunk_id": "mfa_policy.md::chunk_001",
                "source_path": "data/sample_policies/mfa_policy.md",
                "score": 0.9,
            }
        ],
        grounding_status="PASS",
        refusal=False,
        model_name="deterministic-fallback",
        debug={},
    )

    result = evaluate_answer(case, answer_result)

    assert result.passed is True
    assert result.source_match is True
    assert result.refusal_match is True
    assert result.must_contain_match is True


def test_evaluate_answer_supports_expected_refusal() -> None:
    case = EvaluationCase(
        id="q999",
        question="What is the CEO's favorite restaurant?",
        expected_source=None,
        expected_refusal=True,
        must_contain=[],
    )
    answer_result = AnswerResult(
        question=case.question,
        answer="I do not have enough information in the provided policy documents to answer this safely.",
        cited_chunk_ids=[],
        sources=[],
        grounding_status="PASS",
        refusal=True,
        model_name="deterministic-fallback",
        debug={},
    )

    result = evaluate_answer(case, answer_result)

    assert result.passed is True
    assert result.refusal_match is True

def test_write_markdown_report_creates_file(tmp_path) -> None:
    result = evaluate_answer(
        EvaluationCase(
            id="q001",
            question="Do privileged users need MFA?",
            expected_source="mfa_policy.md",
            expected_refusal=False,
            must_contain=["MFA"],
        ),
        AnswerResult(
            question="Do privileged users need MFA?",
            answer="Privileged users must use MFA.",
            cited_chunk_ids=["mfa_policy.md::chunk_001"],
            sources=[
                {
                    "chunk_id": "mfa_policy.md::chunk_001",
                    "source_path": "data/sample_policies/mfa_policy.md",
                    "score": 0.9,
                }
            ],
            grounding_status="PASS",
            refusal=False,
            model_name="deterministic-fallback",
            debug={},
        ),
    )

    report_text = build_markdown_report([result])

    assert "# RAG Evaluation Report" in report_text
    assert "**Summary:** 1/1 passed" in report_text
    assert "q001" in report_text

    report_path = write_markdown_report(
        results=[result],
        output_path=tmp_path / "eval_report.md",
    )

    assert report_path.exists()
    assert "Do privileged users need MFA?" in report_path.read_text()