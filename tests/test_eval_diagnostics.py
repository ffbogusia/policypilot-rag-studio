from app import eval_diagnostics
from eval.run_eval import EvaluationCase, EvaluationResult


def test_run_eval_diagnostics_returns_summary(monkeypatch) -> None:
    def fake_load_golden_questions(golden_path):
        assert str(golden_path) == "eval/golden_questions.jsonl"

        return [
            EvaluationCase(
                id="q001",
                question="Do privileged users need MFA?",
                expected_source="mfa_policy.md",
                expected_refusal=False,
                must_contain=["MFA"],
            )
        ]

    def fake_run_evaluation(cases, index_dir, top_k):
        assert len(cases) == 1
        assert str(index_dir) == ".cache/vector_store"
        assert top_k == 5

        return [
            EvaluationResult(
                id="q001",
                question="Do privileged users need MFA?",
                passed=True,
                expected_source="mfa_policy.md",
                actual_sources=["data/sample_policies/mfa_policy.md"],
                expected_refusal=False,
                actual_refusal=False,
                source_match=True,
                refusal_match=True,
                must_contain_match=True,
                grounding_status="PASS",
                answer_preview="Privileged users must use MFA.",
            )
        ]

    monkeypatch.setattr(
        eval_diagnostics,
        "load_golden_questions",
        fake_load_golden_questions,
    )
    monkeypatch.setattr(
        eval_diagnostics,
        "run_evaluation",
        fake_run_evaluation,
    )

    diagnostics = eval_diagnostics.run_eval_diagnostics()

    assert diagnostics["passed"] == 1
    assert diagnostics["failed"] == 0
    assert diagnostics["total"] == 1
    assert diagnostics["pass_rate"] == 1.0
    assert diagnostics["failed_questions"] == []