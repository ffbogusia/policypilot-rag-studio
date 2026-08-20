import pytest

from app.rag_execution import run_policy_question
from rag.schemas import AnswerResult


def _answer(model_name: str) -> AnswerResult:
    return AnswerResult(
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
        model_name=model_name,
        debug={},
    )


def test_run_policy_question_uses_classic_mode(monkeypatch) -> None:
    def fake_answer_question(question, index_dir, top_k):
        assert question == "Do privileged users need MFA?"
        assert str(index_dir) == ".cache/vector_store"
        assert top_k == 4

        return _answer("classic-test-double")

    monkeypatch.setattr("app.rag_execution.answer_question", fake_answer_question)

    result = run_policy_question(
        question="Do privileged users need MFA?",
        top_k=4,
        execution_mode="classic",
    )

    assert result.model_name == "classic-test-double"
    assert result.debug["execution_mode"] == "classic"


def test_run_policy_question_uses_langgraph_mode(monkeypatch) -> None:
    def fake_run_rag_workflow(question, index_dir, top_k):
        assert question == "Do privileged users need MFA?"
        assert str(index_dir) == ".cache/vector_store"
        assert top_k == 4

        result = _answer("langgraph-test-double")
        result.debug["execution_mode"] = "langgraph"
        return result

    monkeypatch.setattr("app.rag_execution.run_rag_workflow", fake_run_rag_workflow)

    result = run_policy_question(
        question="Do privileged users need MFA?",
        top_k=4,
        execution_mode="langgraph",
    )

    assert result.model_name == "langgraph-test-double"
    assert result.debug["execution_mode"] == "langgraph"


def test_run_policy_question_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported execution mode"):
        run_policy_question(
            question="Do privileged users need MFA?",
            execution_mode="unknown",  # type: ignore[arg-type]
        )