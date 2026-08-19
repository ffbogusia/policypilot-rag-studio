import agent.langgraph_workflow as workflow
from rag.schemas import AnswerResult


def test_run_rag_workflow_returns_answer_result(monkeypatch) -> None:
    def fake_retrieve_chunks_node(state):
        return {
            **state,
            "retrieved_chunks": [],
            "debug": {
                **state.get("debug", {}),
                "retrieval": {"fake": True},
            },
        }

    def fake_check_evidence_node(state):
        return {
            **state,
            "has_enough_evidence": True,
            "debug": {
                **state.get("debug", {}),
                "evidence_check": {"fake": True},
            },
        }

    def fake_generate_answer_node(state):
        return {
            **state,
            "answer_result": AnswerResult(
                question=state["question"],
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
                model_name="langgraph-test-double",
                debug={},
            ),
        }

    monkeypatch.setattr(
        workflow,
        "retrieve_chunks_node",
        fake_retrieve_chunks_node,
    )
    monkeypatch.setattr(
        workflow,
        "check_evidence_node",
        fake_check_evidence_node,
    )
    monkeypatch.setattr(
        workflow,
        "generate_answer_node",
        fake_generate_answer_node,
    )

    result = workflow.run_rag_workflow("Do privileged users need MFA?")

    assert result.question == "Do privileged users need MFA?"
    assert result.refusal is False
    assert result.grounding_status == "PASS"
    assert "MFA" in result.answer