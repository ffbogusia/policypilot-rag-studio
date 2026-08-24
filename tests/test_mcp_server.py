import mcp_server.server as server


def test_mcp_server_imports() -> None:
    assert server.mcp is not None


def test_answer_policy_question_wrapper(monkeypatch) -> None:
    def fake_tool(question, index_dir, top_k, execution_mode):
        assert question == "Do privileged users need MFA?"
        assert str(index_dir) == ".cache/vector_store"
        assert top_k == 3
        assert execution_mode == "langgraph"

        return {
            "answer": "Privileged users must use MFA.",
            "refusal": False,
            "grounding_status": "PASS",
        }

    monkeypatch.setattr(
        "mcp_server.server.answer_policy_question_tool",
        fake_tool,
    )

    response = server.answer_policy_question(
        question="Do privileged users need MFA?",
        top_k=3,
        execution_mode="langgraph",
    )

    assert response["answer"] == "Privileged users must use MFA."
    assert response["refusal"] is False
    assert response["grounding_status"] == "PASS"


def test_search_policy_docs_wrapper(monkeypatch) -> None:
    def fake_tool(query, index_dir, top_k):
        assert query == "Do privileged users need MFA?"
        assert str(index_dir) == ".cache/vector_store"
        assert top_k == 2

        return {
            "query": query,
            "chunks": [
                {
                    "chunk_id": "mfa_policy.md::chunk_001",
                    "source_path": "data/sample_policies/mfa_policy.md",
                }
            ],
        }

    monkeypatch.setattr(
        "mcp_server.server.search_policy_docs_tool",
        fake_tool,
    )

    response = server.search_policy_docs(
        query="Do privileged users need MFA?",
        top_k=2,
    )

    assert response["query"] == "Do privileged users need MFA?"
    assert response["chunks"][0]["chunk_id"] == "mfa_policy.md::chunk_001"


def test_get_chunk_by_id_wrapper(monkeypatch) -> None:
    def fake_tool(chunk_id, index_dir):
        assert chunk_id == "mfa_policy.md::chunk_001"
        assert str(index_dir) == ".cache/vector_store"

        return {
            "found": True,
            "chunk_id": chunk_id,
            "chunk": {
                "text": "Privileged users must use MFA.",
            },
        }

    monkeypatch.setattr(
        "mcp_server.server.get_chunk_by_id_tool",
        fake_tool,
    )

    response = server.get_chunk_by_id("mfa_policy.md::chunk_001")

    assert response["found"] is True
    assert response["chunk"]["text"] == "Privileged users must use MFA."


def test_run_rag_eval_wrapper(monkeypatch) -> None:
    def fake_tool(golden_path, index_dir, top_k):
        assert str(golden_path) == "eval/golden_questions.jsonl"
        assert str(index_dir) == ".cache/vector_store"
        assert top_k == 5

        return {
            "passed": 1,
            "failed": 0,
            "total": 1,
            "pass_rate": 1.0,
        }

    monkeypatch.setattr(
        "mcp_server.server.run_rag_eval_tool",
        fake_tool,
    )

    response = server.run_rag_eval()

    assert response["passed"] == 1
    assert response["failed"] == 0
    assert response["total"] == 1