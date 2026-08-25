from mcp_server import smoke_check


def test_run_smoke_check_returns_status(monkeypatch) -> None:
    def fake_answer_policy_question(question, top_k, execution_mode):
        return {"answer": "Privileged users must use MFA."}

    def fake_search_policy_docs(query, top_k):
        return {
            "chunks": [
                {
                    "chunk_id": "mfa_policy.md::chunk_001",
                }
            ]
        }

    def fake_get_chunk_by_id(chunk_id):
        return {
            "found": True,
            "chunk_id": chunk_id,
            "chunk": {
                "text": "Privileged users must use MFA.",
            },
        }

    def fake_run_rag_eval(top_k):
        return {
            "total": 10,
            "pass_rate": 1.0,
        }

    monkeypatch.setattr(
        smoke_check,
        "answer_policy_question",
        fake_answer_policy_question,
    )
    monkeypatch.setattr(
        smoke_check,
        "search_policy_docs",
        fake_search_policy_docs,
    )
    monkeypatch.setattr(
        smoke_check,
        "get_chunk_by_id",
        fake_get_chunk_by_id,
    )
    monkeypatch.setattr(
        smoke_check,
        "run_rag_eval",
        fake_run_rag_eval,
    )

    result = smoke_check.run_smoke_check()

    assert result["answer_ok"] is True
    assert result["search_ok"] is True
    assert result["chunk_lookup_ok"] is True
    assert result["eval_ok"] is True
    assert result["sample_chunk_id"] == "mfa_policy.md::chunk_001"
    assert result["eval_pass_rate"] == 1.0