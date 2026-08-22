from mcp_server.tools import (
    answer_policy_question_tool,
    answer_result_to_tool_response,
    search_policy_docs_tool,
)

from rag.schemas import AnswerResult, ChunkResult, RetrievalResult


def _retrieved_chunk() -> ChunkResult:
    return ChunkResult(
        chunk_id="mfa_policy.md::chunk_001",
        text="Privileged users must use MFA when accessing sensitive systems.",
        source_path="data/sample_policies/mfa_policy.md",
        title="MFA Policy",
        heading="Privileged access",
        score=0.9,
        retrieval_mode="vector",
        rank=1,
        metadata={"chunk_id": "mfa_policy.md::chunk_001"},
    )


def test_search_policy_docs_tool_returns_retrieved_chunks(monkeypatch) -> None:
    def fake_retrieve(query, mode, top_k, index_dir):
        assert query == "Do privileged users need MFA?"
        assert mode == "vector"
        assert top_k == 3
        assert str(index_dir) == ".cache/vector_store"

        return RetrievalResult(
            query=query,
            retrieval_mode=mode,
            top_k=top_k,
            chunks=[_retrieved_chunk()],
            index_path=".cache/vector_store/index.json",
            debug={"embedding_provider": "hash"},
        )

    monkeypatch.setattr("mcp_server.tools.retrieve", fake_retrieve)

    response = search_policy_docs_tool(
        query="Do privileged users need MFA?",
        top_k=3,
    )

    assert response["query"] == "Do privileged users need MFA?"
    assert response["retrieval_mode"] == "vector"
    assert response["top_k"] == 3
    assert len(response["chunks"]) == 1
    assert response["chunks"][0]["chunk_id"] == "mfa_policy.md::chunk_001"
    assert response["chunks"][0]["source_path"] == "data/sample_policies/mfa_policy.md"
    assert response["debug"]["embedding_provider"] == "hash"