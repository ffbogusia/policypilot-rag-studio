from eval.run_eval import EvaluationCase, EvaluationResult
from mcp_server.tools import (
    answer_policy_question_tool,
    answer_result_to_tool_response,
    get_chunk_by_id_tool,
    run_rag_eval_tool,
    search_policy_docs_tool,
)
from rag.schemas import AnswerResult, ChunkResult, RetrievalResult


def _answer_result() -> AnswerResult:
    return AnswerResult(
        question="Do privileged users need MFA?",
        answer="Privileged users must use MFA.",
        cited_chunk_ids=["mfa_policy.md::chunk_001"],
        sources=[
            {
                "chunk_id": "mfa_policy.md::chunk_001",
                "source_path": "data/sample_policies/mfa_policy.md",
                "title": "MFA Policy",
                "heading": "Privileged access",
                "score": 0.9,
                "preview": "Privileged users must use MFA.",
            }
        ],
        grounding_status="PASS",
        refusal=False,
        model_name="deterministic-fallback",
        debug={
            "execution_mode": "classic",
            "grounding_reason": "Answer cites retrieved chunks.",
            "retrieved_chunk_count": 1,
        },
    )


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


def test_answer_result_to_tool_response_is_json_friendly() -> None:
    response = answer_result_to_tool_response(_answer_result())

    assert response["question"] == "Do privileged users need MFA?"
    assert response["answer"] == "Privileged users must use MFA."
    assert response["refusal"] is False
    assert response["grounding_status"] == "PASS"
    assert response["cited_chunk_ids"] == ["mfa_policy.md::chunk_001"]
    assert response["sources"][0]["source_path"] == "data/sample_policies/mfa_policy.md"
    assert response["debug"]["execution_mode"] == "classic"


def test_answer_policy_question_tool_uses_selected_execution_mode(monkeypatch) -> None:
    def fake_run_policy_question(question, index_dir, top_k, execution_mode):
        assert question == "Do privileged users need MFA?"
        assert str(index_dir) == ".cache/vector_store"
        assert top_k == 4
        assert execution_mode == "langgraph"

        return _answer_result()

    monkeypatch.setattr(
        "mcp_server.tools.run_policy_question",
        fake_run_policy_question,
    )

    response = answer_policy_question_tool(
        question="Do privileged users need MFA?",
        top_k=4,
        execution_mode="langgraph",
    )

    assert response["answer"] == "Privileged users must use MFA."
    assert response["sources"][0]["chunk_id"] == "mfa_policy.md::chunk_001"


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


def test_get_chunk_by_id_tool_returns_matching_chunk(monkeypatch) -> None:
    def fake_load_vector_index(index_dir):
        assert str(index_dir) == ".cache/vector_store"

        return {
            "chunks": [
                {
                    "chunk_id": "mfa_policy.md::chunk_001",
                    "text": "Privileged users must use MFA.",
                    "source_path": "data/sample_policies/mfa_policy.md",
                    "title": "MFA Policy",
                    "heading": "Privileged access",
                    "metadata": {"category": "security"},
                }
            ]
        }

    monkeypatch.setattr(
        "mcp_server.tools.load_vector_index",
        fake_load_vector_index,
    )

    response = get_chunk_by_id_tool("mfa_policy.md::chunk_001")

    assert response["found"] is True
    assert response["chunk_id"] == "mfa_policy.md::chunk_001"
    assert response["chunk"]["source_path"] == "data/sample_policies/mfa_policy.md"
    assert response["chunk"]["text"] == "Privileged users must use MFA."


def test_get_chunk_by_id_tool_returns_not_found(monkeypatch) -> None:
    def fake_load_vector_index(index_dir):
        return {"chunks": []}

    monkeypatch.setattr(
        "mcp_server.tools.load_vector_index",
        fake_load_vector_index,
    )

    response = get_chunk_by_id_tool("missing::chunk_999")

    assert response["found"] is False
    assert response["chunk_id"] == "missing::chunk_999"
    assert response["chunk"] is None


def test_run_rag_eval_tool_returns_summary(monkeypatch) -> None:
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
        "mcp_server.tools.load_golden_questions",
        fake_load_golden_questions,
    )
    monkeypatch.setattr(
        "mcp_server.tools.run_evaluation",
        fake_run_evaluation,
    )

    response = run_rag_eval_tool()

    assert response["passed"] == 1
    assert response["failed"] == 0
    assert response["total"] == 1
    assert response["pass_rate"] == 1.0
    assert response["results"][0]["id"] == "q001"
    assert response["results"][0]["passed"] is True