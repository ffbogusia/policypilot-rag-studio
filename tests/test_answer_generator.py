from rag.answer_generator import generate_answer_from_retrieval
from rag.schemas import ChunkResult, RetrievalResult


def _retrieval_result(chunks: list[ChunkResult]) -> RetrievalResult:
    return RetrievalResult(
        query="Can contractors access production data?",
        retrieval_mode="vector",
        top_k=5,
        chunks=chunks,
        index_path=".cache/vector_store/index.json",
        debug={"embedding_provider": "hash"},
    )


def _chunk() -> ChunkResult:
    return ChunkResult(
        chunk_id="access_policy.md::chunk_002",
        text=(
            "Contractors may access production systems only when there is a "
            "documented business need, manager approval, MFA, and a time-limited access window."
        ),
        source_path="data/sample_policies/access_policy.md",
        title="Production Access Policy",
        heading="Contractor access",
        score=0.8,
        retrieval_mode="vector",
        rank=1,
        metadata={"chunk_id": "access_policy.md::chunk_002"},
    )


def test_generate_answer_from_retrieval_returns_grounded_answer() -> None:
    result = generate_answer_from_retrieval(
        question="Can contractors access production data?",
        retrieval_result=_retrieval_result([_chunk()]),
    )

    assert result.refusal is False
    assert result.grounding_status == "PASS"
    assert result.cited_chunk_ids == ["access_policy.md::chunk_002"]
    assert "Contractors may access production systems" in result.answer


def test_generate_answer_refuses_when_no_chunks_are_available() -> None:
    result = generate_answer_from_retrieval(
        question="What is the CEO's favorite restaurant?",
        retrieval_result=_retrieval_result([]),
    )

    assert result.refusal is True
    assert result.grounding_status == "PASS"
    assert result.cited_chunk_ids == []
    assert "not have enough information" in result.answer


def test_generate_answer_refuses_sensitive_out_of_scope_question() -> None:
    result = generate_answer_from_retrieval(
        question="What is the CEO's favorite restaurant?",
        retrieval_result=_retrieval_result([_chunk()]),
    )

    assert result.refusal is True
    assert result.cited_chunk_ids == []