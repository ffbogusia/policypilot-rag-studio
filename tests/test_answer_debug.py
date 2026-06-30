from rag.answer_generator import generate_answer_from_retrieval
from rag.schemas import ChunkResult, RetrievalResult


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


def test_answer_debug_contains_retrieved_chunks() -> None:
    retrieval_result = RetrievalResult(
        query="Can contractors access production data?",
        retrieval_mode="vector",
        top_k=5,
        chunks=[_chunk()],
        index_path=".cache/vector_store/index.json",
        debug={"embedding_provider": "hash"},
    )

    result = generate_answer_from_retrieval(
        question="Can contractors access production data?",
        retrieval_result=retrieval_result,
    )

    retrieved_chunks = result.debug["retrieved_chunks"]

    assert len(retrieved_chunks) == 1
    assert retrieved_chunks[0]["rank"] == 1
    assert retrieved_chunks[0]["chunk_id"] == "access_policy.md::chunk_002"
    assert retrieved_chunks[0]["source_path"] == "data/sample_policies/access_policy.md"