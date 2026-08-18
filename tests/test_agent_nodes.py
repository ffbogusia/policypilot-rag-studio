from agent.nodes import retrieve_chunks_node
from agent.state import create_initial_state
from rag.schemas import ChunkResult, RetrievalResult


def _chunk() -> ChunkResult:
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


def test_retrieve_chunks_node_adds_retrieved_chunks(monkeypatch) -> None:
    def fake_retrieve(query, mode, top_k, index_dir):
        assert query == "Do privileged users need MFA?"
        assert mode == "vector"
        assert top_k == 3
        assert str(index_dir) == ".cache/vector_store"

        return RetrievalResult(
            query=query,
            retrieval_mode=mode,
            top_k=top_k,
            chunks=[_chunk()],
            index_path=".cache/vector_store/index.json",
            debug={"embedding_provider": "hash"},
        )

    monkeypatch.setattr("agent.nodes.retrieve", fake_retrieve)

    state = create_initial_state("Do privileged users need MFA?")

    new_state = retrieve_chunks_node(
        state=state,
        top_k=3,
    )

    assert new_state["question"] == "Do privileged users need MFA?"
    assert len(new_state["retrieved_chunks"]) == 1
    assert new_state["retrieved_chunks"][0].chunk_id == "mfa_policy.md::chunk_001"
    assert new_state["debug"]["retrieval"]["retrieved_chunk_count"] == 1
    assert new_state["debug"]["retrieval"]["embedding_provider"] == "hash"