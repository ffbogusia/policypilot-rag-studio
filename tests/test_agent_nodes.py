import pytest

from agent.nodes import (
    check_evidence_node,
    generate_answer_node,
    retrieve_chunks_node,
)
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


def _retrieval_result() -> RetrievalResult:
    return RetrievalResult(
        query="Do privileged users need MFA?",
        retrieval_mode="vector",
        top_k=3,
        chunks=[_chunk()],
        index_path=".cache/vector_store/index.json",
        debug={"embedding_provider": "hash"},
    )


def test_retrieve_chunks_node_adds_retrieved_chunks(monkeypatch) -> None:
    def fake_retrieve(query, mode, top_k, index_dir):
        assert query == "Do privileged users need MFA?"
        assert mode == "vector"
        assert top_k == 3
        assert str(index_dir) == ".cache/vector_store"

        return _retrieval_result()

    monkeypatch.setattr("agent.nodes.retrieve", fake_retrieve)

    state = create_initial_state("Do privileged users need MFA?")

    state = create_initial_state(
        question="Do privileged users need MFA?",
        top_k=3,
    )

    new_state = retrieve_chunks_node(state=state)

    assert new_state["question"] == "Do privileged users need MFA?"
    assert new_state["retrieval_result"] is not None
    assert len(new_state["retrieved_chunks"]) == 1
    assert new_state["retrieved_chunks"][0].chunk_id == "mfa_policy.md::chunk_001"
    assert new_state["debug"]["retrieval"]["retrieved_chunk_count"] == 1
    assert new_state["debug"]["retrieval"]["embedding_provider"] == "hash"


def test_check_evidence_node_sets_evidence_decision() -> None:
    state = create_initial_state("Do privileged users need MFA?")
    state["retrieved_chunks"] = [_chunk()]

    new_state = check_evidence_node(state)

    assert new_state["has_enough_evidence"] is True
    assert new_state["debug"]["evidence_check"]["has_enough_evidence"] is True
    assert new_state["debug"]["evidence_check"]["retrieved_chunk_count"] == 1


def test_generate_answer_node_creates_answer_result() -> None:
    state = create_initial_state("Do privileged users need MFA?")
    state["retrieval_result"] = _retrieval_result()
    state["retrieved_chunks"] = [_chunk()]
    state["has_enough_evidence"] = True

    new_state = generate_answer_node(state)

    assert new_state["answer_result"] is not None
    assert new_state["answer_result"].refusal is False
    assert "MFA" in new_state["answer_result"].answer
    assert new_state["debug"]["answer_generation"]["refusal"] is False
    assert new_state["debug"]["answer_generation"]["source_count"] == 1


def test_generate_answer_node_requires_retrieval_result() -> None:
    state = create_initial_state("Do privileged users need MFA?")

    with pytest.raises(ValueError, match="Run retrieve_chunks_node first"):
        generate_answer_node(state)