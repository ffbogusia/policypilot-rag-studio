from agent.state import create_initial_state, summarize_state


def test_create_initial_state_contains_question() -> None:
    state = create_initial_state("Can employees use public AI tools?")

    assert state["question"] == "Can employees use public AI tools?"
    assert state["retrieval_result"] is None
    assert state["retrieved_chunks"] == []
    assert state["has_enough_evidence"] is False
    assert state["answer_result"] is None
    assert state["debug"] == {}


def test_summarize_state_returns_debug_summary() -> None:
    state = create_initial_state("Do privileged users need MFA?")

    summary = summarize_state(state)

    assert summary["question"] == "Do privileged users need MFA?"
    assert summary["has_retrieval_result"] is False
    assert summary["retrieved_chunk_count"] == 0
    assert summary["has_enough_evidence"] is False
    assert summary["has_answer"] is False