from typing import Any, NotRequired, TypedDict

from rag.schemas import AnswerResult, ChunkResult


class RagWorkflowState(TypedDict):
    """Shared state passed between RAG workflow steps."""

    question: str
    retrieved_chunks: NotRequired[list[ChunkResult]]
    has_enough_evidence: NotRequired[bool]
    answer_result: NotRequired[AnswerResult | None]
    debug: NotRequired[dict[str, Any]]


def create_initial_state(question: str) -> RagWorkflowState:
    """Create the initial state for a RAG workflow run."""

    return {
        "question": question,
        "retrieved_chunks": [],
        "has_enough_evidence": False,
        "answer_result": None,
        "debug": {},
    }


def summarize_state(state: RagWorkflowState) -> dict[str, Any]:
    """Return a small debug summary of the workflow state."""

    return {
        "question": state["question"],
        "retrieved_chunk_count": len(state.get("retrieved_chunks", [])),
        "has_enough_evidence": state.get("has_enough_evidence", False),
        "has_answer": state.get("answer_result") is not None,
    }