from typing import Any, NotRequired, TypedDict

from rag.schemas import AnswerResult, ChunkResult, RetrievalResult


class RagWorkflowState(TypedDict):
    """Shared state passed between RAG workflow steps."""

    question: str
    index_dir: NotRequired[str]
    top_k: NotRequired[int]
    max_sources: NotRequired[int]
    retrieval_result: NotRequired[RetrievalResult | None]
    retrieved_chunks: NotRequired[list[ChunkResult]]
    has_enough_evidence: NotRequired[bool]
    answer_result: NotRequired[AnswerResult | None]
    debug: NotRequired[dict[str, Any]]


def create_initial_state(
    question: str,
    index_dir: str = ".cache/vector_store",
    top_k: int = 5,
    max_sources: int = 3,
) -> RagWorkflowState:
    """Create the initial state for a RAG workflow run."""

    return {
        "question": question,
        "index_dir": index_dir,
        "top_k": top_k,
        "max_sources": max_sources,
        "retrieval_result": None,
        "retrieved_chunks": [],
        "has_enough_evidence": False,
        "answer_result": None,
        "debug": {},
    }


def summarize_state(state: RagWorkflowState) -> dict[str, Any]:
    """Return a small debug summary of the workflow state."""

    return {
        "question": state["question"],
        "index_dir": state.get("index_dir", ".cache/vector_store"),
        "top_k": state.get("top_k", 5),
        "has_retrieval_result": state.get("retrieval_result") is not None,
        "retrieved_chunk_count": len(state.get("retrieved_chunks", [])),
        "has_enough_evidence": state.get("has_enough_evidence", False),
        "has_answer": state.get("answer_result") is not None,
    }