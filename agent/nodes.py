from pathlib import Path
from typing import Any

from agent.state import RagWorkflowState
from rag.answer_generator import _has_enough_evidence, generate_answer_from_retrieval
from retrieval.retriever import retrieve


def retrieve_chunks_node(
    state: RagWorkflowState,
    index_dir: str | Path = ".cache/vector_store",
    top_k: int = 5,
) -> RagWorkflowState:
    """
    Retrieve relevant chunks for the question stored in workflow state.

    This function is designed like a LangGraph node:
    it receives state and returns updated state.
    """

    resolved_index_dir = state.get("index_dir", str(index_dir))
    resolved_top_k = state.get("top_k", top_k)

    retrieval_result = retrieve(
        query=state["question"],
        mode="vector",
        top_k=resolved_top_k,
        index_dir=resolved_index_dir,
    )

    debug: dict[str, Any] = dict(state.get("debug", {}))
    debug["retrieval"] = {
        "retrieval_mode": retrieval_result.retrieval_mode,
        "top_k": retrieval_result.top_k,
        "index_path": str(retrieval_result.index_path),
        "retrieved_chunk_count": len(retrieval_result.chunks),
        **retrieval_result.debug,
    }

    return {
        **state,
        "retrieval_result": retrieval_result,
        "retrieved_chunks": retrieval_result.chunks,
        "debug": debug,
    }


def check_evidence_node(
    state: RagWorkflowState,
    min_top_score: float = 0.05,
) -> RagWorkflowState:
    """
    Check whether retrieved chunks contain enough evidence for an answer.

    This node does not generate the answer yet.
    It only writes the evidence decision back into workflow state.
    """

    chunks = state.get("retrieved_chunks", [])
    has_enough_evidence = _has_enough_evidence(
        question=state["question"],
        chunks=chunks,
        min_top_score=min_top_score,
    )

    top_score = max(
        (chunk.score for chunk in chunks),
        default=0.0,
    )

    debug: dict[str, Any] = dict(state.get("debug", {}))
    debug["evidence_check"] = {
        "has_enough_evidence": has_enough_evidence,
        "retrieved_chunk_count": len(chunks),
        "top_score": round(top_score, 4),
        "min_top_score": min_top_score,
    }

    return {
        **state,
        "has_enough_evidence": has_enough_evidence,
        "debug": debug,
    }


def generate_answer_node(
    state: RagWorkflowState,
    max_sources: int = 3,
) -> RagWorkflowState:
    """
    Generate the final grounded answer from the retrieval result.

    This node expects that retrieval has already happened.
    """

    retrieval_result = state.get("retrieval_result")
    resolved_max_sources = state.get("max_sources", max_sources)

    if retrieval_result is None:
        raise ValueError(
            "generate_answer_node requires retrieval_result in workflow state. "
            "Run retrieve_chunks_node first."
        )

    answer_result = generate_answer_from_retrieval(
        question=state["question"],
        retrieval_result=retrieval_result,
        max_sources=resolved_max_sources,
    )

    debug: dict[str, Any] = dict(state.get("debug", {}))
    debug["answer_generation"] = {
        "refusal": answer_result.refusal,
        "grounding_status": answer_result.grounding_status,
        "cited_chunk_count": len(answer_result.cited_chunk_ids),
        "source_count": len(answer_result.sources),
    }

    return {
        **state,
        "answer_result": answer_result,
        "debug": debug,
    }