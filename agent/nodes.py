from pathlib import Path
from typing import Any

from agent.state import RagWorkflowState
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

    retrieval_result = retrieve(
        query=state["question"],
        mode="vector",
        top_k=top_k,
        index_dir=index_dir,
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
        "retrieved_chunks": retrieval_result.chunks,
        "debug": debug,
    }