import argparse
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    check_evidence_node,
    generate_answer_node,
    retrieve_chunks_node,
)
from agent.state import RagWorkflowState, create_initial_state
from rag.schemas import AnswerResult


def build_rag_workflow():
    """Build and compile the LangGraph RAG workflow."""

    workflow = StateGraph(RagWorkflowState)

    workflow.add_node("retrieve_chunks", retrieve_chunks_node)
    workflow.add_node("check_evidence", check_evidence_node)
    workflow.add_node("generate_answer", generate_answer_node)

    workflow.add_edge(START, "retrieve_chunks")
    workflow.add_edge("retrieve_chunks", "check_evidence")
    workflow.add_edge("check_evidence", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


def _attach_workflow_debug(
    answer_result: AnswerResult,
    final_state: RagWorkflowState,
) -> AnswerResult:
    """Attach LangGraph workflow debug data to the final answer result."""

    debug = dict(answer_result.debug)
    debug["execution_mode"] = "langgraph"
    debug["workflow"] = final_state.get("debug", {})

    return AnswerResult(
        question=answer_result.question,
        answer=answer_result.answer,
        cited_chunk_ids=answer_result.cited_chunk_ids,
        sources=answer_result.sources,
        grounding_status=answer_result.grounding_status,
        refusal=answer_result.refusal,
        model_name=answer_result.model_name,
        debug=debug,
    )


def run_rag_workflow(
    question: str,
    index_dir: str | Path = ".cache/vector_store",
    top_k: int = 5,
    max_sources: int = 3,
) -> AnswerResult:
    """Run the LangGraph RAG workflow for one question."""

    app = build_rag_workflow()
    initial_state = create_initial_state(
        question=question,
        index_dir=str(index_dir),
        top_k=top_k,
        max_sources=max_sources,
    )
    final_state = app.invoke(initial_state)

    answer_result = final_state.get("answer_result")

    if answer_result is None:
        raise RuntimeError("LangGraph workflow finished without an answer_result.")

    return _attach_workflow_debug(
        answer_result=answer_result,
        final_state=final_state,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LangGraph RAG workflow.")
    parser.add_argument(
        "question",
        nargs="?",
        default="Do privileged users need MFA?",
        help="Question to answer through the LangGraph workflow.",
    )
    parser.add_argument(
        "--index",
        default=".cache/vector_store",
        help="Path to the local vector index folder.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks.",
    )

    args = parser.parse_args()
    result = run_rag_workflow(
        question=args.question,
        index_dir=args.index,
        top_k=args.top_k,
    )

    print(f"Question: {result.question}")
    print(f"Model: {result.model_name}")
    print(f"Refusal: {result.refusal}")
    print(f"Grounding status: {result.grounding_status}")
    print()
    print("Answer:")
    print(result.answer)

    if result.sources:
        print()
        print("Sources:")

        for source in result.sources:
            print(
                f"- {source['chunk_id']} "
                f"({source['source_path']}, score={source['score']})"
            )


if __name__ == "__main__":
    main()