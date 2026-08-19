import argparse

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


def run_rag_workflow(question: str) -> AnswerResult:
    """Run the LangGraph RAG workflow for one question."""

    app = build_rag_workflow()
    final_state = app.invoke(create_initial_state(question))

    answer_result = final_state.get("answer_result")

    if answer_result is None:
        raise RuntimeError("LangGraph workflow finished without an answer_result.")

    return answer_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LangGraph RAG workflow.")
    parser.add_argument(
        "question",
        nargs="?",
        default="Do privileged users need MFA?",
        help="Question to answer through the LangGraph workflow.",
    )

    args = parser.parse_args()
    result = run_rag_workflow(args.question)

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