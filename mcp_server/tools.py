from pathlib import Path
from typing import Any, Literal

from app.rag_execution import run_policy_question
from rag.schemas import AnswerResult


ExecutionMode = Literal["classic", "langgraph"]


def _source_to_dict(source: dict[str, object]) -> dict[str, object]:
    """Convert one source entry into a JSON-friendly dictionary."""

    return {
        "chunk_id": source.get("chunk_id"),
        "source_path": source.get("source_path"),
        "title": source.get("title"),
        "heading": source.get("heading"),
        "score": source.get("score"),
        "preview": source.get("preview"),
    }


def answer_result_to_tool_response(result: AnswerResult) -> dict[str, Any]:
    """Convert an AnswerResult into an MCP-friendly tool response."""

    return {
        "question": result.question,
        "answer": result.answer,
        "refusal": result.refusal,
        "grounding_status": result.grounding_status,
        "model_name": result.model_name,
        "cited_chunk_ids": result.cited_chunk_ids,
        "sources": [
            _source_to_dict(source)
            for source in result.sources
        ],
        "debug": {
            "execution_mode": result.debug.get("execution_mode"),
            "grounding_reason": result.debug.get("grounding_reason"),
            "retrieved_chunk_count": result.debug.get("retrieved_chunk_count"),
        },
    }


def answer_policy_question_tool(
    question: str,
    index_dir: str | Path = ".cache/vector_store",
    top_k: int = 5,
    execution_mode: ExecutionMode = "classic",
) -> dict[str, Any]:
    """
    Answer a policy question through the selected RAG execution mode.

    This function is intentionally written as a plain Python tool first.
    Later it can be exposed through an MCP server.
    """

    result = run_policy_question(
        question=question,
        index_dir=index_dir,
        top_k=top_k,
        execution_mode=execution_mode,
    )

    return answer_result_to_tool_response(result)