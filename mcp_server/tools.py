from pathlib import Path
from typing import Any, Literal

from app.rag_execution import run_policy_question
from eval.run_eval import (
    build_evaluation_summary,
    load_golden_questions,
    run_evaluation,
)
from ingestion.build_index import load_vector_index
from rag.schemas import AnswerResult, ChunkResult
from retrieval.retriever import retrieve


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


def _chunk_to_dict(chunk: ChunkResult) -> dict[str, object]:
    """Convert one retrieved chunk into a JSON-friendly dictionary."""

    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "source_path": chunk.source_path,
        "title": chunk.title,
        "heading": chunk.heading,
        "score": chunk.score,
        "retrieval_mode": chunk.retrieval_mode,
        "rank": chunk.rank,
        "preview": chunk.preview(220),
    }


def _eval_result_to_dict(result) -> dict[str, Any]:
    """Convert one evaluation result into a JSON-friendly dictionary."""

    return {
        "id": result.id,
        "question": result.question,
        "passed": result.passed,
        "expected_source": result.expected_source,
        "actual_sources": result.actual_sources,
        "expected_refusal": result.expected_refusal,
        "actual_refusal": result.actual_refusal,
        "source_match": result.source_match,
        "refusal_match": result.refusal_match,
        "must_contain_match": result.must_contain_match,
        "grounding_status": result.grounding_status,
        "answer_preview": result.answer_preview,
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
        "sources": [_source_to_dict(source) for source in result.sources],
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


def search_policy_docs_tool(
    query: str,
    index_dir: str | Path = ".cache/vector_store",
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Search policy documents and return retrieved chunks.

    This is an MCP-ready search tool. It returns evidence chunks only,
    without generating a final answer.
    """

    retrieval_result = retrieve(
        query=query,
        mode="vector",
        top_k=top_k,
        index_dir=index_dir,
    )

    return {
        "query": retrieval_result.query,
        "retrieval_mode": retrieval_result.retrieval_mode,
        "top_k": retrieval_result.top_k,
        "index_path": str(retrieval_result.index_path),
        "chunks": [_chunk_to_dict(chunk) for chunk in retrieval_result.chunks],
        "debug": retrieval_result.debug,
    }


def get_chunk_by_id_tool(
    chunk_id: str,
    index_dir: str | Path = ".cache/vector_store",
) -> dict[str, Any]:
    """
    Return one indexed chunk by chunk ID.

    This is useful for inspecting a specific chunk returned by search or citation.
    """

    index_data = load_vector_index(index_dir)
    chunks = index_data.get("chunks", [])

    for chunk in chunks:
        if chunk.get("chunk_id") == chunk_id:
            return {
                "found": True,
                "chunk_id": chunk_id,
                "chunk": {
                    "chunk_id": chunk.get("chunk_id"),
                    "text": chunk.get("text"),
                    "source_path": chunk.get("source_path"),
                    "title": chunk.get("title"),
                    "heading": chunk.get("heading"),
                    "metadata": chunk.get("metadata", {}),
                },
            }

    return {
        "found": False,
        "chunk_id": chunk_id,
        "chunk": None,
    }

def run_rag_eval_tool(
    golden_path: str | Path = "eval/golden_questions.jsonl",
    index_dir: str | Path = ".cache/vector_store",
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Run local RAG evaluation and return a JSON-friendly summary.

    This tool does not write files. It returns structured evaluation results.
    """

    cases = load_golden_questions(golden_path)
    results = run_evaluation(
        cases=cases,
        index_dir=index_dir,
        top_k=top_k,
    )

    summary = build_evaluation_summary(results)

    return {
        "golden_path": str(golden_path),
        "index_dir": str(index_dir),
        "top_k": top_k,
        **summary,
        "results": [
            _eval_result_to_dict(result)
            for result in results
        ],
    }