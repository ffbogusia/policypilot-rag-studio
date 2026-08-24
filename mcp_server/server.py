from pathlib import Path
from typing import Any, Literal

from mcp.server import MCPServer

from mcp_server.tools import (
    answer_policy_question_tool,
    get_chunk_by_id_tool,
    run_rag_eval_tool,
    search_policy_docs_tool,
)


ExecutionMode = Literal["classic", "langgraph"]

mcp = MCPServer("PolicyPilot RAG Studio")


@mcp.tool()
def answer_policy_question(
    question: str,
    index_dir: str = ".cache/vector_store",
    top_k: int = 5,
    execution_mode: ExecutionMode = "classic",
) -> dict[str, Any]:
    """
    Answer a policy question using the selected RAG execution mode.

    Use this tool when you need a grounded answer with citations,
    refusal status and grounding metadata.
    """

    return answer_policy_question_tool(
        question=question,
        index_dir=Path(index_dir),
        top_k=top_k,
        execution_mode=execution_mode,
    )


@mcp.tool()
def search_policy_docs(
    query: str,
    index_dir: str = ".cache/vector_store",
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Search policy documents and return retrieved chunks as evidence.

    Use this tool when you need source chunks without generating
    a final answer.
    """

    return search_policy_docs_tool(
        query=query,
        index_dir=Path(index_dir),
        top_k=top_k,
    )


@mcp.tool()
def get_chunk_by_id(
    chunk_id: str,
    index_dir: str = ".cache/vector_store",
) -> dict[str, Any]:
    """
    Return one indexed chunk by chunk ID.

    Use this tool to inspect a specific chunk returned by search,
    citations or the RAG Debugger.
    """

    return get_chunk_by_id_tool(
        chunk_id=chunk_id,
        index_dir=Path(index_dir),
    )


@mcp.tool()
def run_rag_eval(
    golden_path: str = "eval/golden_questions.jsonl",
    index_dir: str = ".cache/vector_store",
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Run the local RAG evaluation and return a structured summary.

    Use this tool to inspect expected source matching,
    refusal behavior and required phrase checks.
    """

    return run_rag_eval_tool(
        golden_path=Path(golden_path),
        index_dir=Path(index_dir),
        top_k=top_k,
    )