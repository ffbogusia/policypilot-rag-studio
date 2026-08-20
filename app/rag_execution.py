from pathlib import Path
from typing import Literal

from agent.langgraph_workflow import run_rag_workflow
from rag.answer_generator import answer_question
from rag.schemas import AnswerResult


ExecutionMode = Literal["classic", "langgraph"]


def run_policy_question(
    question: str,
    index_dir: str | Path = ".cache/vector_store",
    top_k: int = 5,
    execution_mode: ExecutionMode = "classic",
) -> AnswerResult:
    """Run a policy question through the selected RAG execution mode."""

    if execution_mode == "classic":
        result = answer_question(
            question=question,
            index_dir=index_dir,
            top_k=top_k,
        )
        result.debug["execution_mode"] = "classic"
        return result

    if execution_mode == "langgraph":
        return run_rag_workflow(
            question=question,
            index_dir=index_dir,
            top_k=top_k,
        )

    raise ValueError(f"Unsupported execution mode: {execution_mode}")