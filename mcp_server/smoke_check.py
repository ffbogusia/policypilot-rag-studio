from __future__ import annotations

from mcp_server.server import (
    answer_policy_question,
    get_chunk_by_id,
    run_rag_eval,
    search_policy_docs,
)


def run_smoke_check() -> dict[str, object]:
    """
    Run a small end-to-end check of the MCP tool wrappers.

    This does not start an external MCP client. It verifies that the Python
    wrapper functions exposed by the MCP server can call the underlying RAG tools.
    """

    question = "Do privileged users need MFA?"

    answer_response = answer_policy_question(
        question=question,
        top_k=3,
        execution_mode="classic",
    )

    search_response = search_policy_docs(
        query=question,
        top_k=3,
    )

    chunks = search_response.get("chunks", [])
    first_chunk_id = chunks[0]["chunk_id"] if chunks else None

    chunk_response = (
        get_chunk_by_id(first_chunk_id)
        if first_chunk_id
        else {"found": False, "chunk_id": None, "chunk": None}
    )

    eval_response = run_rag_eval(top_k=3)

    return {
        "answer_ok": bool(answer_response.get("answer")),
        "search_ok": bool(chunks),
        "chunk_lookup_ok": bool(chunk_response.get("found")),
        "eval_ok": eval_response.get("total", 0) > 0,
        "sample_chunk_id": first_chunk_id,
        "eval_pass_rate": eval_response.get("pass_rate"),
    }


def main() -> None:
    result = run_smoke_check()

    print("PolicyPilot MCP smoke check")
    print("---------------------------")
    print(f"Answer tool OK:      {result['answer_ok']}")
    print(f"Search tool OK:      {result['search_ok']}")
    print(f"Chunk lookup OK:     {result['chunk_lookup_ok']}")
    print(f"Evaluation tool OK:  {result['eval_ok']}")
    print(f"Sample chunk ID:     {result['sample_chunk_id']}")
    print(f"Eval pass rate:      {result['eval_pass_rate']}")

    if not all(
        [
            result["answer_ok"],
            result["search_ok"],
            result["chunk_lookup_ok"],
            result["eval_ok"],
        ]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()