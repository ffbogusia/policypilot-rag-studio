from rag.schemas import ChunkResult


def check_grounding(
    answer: str,
    cited_chunk_ids: list[str],
    retrieved_chunks: list[ChunkResult],
    refusal: bool = False,
    min_top_score: float = 0.05,
) -> dict[str, object]:
    """
    Check whether an answer is grounded in retrieved chunks.

    This is a simple deterministic MVP checker.
    It is not an LLM judge.
    """

    if refusal:
        if cited_chunk_ids:
            return {
                "status": "WARN",
                "reason": "Refusal answer should usually not cite sources.",
            }

        return {
            "status": "PASS",
            "reason": "Safe refusal without unsupported citations.",
        }

    if not answer.strip():
        return {
            "status": "FAIL",
            "reason": "Answer is empty.",
        }

    if not retrieved_chunks:
        return {
            "status": "FAIL",
            "reason": "No retrieved chunks were provided.",
        }

    if not cited_chunk_ids:
        return {
            "status": "FAIL",
            "reason": "Answer has no cited chunks.",
        }

    retrieved_chunk_ids = {chunk.chunk_id for chunk in retrieved_chunks}
    missing_citations = [
        chunk_id
        for chunk_id in cited_chunk_ids
        if chunk_id not in retrieved_chunk_ids
    ]

    if missing_citations:
        return {
            "status": "WARN",
            "reason": f"Some cited chunks were not in retrieved results: {missing_citations}",
        }

    top_score = max(chunk.score for chunk in retrieved_chunks)

    if top_score < min_top_score:
        return {
            "status": "WARN",
            "reason": f"Top retrieval score is low: {top_score:.4f}",
        }

    return {
        "status": "PASS",
        "reason": "Answer cites retrieved chunks and has enough retrieval evidence.",
    }