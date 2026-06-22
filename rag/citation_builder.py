from rag.schemas import ChunkResult


def build_citations(
    chunks: list[ChunkResult],
    max_sources: int = 3,
) -> tuple[list[str], list[dict[str, object]]]:
    """
    Build citation metadata from retrieved chunks.

    Returns:
    - cited chunk IDs
    - public source dictionaries for UI/debugging
    """

    cited_chunk_ids: list[str] = []
    sources: list[dict[str, object]] = []

    for chunk in chunks:
        if chunk.chunk_id in cited_chunk_ids:
            continue

        cited_chunk_ids.append(chunk.chunk_id)
        sources.append(
            {
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "heading": chunk.heading,
                "source_path": chunk.source_path,
                "score": round(chunk.score, 4),
                "rank": chunk.rank,
                "preview": chunk.preview(180),
            }
        )

        if len(sources) >= max_sources:
            break

    return cited_chunk_ids, sources