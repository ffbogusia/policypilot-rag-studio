import argparse
import math
from pathlib import Path
from typing import Any

from ingestion.build_index import INDEX_FILE_NAME, load_vector_index
from ingestion.create_embeddings import create_embedding_provider
from rag.schemas import ChunkResult, RetrievalResult


def cosine_similarity(first_vector: list[float], second_vector: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Cosine similarity measures how similar two vectors are by direction.
    Higher score means more similar.
    """

    if len(first_vector) != len(second_vector):
        raise ValueError(
            f"Vector dimensions do not match: "
            f"{len(first_vector)} != {len(second_vector)}"
        )

    first_norm = math.sqrt(sum(value * value for value in first_vector))
    second_norm = math.sqrt(sum(value * value for value in second_vector))

    if first_norm == 0 or second_norm == 0:
        return 0.0

    dot_product = sum(
        first_value * second_value
        for first_value, second_value in zip(first_vector, second_vector)
    )

    return dot_product / (first_norm * second_norm)


def _index_chunk_to_result(
    index_chunk: dict[str, Any],
    score: float,
    rank: int,
    retrieval_mode: str,
) -> ChunkResult:
    """Convert one stored index chunk into a retrieval result."""

    return ChunkResult(
        chunk_id=str(index_chunk["chunk_id"]),
        text=str(index_chunk["text"]),
        source_path=str(index_chunk["source_path"]),
        title=str(index_chunk["title"]),
        heading=index_chunk.get("heading"),
        score=score,
        retrieval_mode=retrieval_mode,
        rank=rank,
        metadata=index_chunk.get("metadata", {}),
    )


def search_vector_index(
    query: str,
    index_dir: str | Path = ".cache/vector_store",
    top_k: int = 5,
) -> RetrievalResult:
    """
    Search a local JSON vector index using cosine similarity.

    Steps:
    1. Load local index.
    2. Embed user query with the same provider used by the index.
    3. Compare query vector with every chunk vector.
    4. Return top_k chunks ranked by similarity.
    """

    clean_query = query.strip()

    if not clean_query:
        raise ValueError("query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    index_data = load_vector_index(index_dir)

    provider_name = str(index_data.get("embedding_provider") or "hash")
    model_name = str(
        index_data.get("embedding_model")
        or "sentence-transformers/all-MiniLM-L6-v2"
    )

    provider = create_embedding_provider(
        provider_name=provider_name,
        model_name=model_name,
    )

    query_vector = provider.embed_texts([clean_query])[0]

    scored_chunks: list[tuple[float, dict[str, Any]]] = []

    for index_chunk in index_data.get("chunks", []):
        chunk_vector = index_chunk.get("vector", [])
        score = cosine_similarity(query_vector, chunk_vector)
        scored_chunks.append((score, index_chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)

    top_results = scored_chunks[:top_k]

    chunks = [
        _index_chunk_to_result(
            index_chunk=index_chunk,
            score=score,
            rank=rank,
            retrieval_mode="vector",
        )
        for rank, (score, index_chunk) in enumerate(top_results, start=1)
    ]

    index_path = Path(index_dir) / INDEX_FILE_NAME

    return RetrievalResult(
        query=clean_query,
        retrieval_mode="vector",
        top_k=top_k,
        chunks=chunks,
        index_path=index_path.as_posix(),
        debug={
            "embedding_provider": provider_name,
            "embedding_model": model_name,
            "total_index_chunks": len(index_data.get("chunks", [])),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the local vector index.")
    parser.add_argument(
        "query",
        nargs="?",
        default="Do admins need MFA?",
        help="Question or search query.",
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
        help="Number of chunks to retrieve.",
    )

    args = parser.parse_args()

    result = search_vector_index(
        query=args.query,
        index_dir=args.index,
        top_k=args.top_k,
    )

    print(f"Query: {result.query}")
    print(f"Retrieval mode: {result.retrieval_mode}")
    print(f"Index: {result.index_path}")
    print(f"Embedding provider: {result.debug['embedding_provider']}")
    print(f"Embedding model: {result.debug['embedding_model']}")
    print(f"Total index chunks: {result.debug['total_index_chunks']}")
    print()

    for chunk in result.chunks:
        print(f"rank: {chunk.rank}")
        print(f"score: {chunk.score:.4f}")
        print(f"source: {chunk.source_path}")
        print(f"title: {chunk.title}")
        print(f"heading: {chunk.heading}")
        print(f"chunk_id: {chunk.chunk_id}")
        print(f"preview: {chunk.preview(160)}")
        print("-" * 80)


if __name__ == "__main__":
    main()