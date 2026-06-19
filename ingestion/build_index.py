import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from ingestion.chunk_documents import chunk_documents
from ingestion.create_embeddings import create_embedding_provider, embed_chunks
from ingestion.load_documents import load_markdown_documents
from rag.schemas import EmbeddedChunkRecord


INDEX_FILE_NAME = "index.json"
def _make_json_safe(value: Any) -> Any:
    """
    Convert Python objects into JSON-safe values.

    YAML may parse values like 2026-06-01 as date objects.
    JSON cannot serialize date objects directly, so we convert them to strings.
    """

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, dict):
        return {
            str(key): _make_json_safe(item_value)
            for key, item_value in value.items()
        }

    if isinstance(value, list):
        return [_make_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [_make_json_safe(item) for item in value]

    if isinstance(value, str | int | float | bool) or value is None:
        return value

    return str(value)


def _embedded_chunk_to_dict(embedded_chunk: EmbeddedChunkRecord) -> dict[str, Any]:
    """Convert an EmbeddedChunkRecord into a JSON-serializable dictionary."""

    chunk = embedded_chunk.chunk

    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "title": chunk.title,
        "heading": chunk.heading,
        "text": chunk.text,
        "source_path": chunk.source_path,
        "chunk_index": chunk.chunk_index,
        "category": chunk.category,
        "vector": embedded_chunk.vector,
        "embedding_provider": embedded_chunk.embedding_provider,
        "embedding_model": embedded_chunk.embedding_model,
        "embedding_dimension": embedded_chunk.dimension,
        "metadata": _make_json_safe(embedded_chunk.metadata),
    }


def save_vector_index(
    embedded_chunks: list[EmbeddedChunkRecord],
    output_dir: str | Path,
) -> Path:
    """
    Save embedded chunks to a local JSON vector index.

    This is a lightweight local vector store format for learning, debugging and tests.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    index_path = output_path / INDEX_FILE_NAME

    embedding_provider = embedded_chunks[0].embedding_provider if embedded_chunks else None
    embedding_model = embedded_chunks[0].embedding_model if embedded_chunks else None
    embedding_dimension = embedded_chunks[0].dimension if embedded_chunks else 0

    index_data = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "chunk_count": len(embedded_chunks),
        "chunks": [
            _embedded_chunk_to_dict(embedded_chunk)
            for embedded_chunk in embedded_chunks
        ],
    }

    index_path.write_text(
        json.dumps(index_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return index_path


def load_vector_index(index_dir: str | Path) -> dict[str, Any]:
    """Load a local JSON vector index."""

    index_path = Path(index_dir) / INDEX_FILE_NAME

    if not index_path.exists():
        raise FileNotFoundError(f"Vector index not found: {index_path}")

    return json.loads(index_path.read_text(encoding="utf-8"))


def build_local_index(
    docs_dir: str | Path = "data/sample_policies",
    output_dir: str | Path = ".cache/vector_store",
    provider_name: str = "hash",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Path:
    """
    Build the local vector index from Markdown policy documents.

    Pipeline:
    load documents -> chunk documents -> create embeddings -> save local vector index
    """

    documents = load_markdown_documents(docs_dir)
    chunks = chunk_documents(documents)

    provider = create_embedding_provider(
        provider_name=provider_name,
        model_name=model_name,
    )

    embedded_chunks = embed_chunks(
        chunks=chunks,
        provider=provider,
    )

    return save_vector_index(
        embedded_chunks=embedded_chunks,
        output_dir=output_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local vector index.")
    parser.add_argument(
        "--docs",
        default="data/sample_policies",
        help="Path to Markdown policy documents.",
    )
    parser.add_argument(
        "--out",
        default=".cache/vector_store",
        help="Output folder for the local vector index.",
    )
    parser.add_argument(
        "--provider",
        choices=["hash", "sentence-transformers"],
        default="hash",
        help="Embedding provider to use.",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence Transformers model name.",
    )

    args = parser.parse_args()

    documents = load_markdown_documents(args.docs)
    chunks = chunk_documents(documents)

    provider = create_embedding_provider(
        provider_name=args.provider,
        model_name=args.model,
    )

    embedded_chunks = embed_chunks(
        chunks=chunks,
        provider=provider,
    )

    index_path = save_vector_index(
        embedded_chunks=embedded_chunks,
        output_dir=args.out,
    )

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Embedded chunks: {len(embedded_chunks)}")
    print(f"Embedding provider: {provider.provider_name}")
    print(f"Embedding model: {provider.model_name}")

    if embedded_chunks:
        print(f"Embedding dimension: {embedded_chunks[0].dimension}")

    print(f"Vector index saved to: {index_path}")


if __name__ == "__main__":
    main()