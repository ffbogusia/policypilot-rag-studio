from pathlib import Path
from typing import Any

from ingestion.build_index import load_vector_index


def get_vector_index_status(
    index_dir: str | Path = ".cache/vector_store",
) -> dict[str, Any]:
    """Return a JSON-friendly status summary for the local vector index."""

    index_path = Path(index_dir) / "index.json"

    if not index_path.exists():
        return {
            "exists": False,
            "index_path": str(index_path),
            "chunk_count": 0,
            "embedding_provider": None,
            "embedding_model": None,
            "embedding_dimension": None,
        }

    index_data = load_vector_index(index_dir)
    chunks = index_data.get("chunks", [])

    return {
        "exists": True,
        "index_path": str(index_path),
        "chunk_count": len(chunks),
        "embedding_provider": index_data.get("embedding_provider"),
        "embedding_model": index_data.get("embedding_model"),
        "embedding_dimension": index_data.get("embedding_dimension"),
    }