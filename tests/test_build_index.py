import json
from pathlib import Path

from ingestion.build_index import build_local_index, load_vector_index


SAMPLE_DOCS_DIR = Path("data/sample_policies")


def test_build_local_index_creates_index_file(tmp_path: Path) -> None:
    index_path = build_local_index(
        docs_dir=SAMPLE_DOCS_DIR,
        output_dir=tmp_path,
        provider_name="hash",
    )

    assert index_path.exists()
    assert index_path.name == "index.json"


def test_saved_index_contains_embedded_chunks(tmp_path: Path) -> None:
    index_path = build_local_index(
        docs_dir=SAMPLE_DOCS_DIR,
        output_dir=tmp_path,
        provider_name="hash",
    )

    index_data = json.loads(index_path.read_text(encoding="utf-8"))

    assert index_data["schema_version"] == "1.0"
    assert index_data["embedding_provider"] == "hash"
    assert index_data["embedding_model"] == "hash-64"
    assert index_data["embedding_dimension"] == 64
    assert index_data["chunk_count"] > 0
    assert len(index_data["chunks"]) == index_data["chunk_count"]

    first_chunk = index_data["chunks"][0]

    assert first_chunk["chunk_id"]
    assert first_chunk["doc_id"]
    assert first_chunk["title"]
    assert first_chunk["text"]
    assert first_chunk["source_path"]
    assert first_chunk["vector"]
    assert len(first_chunk["vector"]) == 64
    assert first_chunk["metadata"]


def test_load_vector_index_reads_saved_index(tmp_path: Path) -> None:
    build_local_index(
        docs_dir=SAMPLE_DOCS_DIR,
        output_dir=tmp_path,
        provider_name="hash",
    )

    index_data = load_vector_index(tmp_path)

    assert index_data["chunk_count"] > 0
    assert index_data["embedding_provider"] == "hash"
    assert "chunks" in index_data