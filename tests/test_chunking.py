from pathlib import Path

import pytest

from ingestion.chunk_documents import chunk_document, chunk_documents
from ingestion.load_documents import load_markdown_documents


SAMPLE_DOCS_DIR = Path("data/sample_policies")


def test_chunk_documents_returns_chunks() -> None:
    documents = load_markdown_documents(SAMPLE_DOCS_DIR)
    chunks = chunk_documents(documents)

    assert len(chunks) > 0


def test_chunks_have_required_fields() -> None:
    documents = load_markdown_documents(SAMPLE_DOCS_DIR)
    chunks = chunk_documents(documents)

    for chunk in chunks:
        assert chunk.chunk_id
        assert chunk.doc_id
        assert chunk.title
        assert chunk.text
        assert chunk.source_path
        assert chunk.chunk_index >= 0
        assert chunk.metadata


def test_chunk_metadata_preserves_source_information() -> None:
    documents = load_markdown_documents(SAMPLE_DOCS_DIR)
    chunks = chunk_documents(documents)

    access_chunks = [
        chunk for chunk in chunks if chunk.source_path.endswith("access_policy.md")
    ]

    assert len(access_chunks) > 0

    first_chunk = access_chunks[0]

    assert first_chunk.doc_id == "access_policy.md"
    assert first_chunk.title == "Production Access Policy"
    assert first_chunk.category == "security"
    assert first_chunk.metadata["source_path"].endswith("access_policy.md")
    assert first_chunk.metadata["chunk_id"] == first_chunk.chunk_id


def test_chunk_ids_are_stable_and_ordered() -> None:
    documents = load_markdown_documents(SAMPLE_DOCS_DIR)
    access_policy = next(
        document for document in documents if document.doc_id == "access_policy.md"
    )

    chunks = chunk_document(access_policy)

    assert chunks[0].chunk_id == "access_policy.md::chunk_000"

    for index, chunk in enumerate(chunks):
        assert chunk.chunk_index == index
        assert chunk.chunk_id.endswith(f"chunk_{index:03d}")


def test_chunk_size_limit_is_respected_for_small_limit() -> None:
    documents = load_markdown_documents(SAMPLE_DOCS_DIR)
    chunks = chunk_documents(documents, chunk_size=250, chunk_overlap=50)

    for chunk in chunks:
        assert len(chunk.text) <= 250


def test_invalid_chunk_overlap_raises_error() -> None:
    documents = load_markdown_documents(SAMPLE_DOCS_DIR)

    with pytest.raises(ValueError):
        chunk_documents(documents, chunk_size=100, chunk_overlap=100)