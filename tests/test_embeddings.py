from pathlib import Path

from ingestion.chunk_documents import chunk_documents
from ingestion.create_embeddings import HashEmbeddingProvider, embed_chunks
from ingestion.load_documents import load_markdown_documents


SAMPLE_DOCS_DIR = Path("data/sample_policies")


def test_hash_embedding_provider_returns_vectors() -> None:
    provider = HashEmbeddingProvider(dimension=16)

    vectors = provider.embed_texts(["MFA is required.", "Report phishing emails."])

    assert len(vectors) == 2
    assert len(vectors[0]) == 16
    assert len(vectors[1]) == 16


def test_hash_embeddings_are_deterministic() -> None:
    provider = HashEmbeddingProvider(dimension=16)

    first_vector = provider.embed_texts(["MFA is required."])[0]
    second_vector = provider.embed_texts(["MFA is required."])[0]

    assert first_vector == second_vector


def test_embed_chunks_preserves_chunk_metadata() -> None:
    documents = load_markdown_documents(SAMPLE_DOCS_DIR)
    chunks = chunk_documents(documents)

    provider = HashEmbeddingProvider(dimension=16)
    embedded_chunks = embed_chunks(chunks, provider)

    assert len(embedded_chunks) == len(chunks)

    first_embedded_chunk = embedded_chunks[0]

    assert first_embedded_chunk.chunk.chunk_id
    assert first_embedded_chunk.vector
    assert first_embedded_chunk.dimension == 16
    assert first_embedded_chunk.embedding_provider == "hash"
    assert first_embedded_chunk.embedding_model == "hash-16"
    assert first_embedded_chunk.metadata["embedding_dimension"] == 16
    assert first_embedded_chunk.metadata["chunk_id"] == first_embedded_chunk.chunk.chunk_id


def test_empty_text_returns_zero_vector() -> None:
    provider = HashEmbeddingProvider(dimension=8)

    vector = provider.embed_texts([""])[0]

    assert vector == [0.0] * 8