import argparse
import hashlib
import math
import re
from typing import Protocol

from ingestion.chunk_documents import chunk_documents
from ingestion.load_documents import load_markdown_documents
from rag.schemas import ChunkRecord, EmbeddedChunkRecord


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


class EmbeddingProvider(Protocol):
    """Small interface shared by all embedding providers."""

    provider_name: str
    model_name: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Convert texts into embedding vectors."""
        ...


class HashEmbeddingProvider:
    """
    Deterministic zero-dependency embedding fallback.

    This is useful for tests and offline debugging.
    It is not as semantically strong as Sentence Transformers.
    """

    provider_name = "hash"

    def __init__(self, dimension: int = 64) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0.")

        self.dimension = dimension
        self.model_name = f"hash-{dimension}"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = TOKEN_PATTERN.findall(text.lower())

        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            bucket = int(digest[:8], 16) % self.dimension
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vector[bucket] += sign

        return _normalize_vector(vector)


class SentenceTransformerEmbeddingProvider:
    """
    Local embedding provider using Sentence Transformers.

    Default model:
    sentence-transformers/all-MiniLM-L6-v2
    """

    provider_name = "sentence-transformers"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.model_name = model_name

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: python -m pip install sentence-transformers"
            ) from exc

        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [embedding.astype(float).tolist() for embedding in embeddings]


def _normalize_vector(vector: list[float]) -> list[float]:
    """Normalize a vector to unit length."""

    norm = math.sqrt(sum(value * value for value in vector))

    if norm == 0:
        return vector

    return [value / norm for value in vector]


def create_embedding_provider(
    provider_name: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> EmbeddingProvider:
    """Create an embedding provider by name."""

    if provider_name == "hash":
        return HashEmbeddingProvider()

    if provider_name == "sentence-transformers":
        return SentenceTransformerEmbeddingProvider(model_name=model_name)

    raise ValueError(f"Unknown embedding provider: {provider_name}")


def embed_chunks(
    chunks: list[ChunkRecord],
    provider: EmbeddingProvider,
) -> list[EmbeddedChunkRecord]:
    """
    Convert chunks into embedded chunks.

    This function does not save anything to a vector store yet.
    It only attaches vectors to chunks.
    """

    texts = [chunk.text for chunk in chunks]
    vectors = provider.embed_texts(texts)

    if len(vectors) != len(chunks):
        raise ValueError(
            f"Expected {len(chunks)} vectors, but got {len(vectors)}."
        )

    embedded_chunks: list[EmbeddedChunkRecord] = []

    for chunk, vector in zip(chunks, vectors):
        embedded_chunks.append(
            EmbeddedChunkRecord(
                chunk=chunk,
                vector=vector,
                embedding_provider=provider.provider_name,
                embedding_model=provider.model_name,
                metadata={
                    **chunk.metadata,
                    "embedding_provider": provider.provider_name,
                    "embedding_model": provider.model_name,
                    "embedding_dimension": len(vector),
                },
            )
        )

    return embedded_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Create local embeddings for policy chunks.")
    parser.add_argument(
        "--docs",
        default="data/sample_policies",
        help="Path to the folder with Markdown policy documents.",
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
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of embedded chunks to preview.",
    )

    args = parser.parse_args()

    documents = load_markdown_documents(args.docs)
    chunks = chunk_documents(documents)
    provider = create_embedding_provider(
        provider_name=args.provider,
        model_name=args.model,
    )
    embedded_chunks = embed_chunks(chunks, provider)

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Embedded chunks: {len(embedded_chunks)}")
    print(f"Embedding provider: {provider.provider_name}")
    print(f"Embedding model: {provider.model_name}")

    if embedded_chunks:
        print(f"Embedding dimension: {embedded_chunks[0].dimension}")

    for embedded_chunk in embedded_chunks[: args.limit]:
        vector_preview = embedded_chunk.vector[:5]

        print()
        print(f"- chunk_id: {embedded_chunk.chunk.chunk_id}")
        print(f"  title: {embedded_chunk.chunk.title}")
        print(f"  heading: {embedded_chunk.chunk.heading}")
        print(f"  dimension: {embedded_chunk.dimension}")
        print(f"  vector_preview: {vector_preview}")
        print(f"  text_preview: {embedded_chunk.chunk.preview(100)}")


if __name__ == "__main__":
    main()