from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentRecord:
    """
    Represents one source document loaded from the sample policy folder.

    This is an early RAG pipeline object:
    raw Markdown file -> DocumentRecord -> chunks -> embeddings -> retrieval.
    """

    doc_id: str
    title: str
    source_path: str
    category: str
    text: str
    owner: str | None = None
    version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def preview(self, max_chars: int = 120) -> str:
        """Return a short one-line preview of the document text."""
        clean_text = " ".join(self.text.split())

        if len(clean_text) <= max_chars:
            return clean_text

        return clean_text[:max_chars].rstrip() + "..."


@dataclass(frozen=True)
class ChunkRecord:
    """
    Represents one searchable chunk created from a source document.

    Later RAG steps will embed and retrieve these chunks.
    """

    chunk_id: str
    doc_id: str
    title: str
    text: str
    source_path: str
    chunk_index: int
    heading: str | None = None
    category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def preview(self, max_chars: int = 120) -> str:
        """Return a short one-line preview of the chunk text."""
        clean_text = " ".join(self.text.split())

        if len(clean_text) <= max_chars:
            return clean_text

        return clean_text[:max_chars].rstrip() + "..."


@dataclass(frozen=True)
class EmbeddedChunkRecord:
    """
    Represents a chunk together with its embedding vector.

    This is the bridge between chunking and vector search.
    """

    chunk: ChunkRecord
    vector: list[float]
    embedding_provider: str
    embedding_model: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dimension(self) -> int:
        """Return the number of values in the embedding vector."""
        return len(self.vector)
    
@dataclass(frozen=True)
class ChunkResult:
    """
    Represents one retrieved chunk returned by a retriever.
    """

    chunk_id: str
    text: str
    source_path: str
    title: str
    score: float
    retrieval_mode: str
    rank: int
    heading: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def preview(self, max_chars: int = 120) -> str:
        """Return a short one-line preview of the retrieved chunk."""
        clean_text = " ".join(self.text.split())

        if len(clean_text) <= max_chars:
            return clean_text

        return clean_text[:max_chars].rstrip() + "..."


@dataclass(frozen=True)
class RetrievalResult:
    """
    Represents the full retrieval response for one user query.
    """

    query: str
    retrieval_mode: str
    top_k: int
    chunks: list[ChunkResult]
    index_path: str
    debug: dict[str, Any] = field(default_factory=dict)