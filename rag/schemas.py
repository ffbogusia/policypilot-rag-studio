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