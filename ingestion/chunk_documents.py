import re

from rag.schemas import ChunkRecord, DocumentRecord


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")


def _clean_text(text: str) -> str:
    """Normalize whitespace without destroying Markdown structure completely."""
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned = "\n".join(lines).strip()
    return cleaned


def _split_markdown_by_headings(text: str) -> list[tuple[str | None, str]]:
    """
    Split Markdown text into sections using headings.

    Returns a list of:
    (heading, section_text)
    """

    sections: list[tuple[str | None, list[str]]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        heading_match = HEADING_PATTERN.match(line.strip())

        if heading_match:
            if current_lines:
                sections.append((current_heading, current_lines))

            current_heading = heading_match.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, current_lines))

    result: list[tuple[str | None, str]] = []

    for heading, lines in sections:
        section_text = _clean_text("\n".join(lines))

        if section_text:
            result.append((heading, section_text))

    return result


def _split_long_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split long text into smaller chunks.

    This is a simple character-based fallback used when a Markdown section is too long.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(chunk_text)

        if end >= len(text):
            break

        start = end - chunk_overlap

    return chunks


def chunk_document(
    document: DocumentRecord,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> list[ChunkRecord]:
    """
    Split one DocumentRecord into ChunkRecord objects.

    Strategy:
    1. Split Markdown by headings.
    2. If a section is too long, split it into smaller overlapping chunks.
    3. Preserve document metadata on every chunk.
    """

    sections = _split_markdown_by_headings(document.text)
    chunks: list[ChunkRecord] = []

    for heading, section_text in sections:
        section_chunks = _split_long_text(
            text=section_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for section_chunk in section_chunks:
            chunk_index = len(chunks)
            chunk_id = f"{document.doc_id}::chunk_{chunk_index:03d}"

            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    title=document.title,
                    heading=heading,
                    text=section_chunk,
                    source_path=document.source_path,
                    chunk_index=chunk_index,
                    category=document.category,
                    metadata={
                        **document.metadata,
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                        "heading": heading,
                    },
                )
            )

    return chunks


def chunk_documents(
    documents: list[DocumentRecord],
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> list[ChunkRecord]:
    """Split many documents into chunks."""

    all_chunks: list[ChunkRecord] = []

    for document in documents:
        all_chunks.extend(
            chunk_document(
                document=document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    return all_chunks