import argparse
from pathlib import Path
from typing import Any

import yaml

from rag.schemas import DocumentRecord


def _optional_str(value: Any) -> str | None:
    """Convert metadata values to strings while preserving missing values as None."""
    if value is None:
        return None

    return str(value)


def _parse_frontmatter(raw_text: str) -> tuple[dict[str, Any], str]:
    """
    Parse a simple Markdown frontmatter block.

    Expected format:

    ---
    title: Example Policy
    category: security
    ---

    # Document body
    """

    lines = raw_text.splitlines()

    if not lines or lines[0].strip() != "---":
        return {}, raw_text.strip()

    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            metadata_text = "\n".join(lines[1:index])
            body_text = "\n".join(lines[index + 1 :]).strip()

            metadata = yaml.safe_load(metadata_text) or {}

            if not isinstance(metadata, dict):
                raise ValueError("Markdown frontmatter must be a key-value mapping.")

            return metadata, body_text

    return {}, raw_text.strip()


def load_markdown_documents(docs_dir: str | Path) -> list[DocumentRecord]:
    """
    Load Markdown policy documents from a folder.

    This function only loads documents and metadata.
    It does not chunk, embed or index documents.
    """

    docs_path = Path(docs_dir)

    if not docs_path.exists():
        raise FileNotFoundError(f"Documents folder does not exist: {docs_path}")

    if not docs_path.is_dir():
        raise NotADirectoryError(f"Expected a folder, got: {docs_path}")

    markdown_files = sorted(docs_path.glob("*.md"))
    documents: list[DocumentRecord] = []

    for file_path in markdown_files:
        raw_text = file_path.read_text(encoding="utf-8")
        metadata, body_text = _parse_frontmatter(raw_text)

        doc_id = str(metadata.get("doc_id") or file_path.name)
        title = str(metadata.get("title") or file_path.stem.replace("_", " ").title())
        category = str(metadata.get("category") or "uncategorized")
        source_path = file_path.as_posix()

        document = DocumentRecord(
            doc_id=doc_id,
            title=title,
            source_path=source_path,
            category=category,
            owner=_optional_str(metadata.get("owner")),
            version=_optional_str(metadata.get("version")),
            text=body_text,
            metadata={
                **metadata,
                "doc_id": doc_id,
                "source_path": source_path,
            },
        )

        documents.append(document)

    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Load sample Markdown policy documents.")
    parser.add_argument(
        "--docs",
        default="data/sample_policies",
        help="Path to the folder with Markdown policy documents.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of loaded documents to print.",
    )

    args = parser.parse_args()
    documents = load_markdown_documents(args.docs)

    print(f"Loaded documents: {len(documents)}")

    for document in documents[: args.limit]:
        print()
        print(f"- doc_id: {document.doc_id}")
        print(f"  title: {document.title}")
        print(f"  category: {document.category}")
        print(f"  owner: {document.owner}")
        print(f"  version: {document.version}")
        print(f"  source_path: {document.source_path}")
        print(f"  chars: {len(document.text)}")
        print(f"  preview: {document.preview(100)}")


if __name__ == "__main__":
    main()