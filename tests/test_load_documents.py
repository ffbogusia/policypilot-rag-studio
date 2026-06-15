from pathlib import Path

from ingestion.load_documents import load_markdown_documents


SAMPLE_DOCS_DIR = Path("data/sample_policies")


def test_loads_sample_policy_documents() -> None:
    documents = load_markdown_documents(SAMPLE_DOCS_DIR)

    assert len(documents) >= 3


def test_loaded_documents_have_required_fields() -> None:
    documents = load_markdown_documents(SAMPLE_DOCS_DIR)

    for document in documents:
        assert document.doc_id
        assert document.title
        assert document.source_path
        assert document.category
        assert document.text
        assert document.metadata


def test_frontmatter_metadata_is_preserved() -> None:
    documents = load_markdown_documents(SAMPLE_DOCS_DIR)

    access_policy = next(
        document for document in documents if document.doc_id == "access_policy.md"
    )

    assert access_policy.title == "Production Access Policy"
    assert access_policy.category == "security"
    assert access_policy.owner == "Security Operations"
    assert access_policy.version == "1.0"


def test_document_text_does_not_include_frontmatter() -> None:
    documents = load_markdown_documents(SAMPLE_DOCS_DIR)

    for document in documents:
        assert not document.text.lstrip().startswith("---")
        assert document.text.lstrip().startswith("#")