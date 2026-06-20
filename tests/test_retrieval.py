from pathlib import Path

import pytest

from ingestion.build_index import build_local_index
from retrieval.retriever import retrieve
from retrieval.vector_search import cosine_similarity, search_vector_index


SAMPLE_DOCS_DIR = Path("data/sample_policies")


def test_cosine_similarity_returns_one_for_same_vector() -> None:
    score = cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])

    assert score == pytest.approx(1.0)


def test_cosine_similarity_returns_zero_for_orthogonal_vectors() -> None:
    score = cosine_similarity([1.0, 0.0], [0.0, 1.0])

    assert score == pytest.approx(0.0)


def test_vector_search_returns_ranked_chunks(tmp_path: Path) -> None:
    build_local_index(
        docs_dir=SAMPLE_DOCS_DIR,
        output_dir=tmp_path,
        provider_name="hash",
    )

    result = search_vector_index(
        query="MFA privileged access",
        index_dir=tmp_path,
        top_k=3,
    )

    assert result.query == "MFA privileged access"
    assert result.retrieval_mode == "vector"
    assert len(result.chunks) == 3

    scores = [chunk.score for chunk in result.chunks]

    assert scores == sorted(scores, reverse=True)
    assert result.chunks[0].rank == 1
    assert result.chunks[1].rank == 2
    assert result.chunks[2].rank == 3


def test_vector_search_finds_expected_mfa_source(tmp_path: Path) -> None:
    build_local_index(
        docs_dir=SAMPLE_DOCS_DIR,
        output_dir=tmp_path,
        provider_name="hash",
    )

    result = search_vector_index(
        query="MFA privileged access number matching",
        index_dir=tmp_path,
        top_k=3,
    )

    retrieved_sources = {
        Path(chunk.source_path).name
        for chunk in result.chunks
    }

    assert "mfa_policy.md" in retrieved_sources


def test_unified_retriever_uses_vector_mode(tmp_path: Path) -> None:
    build_local_index(
        docs_dir=SAMPLE_DOCS_DIR,
        output_dir=tmp_path,
        provider_name="hash",
    )

    result = retrieve(
        query="phishing email report security",
        mode="vector",
        top_k=3,
        index_dir=tmp_path,
    )

    assert result.retrieval_mode == "vector"
    assert len(result.chunks) == 3


def test_unknown_retrieval_mode_raises_error(tmp_path: Path) -> None:
    build_local_index(
        docs_dir=SAMPLE_DOCS_DIR,
        output_dir=tmp_path,
        provider_name="hash",
    )

    with pytest.raises(NotImplementedError):
        retrieve(
            query="test",
            mode="hybrid",
            index_dir=tmp_path,
        )