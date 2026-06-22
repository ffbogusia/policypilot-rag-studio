from rag.citation_builder import build_citations
from rag.schemas import ChunkResult


def _chunk(chunk_id: str, rank: int = 1) -> ChunkResult:
    return ChunkResult(
        chunk_id=chunk_id,
        text="All privileged accounts must use MFA.",
        source_path="data/sample_policies/mfa_policy.md",
        title="Multi-Factor Authentication Policy",
        heading="Privileged access",
        score=0.8,
        retrieval_mode="vector",
        rank=rank,
        metadata={"chunk_id": chunk_id},
    )


def test_build_citations_returns_chunk_ids_and_sources() -> None:
    cited_chunk_ids, sources = build_citations([_chunk("mfa_policy.md::chunk_001")])

    assert cited_chunk_ids == ["mfa_policy.md::chunk_001"]
    assert len(sources) == 1
    assert sources[0]["source_path"] == "data/sample_policies/mfa_policy.md"
    assert sources[0]["heading"] == "Privileged access"


def test_build_citations_respects_max_sources() -> None:
    chunks = [
        _chunk("chunk_001", rank=1),
        _chunk("chunk_002", rank=2),
        _chunk("chunk_003", rank=3),
    ]

    cited_chunk_ids, sources = build_citations(chunks, max_sources=2)

    assert cited_chunk_ids == ["chunk_001", "chunk_002"]
    assert len(sources) == 2


def test_build_citations_skips_duplicate_chunk_ids() -> None:
    chunks = [
        _chunk("chunk_001", rank=1),
        _chunk("chunk_001", rank=2),
    ]

    cited_chunk_ids, sources = build_citations(chunks)

    assert cited_chunk_ids == ["chunk_001"]
    assert len(sources) == 1