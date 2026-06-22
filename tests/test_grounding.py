from rag.grounding_checker import check_grounding
from rag.schemas import ChunkResult


def _chunk(score: float = 0.8) -> ChunkResult:
    return ChunkResult(
        chunk_id="mfa_policy.md::chunk_001",
        text="All privileged accounts must use MFA.",
        source_path="data/sample_policies/mfa_policy.md",
        title="Multi-Factor Authentication Policy",
        heading="Privileged access",
        score=score,
        retrieval_mode="vector",
        rank=1,
        metadata={"chunk_id": "mfa_policy.md::chunk_001"},
    )


def test_grounding_passes_for_answer_with_retrieved_citation() -> None:
    chunk = _chunk()

    result = check_grounding(
        answer="Privileged accounts must use MFA.",
        cited_chunk_ids=["mfa_policy.md::chunk_001"],
        retrieved_chunks=[chunk],
    )

    assert result["status"] == "PASS"


def test_grounding_fails_when_answer_has_no_citations() -> None:
    chunk = _chunk()

    result = check_grounding(
        answer="Privileged accounts must use MFA.",
        cited_chunk_ids=[],
        retrieved_chunks=[chunk],
    )

    assert result["status"] == "FAIL"


def test_grounding_passes_for_safe_refusal_without_citations() -> None:
    result = check_grounding(
        answer="I do not have enough information.",
        cited_chunk_ids=[],
        retrieved_chunks=[],
        refusal=True,
    )

    assert result["status"] == "PASS"


def test_grounding_warns_for_low_top_score() -> None:
    chunk = _chunk(score=0.01)

    result = check_grounding(
        answer="Privileged accounts must use MFA.",
        cited_chunk_ids=["mfa_policy.md::chunk_001"],
        retrieved_chunks=[chunk],
        min_top_score=0.05,
    )

    assert result["status"] == "WARN"