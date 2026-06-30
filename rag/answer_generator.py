import argparse
import re
from pathlib import Path

from rag.citation_builder import build_citations
from rag.grounding_checker import check_grounding
from rag.prompt_templates import build_rag_prompt
from rag.schemas import AnswerResult, ChunkResult, RetrievalResult
from retrieval.retriever import retrieve


REFUSAL_MESSAGE = (
    "I do not have enough information in the provided policy documents to answer this safely."
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "have",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "may",
    "of",
    "on",
    "or",
    "should",
    "the",
    "to",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
}

SENSITIVE_OUT_OF_SCOPE_TERMS = {
    "ceo",
    "favorite",
    "restaurant",
    "salary",
    "fired",
    "secret",
}


def _tokens(text: str) -> set[str]:
    """Return meaningful lowercase tokens."""

    raw_tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())

    return {
        token
        for token in raw_tokens
        if len(token) > 2 and token not in STOPWORDS
    }


def _split_sentences(text: str) -> list[str]:
    """Split text into simple sentence-like units."""

    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = [part.strip() for part in parts if part.strip()]

    return sentences


def _has_enough_evidence(
    question: str,
    chunks: list[ChunkResult],
    min_top_score: float = 0.05,
) -> bool:
    """Decide whether retrieved chunks contain enough evidence for an MVP answer."""

    if not chunks:
        return False

    question_tokens = _tokens(question)

    if question_tokens & SENSITIVE_OUT_OF_SCOPE_TERMS:
        return False

    top_score = max(chunk.score for chunk in chunks)

    if top_score < min_top_score:
        return False

    combined_source_tokens: set[str] = set()

    for chunk in chunks[:3]:
        combined_source_tokens.update(_tokens(chunk.text))

    overlap = question_tokens & combined_source_tokens

    return len(overlap) > 0


def _select_relevant_sentences(
    question: str,
    chunks: list[ChunkResult],
    max_sentences: int = 3,
) -> list[str]:
    """Select simple evidence sentences from retrieved chunks."""

    question_tokens = _tokens(question)
    selected_sentences: list[str] = []

    for chunk in chunks:
        for sentence in _split_sentences(chunk.text):
            sentence_tokens = _tokens(sentence)

            if question_tokens & sentence_tokens:
                selected_sentences.append(sentence)

            if len(selected_sentences) >= max_sentences:
                return selected_sentences

    for chunk in chunks[:max_sentences]:
        selected_sentences.append(chunk.preview(220))

    return selected_sentences[:max_sentences]


def _build_fallback_answer(
    question: str,
    chunks: list[ChunkResult],
) -> str:
    """
    Build a deterministic grounded answer from retrieved evidence.

    This is a local fallback, not a fluent LLM answer.
    """

    sentences = _select_relevant_sentences(question, chunks)

    bullet_points = "\n".join(f"- {sentence}" for sentence in sentences)

    return (
        "Based on the retrieved policy sources, the relevant evidence is:\n"
        f"{bullet_points}"
    )


def _chunks_debug_payload(chunks: list[ChunkResult]) -> list[dict[str, object]]:
    """Create a UI-friendly debug payload for retrieved chunks."""

    return [
        {
            "rank": chunk.rank,
            "score": round(chunk.score, 4),
            "chunk_id": chunk.chunk_id,
            "title": chunk.title,
            "heading": chunk.heading,
            "source_path": chunk.source_path,
            "preview": chunk.preview(220),
        }
        for chunk in chunks
    ]


def generate_answer_from_retrieval(
    question: str,
    retrieval_result: RetrievalResult,
    max_sources: int = 3,
) -> AnswerResult:
    """Generate a grounded answer from retrieved chunks."""

    chunks = retrieval_result.chunks
    prompt = build_rag_prompt(question=question, chunks=chunks)

    if not _has_enough_evidence(question, chunks):
        grounding = check_grounding(
            answer=REFUSAL_MESSAGE,
            cited_chunk_ids=[],
            retrieved_chunks=chunks,
            refusal=True,
        )

        return AnswerResult(
            question=question,
            answer=REFUSAL_MESSAGE,
            cited_chunk_ids=[],
            sources=[],
            grounding_status=str(grounding["status"]),
            refusal=True,
            model_name="deterministic-fallback",
            debug={
                "grounding_reason": grounding["reason"],
                "prompt_preview": prompt[:500],
                "retrieved_chunk_count": len(chunks),
                "retrieved_chunks": _chunks_debug_payload(chunks),
            },
        )

    cited_chunk_ids, sources = build_citations(
        chunks=chunks,
        max_sources=max_sources,
    )

    answer = _build_fallback_answer(
        question=question,
        chunks=chunks[:max_sources],
    )

    grounding = check_grounding(
        answer=answer,
        cited_chunk_ids=cited_chunk_ids,
        retrieved_chunks=chunks,
        refusal=False,
    )

    return AnswerResult(
        question=question,
        answer=answer,
        cited_chunk_ids=cited_chunk_ids,
        sources=sources,
        grounding_status=str(grounding["status"]),
        refusal=False,
        model_name="deterministic-fallback",
        debug={
            "grounding_reason": grounding["reason"],
            "prompt_preview": prompt[:500],
            "retrieved_chunk_count": len(chunks),
            "retrieved_chunks": _chunks_debug_payload(chunks),
        },
    )


def answer_question(
    question: str,
    index_dir: str | Path = ".cache/vector_store",
    top_k: int = 5,
) -> AnswerResult:
    """Retrieve chunks and generate a grounded answer."""

    retrieval_result = retrieve(
        query=question,
        mode="vector",
        top_k=top_k,
        index_dir=index_dir,
    )

    return generate_answer_from_retrieval(
        question=question,
        retrieval_result=retrieval_result,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer a policy question with citations.")
    parser.add_argument(
        "question",
        nargs="?",
        default="Can contractors access production data?",
        help="Question to answer from policy documents.",
    )
    parser.add_argument(
        "--index",
        default=".cache/vector_store",
        help="Path to the local vector index folder.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks.",
    )

    args = parser.parse_args()

    result = answer_question(
        question=args.question,
        index_dir=args.index,
        top_k=args.top_k,
    )

    print(f"Question: {result.question}")
    print(f"Model: {result.model_name}")
    print(f"Refusal: {result.refusal}")
    print(f"Grounding status: {result.grounding_status}")
    print(f"Grounding reason: {result.debug['grounding_reason']}")
    print()
    print("Answer:")
    print(result.answer)

    if result.sources:
        print()
        print("Sources:")

        for source in result.sources:
            print(
                f"- {source['chunk_id']} "
                f"({source['source_path']}, score={source['score']})"
            )


if __name__ == "__main__":
    main()