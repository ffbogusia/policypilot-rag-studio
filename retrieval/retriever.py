from pathlib import Path

from retrieval.vector_search import search_vector_index
from rag.schemas import RetrievalResult


def retrieve(
    query: str,
    mode: str = "vector",
    top_k: int = 5,
    index_dir: str | Path = ".cache/vector_store",
) -> RetrievalResult:
    """
    Unified retrieval entrypoint.

    For now only vector mode is implemented.
    Later this can route to keyword or hybrid retrieval.
    """

    if mode == "vector":
        return search_vector_index(
            query=query,
            index_dir=index_dir,
            top_k=top_k,
        )

    raise NotImplementedError(f"Retrieval mode is not implemented yet: {mode}")