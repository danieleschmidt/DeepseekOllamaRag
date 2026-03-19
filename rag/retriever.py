"""Retriever: query DocumentStore and return top-k chunks."""

from typing import List, Tuple

from .document_store import DocumentStore


class Retriever:
    """
    Wraps a DocumentStore and provides a clean retrieval interface.

    Parameters
    ----------
    store : DocumentStore
        Pre-built document store.
    top_k : int
        Number of chunks to return per query (default 3).
    """

    def __init__(self, store: DocumentStore, top_k: int = 3) -> None:
        self.store = store
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Tuple[str, float]]:
        """
        Return [(chunk_text, score), ...] for the given query.

        Results are sorted by relevance (highest score first).
        """
        return self.store.search(query, top_k=self.top_k)

    def retrieve_text(self, query: str) -> List[str]:
        """Return only the chunk texts (no scores)."""
        return [chunk for chunk, _ in self.retrieve(query)]
