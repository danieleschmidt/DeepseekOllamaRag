"""
DocumentStore: stores text chunks with TF-IDF / cosine-similarity retrieval.

Uses only stdlib + numpy — no external ML dependencies required.
"""

import math
import re
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lowercase and split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _tf(tokens: List[str]) -> Dict[str, float]:
    """Term frequency (raw count / total tokens)."""
    if not tokens:
        return {}
    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = len(tokens)
    return {t: c / total for t, c in counts.items()}


# ---------------------------------------------------------------------------
# DocumentStore
# ---------------------------------------------------------------------------

class DocumentStore:
    """
    In-memory document store backed by TF-IDF vectors (numpy).

    Usage
    -----
    store = DocumentStore()
    store.add_documents(["chunk 1 text", "chunk 2 text", ...])
    results = store.search("my query", top_k=3)
    # results → list of (chunk_text, score)
    """

    def __init__(self) -> None:
        self._chunks: List[str] = []
        self._vocabulary: List[str] = []        # sorted vocab list
        self._tfidf_matrix: np.ndarray = np.empty((0,))  # (n_chunks, vocab)
        self._built = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(self, chunks: List[str]) -> None:
        """Add a list of text chunks and (re)build the TF-IDF index."""
        if not chunks:
            return
        self._chunks.extend(chunks)
        self._build_index()

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Return top-k (chunk, score) pairs most relevant to *query*.
        Score is cosine similarity in TF-IDF space.
        """
        if not self._built or len(self._chunks) == 0:
            return []

        q_vec = self._vectorize_query(query)
        if np.linalg.norm(q_vec) == 0:
            # Query has no known terms — fall back to empty result
            return []

        scores = self._cosine_similarity(q_vec, self._tfidf_matrix)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self._chunks[i], float(scores[i])) for i in top_indices]

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Build vocabulary and TF-IDF matrix from all stored chunks."""
        # 1. Tokenize every chunk
        tokenized = [_tokenize(c) for c in self._chunks]

        # 2. Build vocabulary (union of all tokens)
        vocab_set: set = set()
        for tokens in tokenized:
            vocab_set.update(tokens)
        self._vocabulary = sorted(vocab_set)
        vocab_index = {t: i for i, t in enumerate(self._vocabulary)}
        V = len(self._vocabulary)
        N = len(self._chunks)

        if V == 0 or N == 0:
            self._tfidf_matrix = np.zeros((N, 0))
            self._built = True
            return

        # 3. Compute document frequency for each term
        df = np.zeros(V, dtype=np.float64)
        for tokens in tokenized:
            for t in set(tokens):
                if t in vocab_index:
                    df[vocab_index[t]] += 1

        # IDF with smoothing
        idf = np.log((N + 1) / (df + 1)) + 1.0

        # 4. Build TF-IDF matrix
        matrix = np.zeros((N, V), dtype=np.float64)
        for i, tokens in enumerate(tokenized):
            tf_map = _tf(tokens)
            for term, tf_val in tf_map.items():
                if term in vocab_index:
                    j = vocab_index[term]
                    matrix[i, j] = tf_val * idf[j]

        # L2-normalise rows
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # avoid div-by-zero
        self._tfidf_matrix = matrix / norms
        self._built = True

    def _vectorize_query(self, query: str) -> np.ndarray:
        """Project a query string into the current TF-IDF space."""
        V = len(self._vocabulary)
        if V == 0:
            return np.zeros(1)

        vocab_index = {t: i for i, t in enumerate(self._vocabulary)}
        tokens = _tokenize(query)
        tf_map = _tf(tokens)

        N = len(self._chunks)
        # Recompute df quickly (already stored implicitly via matrix)
        # Instead, just use raw TF against learned IDF
        # We need stored IDF — store it at build time
        # NOTE: we rebuild idf inline here for simplicity (cheap)
        tokenized = [_tokenize(c) for c in self._chunks]
        df = np.zeros(V, dtype=np.float64)
        for tok_list in tokenized:
            for t in set(tok_list):
                if t in vocab_index:
                    df[vocab_index[t]] += 1
        idf = np.log((N + 1) / (df + 1)) + 1.0

        vec = np.zeros(V, dtype=np.float64)
        for term, tf_val in tf_map.items():
            if term in vocab_index:
                j = vocab_index[term]
                vec[j] = tf_val * idf[j]

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    @staticmethod
    def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Dot product of normalised query vec and normalised matrix rows."""
        return matrix @ query_vec
