"""Tests for DocumentStore."""

import pytest
from rag.document_store import DocumentStore, _tokenize, _tf


class TestTokenize:
    def test_lowercase(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self):
        assert _tokenize("foo.bar,baz!") == ["foo", "bar", "baz"]

    def test_numbers(self):
        tokens = _tokenize("version 3.12")
        assert "version" in tokens
        assert "3" in tokens or "12" in tokens  # split on dot

    def test_empty(self):
        assert _tokenize("") == []


class TestTF:
    def test_basic(self):
        tf = _tf(["a", "b", "a"])
        assert tf["a"] == pytest.approx(2 / 3)
        assert tf["b"] == pytest.approx(1 / 3)

    def test_empty(self):
        assert _tf([]) == {}


class TestDocumentStore:
    def _make_store(self):
        store = DocumentStore()
        store.add_documents([
            "Python is a great programming language",
            "Machine learning requires large datasets",
            "RAG combines retrieval and generation",
            "Ollama runs LLMs locally on your machine",
            "Neural networks learn from training data",
        ])
        return store

    def test_chunk_count(self):
        store = self._make_store()
        assert store.chunk_count == 5

    def test_search_returns_top_k(self):
        store = self._make_store()
        results = store.search("Python programming", top_k=2)
        assert len(results) == 2

    def test_search_returns_tuples(self):
        store = self._make_store()
        results = store.search("machine learning", top_k=1)
        assert len(results) == 1
        chunk, score = results[0]
        assert isinstance(chunk, str)
        assert isinstance(score, float)

    def test_relevance_ordering(self):
        store = self._make_store()
        results = store.search("Ollama local LLM", top_k=3)
        # The Ollama chunk should score highest
        top_chunk = results[0][0]
        assert "Ollama" in top_chunk

    def test_empty_store_returns_empty(self):
        store = DocumentStore()
        assert store.search("anything") == []

    def test_unknown_query_returns_results(self):
        store = self._make_store()
        # Query with totally unknown words returns results (possibly zero-score)
        results = store.search("zzzzqqqq", top_k=2)
        # Should return empty (no matching terms)
        assert isinstance(results, list)

    def test_add_documents_incremental(self):
        store = DocumentStore()
        store.add_documents(["first document about Python"])
        assert store.chunk_count == 1
        store.add_documents(["second document about Java"])
        assert store.chunk_count == 2

    def test_scores_are_normalized(self):
        store = self._make_store()
        results = store.search("Python", top_k=5)
        for _, score in results:
            assert -1.0 <= score <= 1.01  # cosine similarity range
