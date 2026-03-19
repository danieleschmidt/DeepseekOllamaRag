"""Tests for Retriever."""

from rag.document_store import DocumentStore
from rag.retriever import Retriever


def _make_retriever(top_k: int = 3) -> Retriever:
    store = DocumentStore()
    store.add_documents([
        "Python programming language syntax and features",
        "Machine learning and neural network training",
        "Ollama runs large language models on local hardware",
        "RAG retrieval augmented generation pipeline",
        "Docker containers for application deployment",
    ])
    return Retriever(store, top_k=top_k)


class TestRetriever:
    def test_retrieve_returns_list(self):
        r = _make_retriever()
        results = r.retrieve("Python")
        assert isinstance(results, list)

    def test_retrieve_top_k(self):
        r = _make_retriever(top_k=2)
        results = r.retrieve("machine learning")
        assert len(results) == 2

    def test_retrieve_text_strings_only(self):
        r = _make_retriever()
        texts = r.retrieve_text("Ollama LLM local")
        assert all(isinstance(t, str) for t in texts)

    def test_retrieve_text_top_result_relevant(self):
        r = _make_retriever(top_k=3)
        texts = r.retrieve_text("Ollama local hardware")
        assert any("Ollama" in t for t in texts)

    def test_empty_store(self):
        store = DocumentStore()
        r = Retriever(store, top_k=3)
        assert r.retrieve("anything") == []
