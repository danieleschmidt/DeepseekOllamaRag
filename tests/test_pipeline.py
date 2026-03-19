"""Tests for RAGPipeline (mock backend only)."""

import pytest
from rag.ollama_client import OllamaClient
from rag.pipeline import RAGPipeline


DOCS = [
    "Python is a popular high-level programming language.",
    "RAG stands for Retrieval-Augmented Generation, an AI technique.",
    "Ollama lets you run large language models locally.",
    "DeepSeek is a family of open-weight language models.",
    "Vector databases are used to store and search embeddings.",
]


def _pipeline(**kwargs) -> RAGPipeline:
    client = OllamaClient(force_mock=True)
    return RAGPipeline(ollama_client=client, chunk_size=300, overlap=50, top_k=2, **kwargs)


class TestRAGPipelineIndex:
    def test_index_returns_chunk_count(self):
        p = _pipeline()
        n = p.index(DOCS)
        assert n == len(DOCS)  # each short doc is one chunk

    def test_index_with_labels(self):
        p = _pipeline()
        labels = [f"doc_{i}" for i in range(len(DOCS))]
        n = p.index(DOCS, source_labels=labels)
        assert n > 0

    def test_chunk_count_after_index(self):
        p = _pipeline()
        p.index(DOCS)
        assert p.chunk_count == len(DOCS)

    def test_empty_index(self):
        p = _pipeline()
        n = p.index([])
        assert n == 0


class TestRAGPipelineQuery:
    def setup_method(self):
        self.pipeline = _pipeline()
        self.pipeline.index(DOCS)

    def test_query_returns_dict(self):
        result = self.pipeline.query("What is Python?")
        assert isinstance(result, dict)

    def test_query_keys_present(self):
        result = self.pipeline.query("What is Python?")
        assert "question" in result
        assert "context" in result
        assert "answer" in result
        assert "is_mock" in result

    def test_query_question_preserved(self):
        q = "What is RAG?"
        result = self.pipeline.query(q)
        assert result["question"] == q

    def test_query_is_mock_true(self):
        result = self.pipeline.query("hello")
        assert result["is_mock"] is True

    def test_query_context_is_list(self):
        result = self.pipeline.query("Ollama models")
        assert isinstance(result["context"], list)
        assert len(result["context"]) <= 2  # top_k=2

    def test_query_answer_is_string(self):
        result = self.pipeline.query("DeepSeek models")
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_query_empty_store(self):
        p = _pipeline()  # not indexed
        result = p.query("anything")
        assert result["answer"] is not None

    def test_multiple_queries(self):
        questions = [
            "What is Python?",
            "How does RAG work?",
            "What is Ollama?",
        ]
        for q in questions:
            result = self.pipeline.query(q)
            assert "answer" in result
