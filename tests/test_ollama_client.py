"""Tests for OllamaClient (mock backend only)."""

from rag.ollama_client import OllamaClient


class TestOllamaClientMock:
    """All tests use force_mock=True — no Ollama required."""

    def _client(self, model: str = "test-model") -> OllamaClient:
        return OllamaClient(model=model, force_mock=True)

    def test_is_mock(self):
        client = self._client()
        assert client.is_mock is True

    def test_generate_returns_string(self):
        client = self._client()
        response = client.generate("What is Python?")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_generate_contains_mock_marker(self):
        client = self._client()
        response = client.generate("hello")
        assert "MOCK" in response

    def test_embed_returns_list_of_floats(self):
        client = self._client()
        vec = client.embed("test text")
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert all(isinstance(v, float) for v in vec)

    def test_embed_deterministic(self):
        client = self._client()
        v1 = client.embed("same text")
        v2 = client.embed("same text")
        assert v1 == v2

    def test_embed_different_texts(self):
        client = self._client()
        v1 = client.embed("Python programming")
        v2 = client.embed("Machine learning")
        assert v1 != v2

    def test_list_models(self):
        client = self._client()
        models = client.list_models()
        assert isinstance(models, list)

    def test_generate_with_system_prompt(self):
        client = self._client()
        response = client.generate("question", system="You are a helpful assistant.")
        assert isinstance(response, str)

    def test_generate_model_override(self):
        client = self._client(model="default-model")
        response = client.generate("prompt", model="override-model")
        assert isinstance(response, str)
