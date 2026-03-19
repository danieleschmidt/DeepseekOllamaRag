"""
OllamaClient: thin HTTP client for the Ollama REST API.

Falls back to a mock backend when Ollama is not reachable.
"""

import json
import logging
from typing import Dict, List, Optional, Any

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:  # pragma: no cover
    _HAS_REQUESTS = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock backend (no Ollama required)
# ---------------------------------------------------------------------------

class _MockBackend:
    """Returns canned responses when Ollama is unavailable."""

    def generate(self, model: str, prompt: str, **kwargs) -> str:  # noqa: ARG002
        return (
            "[MOCK RESPONSE] "
            f"Based on the provided context, here is a synthesised answer to your question. "
            f"(Ollama not available — running in mock mode.)"
        )

    def embed(self, model: str, prompt: str) -> List[float]:  # noqa: ARG002
        # Return a deterministic fake embedding vector (32 dims)
        import hashlib
        h = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
        import math
        return [math.sin(h * (i + 1) * 0.001) for i in range(32)]

    def list_models(self) -> List[str]:
        return ["mock-model"]


# ---------------------------------------------------------------------------
# Real Ollama backend
# ---------------------------------------------------------------------------

class _OllamaBackend:
    def __init__(self, base_url: str, timeout: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, model: str, prompt: str, system: str = "", **kwargs) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        payload.update(kwargs)

        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def embed(self, model: str, prompt: str) -> List[float]:
        resp = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": model, "prompt": prompt},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json().get("embedding", [])

    def list_models(self) -> List[str]:
        resp = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]


# ---------------------------------------------------------------------------
# Public client
# ---------------------------------------------------------------------------

class OllamaClient:
    """
    Thin wrapper around the Ollama HTTP API.

    Automatically detects whether Ollama is running and falls back to a
    mock backend if it is not (useful for tests and offline development).

    Parameters
    ----------
    base_url : str
        Ollama server URL (default ``http://localhost:11434``).
    model : str
        Default model name (e.g. ``deepseek-r1``, ``llama3.2``, ``phi3``).
    timeout : int
        HTTP request timeout in seconds (default 120).
    force_mock : bool
        Skip the connectivity check and always use the mock backend.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "deepseek-r1",
        timeout: int = 120,
        force_mock: bool = False,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self._mock = False

        if force_mock or not _HAS_REQUESTS:
            self._backend: Any = _MockBackend()
            self._mock = True
            logger.info("OllamaClient: using mock backend (forced)")
        else:
            self._backend = _OllamaBackend(base_url, timeout)
            if not self._ping():
                logger.warning(
                    "OllamaClient: Ollama not reachable at %s — falling back to mock",
                    base_url,
                )
                self._backend = _MockBackend()
                self._mock = True
            else:
                logger.info("OllamaClient: connected to Ollama at %s", base_url)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_mock(self) -> bool:
        """True when running against the mock backend."""
        return self._mock

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: str = "",
        **kwargs,
    ) -> str:
        """
        Generate a response from the LLM.

        Parameters
        ----------
        prompt : str
            User prompt / question.
        model : str, optional
            Override the default model.
        system : str
            Optional system prompt.
        **kwargs
            Extra parameters forwarded to the Ollama API (e.g. ``temperature``).
        """
        _model = model or self.model
        logger.debug("generate | model=%s | prompt_len=%d", _model, len(prompt))
        return self._backend.generate(_model, prompt, system=system, **kwargs)

    def embed(self, text: str, model: Optional[str] = None) -> List[float]:
        """Return an embedding vector for *text*."""
        _model = model or self.model
        return self._backend.embed(_model, text)

    def list_models(self) -> List[str]:
        """Return the list of locally available models."""
        return self._backend.list_models()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ping(self) -> bool:
        """Return True if Ollama is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return resp.status_code < 400
        except Exception:
            return False
