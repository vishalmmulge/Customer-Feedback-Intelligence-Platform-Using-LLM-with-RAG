"""Ollama-backed text embeddings.

Calls the Ollama ``/api/embeddings`` endpoint to turn a string into a dense
vector. Ollama must be running locally (or at ``OLLAMA_BASE_URL``) with an
embedding model pulled (e.g. ``ollama pull nomic-embed-text``).

The class is deliberately thin: if Ollama is unavailable it raises
:class:`ProviderError`, which callers (retriever / upload) catch and convert
into a user-friendly response rather than a 500 crash.
"""

from typing import List

import httpx

from ..core.config import settings
from ..core.exceptions import ProviderError
from ..core.logging import logger


class OllamaClient:
    """Minimal HTTP client for the Ollama API."""

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self._client = httpx.Client(timeout=60.0)

    def embed(self, model: str, input_text: str) -> List[float]:
        """Get an embedding vector for a single text."""
        try:
            resp = self._client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": model, "prompt": input_text},
            )
            resp.raise_for_status()
            data = resp.json()
            embedding = data.get("embedding")
            if not embedding:
                raise ProviderError("Ollama returned an empty embedding")
            return [float(x) for x in embedding]
        except ProviderError:
            raise
        except Exception as exc:  # network / timeout / connection refused
            raise ProviderError(f"Ollama embedding request failed: {exc}")

    def chat(self, model: str, messages: List[dict], temperature: float = 0.2) -> str:
        """Send a chat completion to Ollama and return the assistant message."""
        try:
            resp = self._client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            message = data.get("message", {})
            content = message.get("content", "")
            if not content:
                raise ProviderError("Ollama returned an empty completion")
            return content
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Ollama chat request failed: {exc}")

    def close(self) -> None:
        self._client.close()

    def ping(self, timeout: float = 5.0) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            resp = self._client.get(f"{self.base_url}/api/tags", timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False


class EmbeddingsService:
    """Encodes text into vectors using Ollama."""

    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient()
        self.model = settings.OLLAMA_EMBED_MODEL

    def embed_text(self, text: str) -> List[float]:
        """Embed a single string."""
        if not text or not text.strip():
            text = ""
        return self.client.embed(self.model, text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of strings, one request per text."""
        vectors: List[List[float]] = []
        for i, text in enumerate(texts, 1):
            vectors.append(self.embed_text(text))
            if i % 25 == 0:
                logger.info(f"Embedded {i}/{len(texts)} texts")
        logger.info(f"Embedded {len(texts)} text(s) with model {self.model}")
        return vectors
