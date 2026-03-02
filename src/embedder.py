"""
Embedding backend — Ollama.

Talks to Ollama's /api/embed HTTP endpoint using httpx.
Set OLLAMA_URL and EMBEDDING_MODEL in your environment.
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    _DEFAULT_MODEL = "nomic-embed-text"

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

        logger.info("Connecting to Ollama at %s (model=%s)…", self._base_url, model_name)
        probe = self._embed_batch(["dimension probe"])
        self._dim = len(probe[0])
        logger.info("Ollama embedder ready (dim=%d)", self._dim)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model_name, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()

        if "embeddings" in data:
            return data["embeddings"]

        raise RuntimeError(
            f"Unexpected response from Ollama /api/embed: {list(data.keys())}. "
            "Make sure you are running Ollama >= 0.1.44."
        )

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results.extend(self._embed_batch(batch))
        return results


def build_embedder() -> OllamaEmbedder:
    model = os.environ.get("EMBEDDING_MODEL", OllamaEmbedder._DEFAULT_MODEL)
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    return OllamaEmbedder(model, base_url=url)
