"""
Embedding backends.

Supports three backends selected by the EMBEDDING_BACKEND env var:

  local  – fastembed (ONNX Runtime, no PyTorch required).  Fast to install,
            small image.  Set EMBEDDING_MODEL to any model name listed by
            fastembed.list_supported_models().  Default: BAAI/bge-small-en-v1.5.

  ollama – Ollama HTTP API.
            Set OLLAMA_URL to your Ollama instance (default: http://localhost:11434).
            Set EMBEDDING_MODEL to any model Ollama has pulled that supports
            embeddings, e.g. "nomic-embed-text", "all-minilm", "mxbai-embed-large".
            No extra Python packages required (uses httpx).

  openai – OpenAI Embeddings API.
            Requires OPENAI_API_KEY.  Set EMBEDDING_MODEL to e.g.
            "text-embedding-3-small" (default) or "text-embedding-3-large".
"""

import logging
import os
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]: ...


# ---------------------------------------------------------------------------
# Local (fastembed / ONNX) backend
# ---------------------------------------------------------------------------

class LocalEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        try:
            from fastembed import TextEmbedding  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "fastembed is required for the local backend: pip install fastembed"
            ) from exc

        cache_dir = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/fastembed")
        logger.info("Loading embedding model: %s  (cache: %s)", model_name, cache_dir)

        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

        probe = next(iter(self._model.embed(["dimension probe"])))
        self._dim: int = len(probe)
        logger.info("Model ready (dim=%d)", self._dim)

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        embeddings = list(self._model.embed(texts, batch_size=batch_size))
        return [e.tolist() for e in embeddings]


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

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
        """Call Ollama's /api/embed endpoint (batch-capable since Ollama 0.1.44)."""
        resp = self._client.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model_name, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()

        if "embeddings" in data:
            return data["embeddings"]

        # Older Ollama versions only support /api/embeddings (single string).
        # Fall back to one-at-a-time if /api/embed returned something unexpected.
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


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

class OpenAIEmbedder:
    _DEFAULT_MODEL = "text-embedding-3-small"
    _DIM_MAP = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "openai package is required for the openai backend: pip install openai"
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set")

        self._model_name = model_name
        self._client = OpenAI(api_key=api_key)
        self._dim = self._DIM_MAP.get(model_name, 1536)
        logger.info("OpenAI embedder ready (model=%s, dim=%d)", model_name, self._dim)

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.embeddings.create(input=batch, model=self._model_name)
            results.extend(item.embedding for item in response.data)
        return results


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_embedder() -> Embedder:
    backend = os.environ.get("EMBEDDING_BACKEND", "local").lower()
    model = os.environ.get("EMBEDDING_MODEL", "")

    if backend == "ollama":
        url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        return OllamaEmbedder(model or OllamaEmbedder._DEFAULT_MODEL, base_url=url)
    elif backend == "openai":
        return OpenAIEmbedder(model or OpenAIEmbedder._DEFAULT_MODEL)
    elif backend == "local":
        return LocalEmbedder(model or "BAAI/bge-small-en-v1.5")
    else:
        raise ValueError(
            f"Unknown EMBEDDING_BACKEND: {backend!r}  (choose 'local', 'ollama', or 'openai')"
        )
