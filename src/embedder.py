"""
Embedding backends.

Supports two backends selected by the EMBEDDING_BACKEND env var:

  local  – sentence-transformers running entirely in-process (default).
            Set EMBEDDING_MODEL to any sentence-transformers model name.

  openai – OpenAI Embeddings API.
            Requires OPENAI_API_KEY.  Set EMBEDDING_MODEL to e.g.
            "text-embedding-3-small" (default) or "text-embedding-3-large".
"""

import logging
import os
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]: ...


# ---------------------------------------------------------------------------
# Local (sentence-transformers) backend
# ---------------------------------------------------------------------------

class LocalEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info("Loading local embedding model: %s", model_name)
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._model = SentenceTransformer(model_name)
        self._dim: int = self._model.get_sentence_embedding_dimension()
        logger.info("Model loaded  (dim=%d)", self._dim)

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        vectors: np.ndarray = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,
        )
        return vectors.tolist()


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

    if backend == "openai":
        return OpenAIEmbedder(model or OpenAIEmbedder._DEFAULT_MODEL)
    elif backend == "local":
        return LocalEmbedder(model or "all-MiniLM-L6-v2")
    else:
        raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend!r}  (choose 'local' or 'openai')")
