"""
Embedding backend — OpenAI-compatible API.

Talks to the /v1/embeddings HTTP endpoint using httpx:
https://platform.openai.com/docs/api-reference/embeddings

Works with OpenAI itself and with any OpenAI-compatible server (Ollama's /v1,
vLLM, LM Studio, LiteLLM, …).
"""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class OpenAIEmbedder:
    _DEFAULT_MODEL = "text-embedding-3-small"
    _DEFAULT_BASE_URL = "https://api.openai.com/v1"
    _DEFAULT_CONTEXT_LENGTH = 8191

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 120.0,
        context_length: int = _DEFAULT_CONTEXT_LENGTH,
        api_key: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._dimensions = dimensions
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._client = httpx.Client(headers=headers, timeout=timeout)
        self._context_length = context_length

        logger.info(
            "Connecting to embeddings API at %s (model=%s, context_length=%d, auth=%s)…",
            self._base_url,
            model_name,
            context_length,
            "enabled" if api_key else "disabled",
        )
        probe = self._embed_batch(["dimension probe"])
        self._dim = len(probe[0])
        logger.info("Embedder ready (dim=%d)", self._dim)

    def _truncate(self, text: str) -> str:
        """Truncate text to stay within the model's context window.

        Uses a conservative 1 token ≈ 0.75 words heuristic; most sub-word
        tokenisers average 1.2-1.5 tokens per whitespace-delimited word,
        so this leaves headroom.
        """
        max_words = int(self._context_length * 0.75)
        words = text.split()
        if len(words) <= max_words:
            return text
        logger.warning(
            "Truncating text from %d to %d words (context_length=%d tokens).",
            len(words),
            max_words,
            self._context_length,
        )
        return " ".join(words[:max_words])

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        url = f"{self._base_url}/embeddings"
        payload: dict[str, Any] = {"model": self._model_name, "input": texts, "encoding_format": "float"}
        if self._dimensions:
            payload["dimensions"] = self._dimensions

        resp = self._client.post(url, json=payload)
        if resp.is_error:
            raise RuntimeError(f"Embeddings request to {url} failed ({resp.status_code}): {resp.text[:500]}")
        data = resp.json()

        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list) or len(items) != len(texts):
            shape = list(data) if isinstance(data, dict) else type(data).__name__
            raise RuntimeError(f"Unexpected response from {url}: {shape}")
        # The API may return items out of order; sort by their index.
        items = sorted(items, key=lambda d: d.get("index", 0))
        return [d["embedding"] for d in items]

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = [self._truncate(t) for t in texts[i : i + batch_size]]
            results.extend(self._embed_batch(batch))
        return results
