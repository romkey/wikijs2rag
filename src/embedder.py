"""
Embedding backend — Ollama.

Talks to Ollama's /api/embed HTTP endpoint using httpx.
Set OLLAMA_URL, EMBEDDING_MODEL, and optionally OLLAMA_API_KEY in your environment.
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    _DEFAULT_MODEL = "nomic-embed-text"
    _DEFAULT_CONTEXT_LENGTH = 8192

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        context_length: int = _DEFAULT_CONTEXT_LENGTH,
        api_key: str | None = None,
    ):
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._client = httpx.Client(headers=headers, timeout=timeout)
        self._context_length = context_length

        logger.info(
            "Connecting to Ollama at %s (model=%s, context_length=%d, auth=%s)…",
            self._base_url, model_name, context_length, "enabled" if api_key else "disabled",
        )
        probe = self._embed_batch(["dimension probe"])
        self._dim = len(probe[0])
        logger.info("Ollama embedder ready (dim=%d)", self._dim)

    def _truncate(self, text: str) -> str:
        """Truncate text to stay within the model's context window.

        Uses a conservative 1 token ≈ 0.75 words heuristic; most sub-word
        tokenisers average 1.2–1.5 tokens per whitespace-delimited word,
        so this leaves headroom.
        """
        max_words = int(self._context_length * 0.75)
        words = text.split()
        if len(words) <= max_words:
            return text
        logger.warning(
            "Truncating text from %d to %d words (context_length=%d tokens).",
            len(words), max_words, self._context_length,
        )
        return " ".join(words[:max_words])

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
            batch = [self._truncate(t) for t in texts[i : i + batch_size]]
            results.extend(self._embed_batch(batch))
        return results


def build_embedder() -> OllamaEmbedder:
    model = os.environ.get("EMBEDDING_MODEL", OllamaEmbedder._DEFAULT_MODEL)
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    api_key = os.environ.get("OLLAMA_API_KEY", "").strip() or None
    ctx = int(os.environ.get("EMBEDDING_CONTEXT_LENGTH", str(OllamaEmbedder._DEFAULT_CONTEXT_LENGTH)))
    return OllamaEmbedder(model, base_url=url, context_length=ctx, api_key=api_key)
