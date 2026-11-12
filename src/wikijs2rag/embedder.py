"""
Embedding backend — OpenAI-compatible API.

Talks to the /v1/embeddings HTTP endpoint using httpx:
https://platform.openai.com/docs/api-reference/embeddings

Works with any OpenAI-compatible server: Hugging Face Text Embeddings
Inference (TEI, the default), OpenAI itself, vLLM, LiteLLM, ….

Many open models expect a task prefix on each input, one for search queries
and another for documents (nomic-ai/modernbert-embed-base wants
"search_query: " and "search_document: "). The API has no field for that, so
the prefix is prepended to the text before it is sent.
"""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# (query prefix, document prefix) for models known to need them, matched by a
# case-insensitive substring of the model name. Models not listed get none.
KNOWN_PREFIXES: tuple[tuple[str, str, str], ...] = (
    ("modernbert-embed", "search_query: ", "search_document: "),
    ("nomic-embed-text", "search_query: ", "search_document: "),
    ("mxbai-embed-large", "Represent this sentence for searching relevant passages: ", ""),
    (
        "qwen3-embedding",
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
        "",
    ),
    ("e5-", "query: ", "passage: "),
)


def default_prefixes(model: str) -> tuple[str, str]:
    """The (query, document) prefixes *model* expects, or empty strings."""
    name = model.lower()
    for key, query, document in KNOWN_PREFIXES:
        if key in name:
            return query, document
    return "", ""


class OpenAIEmbedder:
    _DEFAULT_MODEL = "nomic-ai/modernbert-embed-base"
    _DEFAULT_BASE_URL = "http://localhost:8080/v1"
    _DEFAULT_CONTEXT_LENGTH = 8192

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 120.0,
        context_length: int = _DEFAULT_CONTEXT_LENGTH,
        api_key: str | None = None,
        dimensions: int | None = None,
        query_prefix: str = "",
        document_prefix: str = "",
    ) -> None:
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._dimensions = dimensions
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._client = httpx.Client(headers=headers, timeout=timeout)
        self._context_length = context_length

        logger.info(
            "Connecting to embeddings API at %s (model=%s, context_length=%d, auth=%s, "
            "query_prefix=%r, document_prefix=%r)…",
            self._base_url,
            model_name,
            context_length,
            "enabled" if api_key else "disabled",
            query_prefix,
            document_prefix,
        )
        probe = self._embed_batch(["dimension probe"])
        self._dim = len(probe[0])
        logger.info("Embedder ready (dim=%d)", self._dim)

    def _truncate(self, text: str, reserved_words: int = 0) -> str:
        """Truncate text to stay within the model's context window.

        Uses a conservative 1 token ≈ 0.75 words heuristic; most sub-word
        tokenisers average 1.2-1.5 tokens per whitespace-delimited word,
        so this leaves headroom. *reserved_words* leaves room for a prefix.
        """
        max_words = max(1, int(self._context_length * 0.75) - reserved_words)
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

    def encode(self, texts: list[str], batch_size: int = 32, prefix: str = "") -> list[list[float]]:
        """Embed *texts*, each truncated to fit the context and then prefixed."""
        reserved = len(prefix.split())
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = [prefix + self._truncate(t, reserved) for t in texts[i : i + batch_size]]
            results.extend(self._embed_batch(batch))
        return results

    def encode_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed texts to store, with the document prefix."""
        return self.encode(texts, batch_size, prefix=self.document_prefix)

    def encode_queries(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed search queries, with the query prefix."""
        return self.encode(texts, batch_size, prefix=self.query_prefix)
