"""OpenAI-compatible chat client for ingest-time LLM work, with a result cache.

Every result is cached in the staging database by a hash of (task, model,
prompt), so re-indexing unchanged content never calls the model again.
Model names are configuration; the gateway decides which machine runs them.
"""

import base64
import hashlib
import json
import logging
import re
from typing import Any

import httpx

from wikijs2rag.staging import Staging

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(
        self,
        staging: Staging,
        base_url: str,
        model: str,
        api_key: str | None = None,
        vision_model: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self._staging = staging
        self._url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.vision_model = vision_model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._client = httpx.Client(headers=headers, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def _key(self, task: str, model: str, prompt: str) -> str:
        return hashlib.sha256(f"{task}\0{model}\0{prompt}".encode()).hexdigest()

    def _complete(self, model: str, messages: list[dict[str, Any]], max_tokens: int) -> str:
        payload = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
        try:
            resp = self._client.post(self._url, json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"Request to {self._url} failed: {exc}") from exc
        if resp.is_error:
            raise LLMError(f"Request to {self._url} failed ({resp.status_code}): {resp.text[:300]}")
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response from {self._url}") from exc
        if not isinstance(content, str):
            raise LLMError(f"Unexpected response from {self._url}")
        # Reasoning models may prefix their answer with a <think> block.
        return re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()

    def text(self, task: str, system: str, prompt: str, max_tokens: int = 300) -> str:
        key = self._key(task, self.model, system + "\0" + prompt)
        cached = self._staging.cache_get(key)
        if cached is not None:
            return cached
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        result = self._complete(self.model, messages, max_tokens)
        self._staging.cache_put(key, task, self.model, result)
        return result

    def json(self, task: str, system: str, prompt: str, max_tokens: int = 300) -> Any:  # parsed JSON
        raw = self.text(task, system + "\nReply with JSON only, no other text.", prompt, max_tokens)
        return parse_json(raw)

    def describe_image(self, image: bytes, media_type: str, prompt: str, context_key: str) -> str:
        """Vision description of an image; cached by the image's hash."""
        if not self.vision_model:
            raise LLMError("No VISION_MODEL configured")
        digest = hashlib.sha256(image).hexdigest()
        key = self._key("vision", self.vision_model, digest + "\0" + prompt + "\0" + context_key)
        cached = self._staging.cache_get(key)
        if cached is not None:
            return cached
        data_url = f"data:{media_type};base64,{base64.b64encode(image).decode()}"
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": data_url}}],
            }
        ]
        result = self._complete(self.vision_model, messages, 600)
        self._staging.cache_put(key, "vision", self.vision_model, result)
        return result


def parse_json(raw: str) -> Any:  # parsed JSON
    """The first JSON object or array in *raw* (models wrap JSON in prose or fences)."""
    pairs = sorted((("{", "}"), ("[", "]")), key=lambda p: (raw.find(p[0]) == -1, raw.find(p[0])))
    for opener, closer in pairs:
        start, end = raw.find(opener), raw.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except ValueError:
                continue
    raise LLMError(f"Model reply is not JSON: {raw[:200]!r}")
