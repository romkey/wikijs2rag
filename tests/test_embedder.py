import json

import httpx
import pytest
import respx

from wikijs2rag.embedder import OllamaEmbedder

URL = "http://ollama.test:11434"


def _mock_embed(dim: int = 3) -> respx.Route:
    def handler(request: httpx.Request) -> httpx.Response:
        n = len(json.loads(request.content)["input"])
        return httpx.Response(200, json={"embeddings": [[0.1] * dim for _ in range(n)]})

    return respx.post(f"{URL}/api/embed").mock(side_effect=handler)


@respx.mock
def test_init_probes_dimension_and_sends_auth() -> None:
    route = _mock_embed(dim=5)
    e = OllamaEmbedder("m", base_url=URL + "/", api_key="dummy-key")
    assert e.dimension == 5
    assert route.calls.last.request.headers["Authorization"] == "Bearer dummy-key"


@respx.mock
def test_no_auth_header_without_key() -> None:
    route = _mock_embed()
    OllamaEmbedder("m", base_url=URL)
    assert "Authorization" not in route.calls.last.request.headers


@respx.mock
def test_encode_batches() -> None:
    route = _mock_embed()
    e = OllamaEmbedder("m", base_url=URL)
    out = e.encode(["a", "b", "c", "d", "e"], batch_size=2)
    assert len(out) == 5
    assert route.call_count == 1 + 3  # probe + 3 batches


@respx.mock
def test_encode_truncates_long_text() -> None:
    route = _mock_embed()
    e = OllamaEmbedder("m", base_url=URL, context_length=8)  # max 6 words
    e.encode(["one two three four five six seven eight"])
    sent = json.loads(route.calls.last.request.content)["input"][0]
    assert sent == "one two three four five six"


@respx.mock
def test_unexpected_response_raises() -> None:
    respx.post(f"{URL}/api/embed").mock(return_value=httpx.Response(200, json={"error": "x"}))
    with pytest.raises(RuntimeError, match="Unexpected response"):
        OllamaEmbedder("m", base_url=URL)


@respx.mock
def test_http_error_raises() -> None:
    respx.post(f"{URL}/api/embed").mock(return_value=httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        OllamaEmbedder("m", base_url=URL)
