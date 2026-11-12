import json

import httpx
import pytest
import respx

from wikijs2rag.embedder import OpenAIEmbedder

URL = "http://embeddings.test/v1"


def _mock_embed(dim: int = 3, reverse: bool = False) -> respx.Route:
    def handler(request: httpx.Request) -> httpx.Response:
        n = len(json.loads(request.content)["input"])
        items = [{"index": i, "embedding": [float(i)] * dim} for i in range(n)]
        if reverse:
            items.reverse()
        return httpx.Response(200, json={"data": items})

    return respx.post(f"{URL}/embeddings").mock(side_effect=handler)


@respx.mock
def test_init_probes_dimension_and_sends_auth() -> None:
    route = _mock_embed(dim=5)
    e = OpenAIEmbedder("m", base_url=URL + "/", api_key="dummy-key")
    assert e.dimension == 5
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer dummy-key"
    body = json.loads(request.content)
    assert body == {"model": "m", "input": ["dimension probe"], "encoding_format": "float"}


@respx.mock
def test_no_auth_header_without_key() -> None:
    route = _mock_embed()
    OpenAIEmbedder("m", base_url=URL)
    assert "Authorization" not in route.calls.last.request.headers


@respx.mock
def test_sends_dimensions_when_set() -> None:
    route = _mock_embed()
    OpenAIEmbedder("m", base_url=URL, dimensions=256)
    assert json.loads(route.calls.last.request.content)["dimensions"] == 256


@respx.mock
def test_encode_batches_and_orders_by_index() -> None:
    route = _mock_embed(dim=1, reverse=True)
    e = OpenAIEmbedder("m", base_url=URL)
    out = e.encode(["a", "b", "c", "d", "e"], batch_size=2)
    assert out == [[0.0], [1.0], [0.0], [1.0], [0.0]]
    assert route.call_count == 1 + 3  # probe + 3 batches


@respx.mock
def test_encode_truncates_long_text() -> None:
    route = _mock_embed()
    e = OpenAIEmbedder("m", base_url=URL, context_length=8)  # max 6 words
    e.encode(["one two three four five six seven eight"])
    sent = json.loads(route.calls.last.request.content)["input"][0]
    assert sent == "one two three four five six"


@pytest.mark.parametrize("body", [{"error": "x"}, {"data": []}, ["not", "a", "dict"]])
@respx.mock
def test_unexpected_response_raises(body: object) -> None:
    respx.post(f"{URL}/embeddings").mock(return_value=httpx.Response(200, json=body))
    with pytest.raises(RuntimeError, match="Unexpected response"):
        OpenAIEmbedder("m", base_url=URL)


@respx.mock
def test_http_error_raises_with_body() -> None:
    respx.post(f"{URL}/embeddings").mock(return_value=httpx.Response(401, text="bad key"))
    with pytest.raises(RuntimeError, match=r"failed \(401\): bad key"):
        OpenAIEmbedder("m", base_url=URL)
