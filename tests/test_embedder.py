import json

import httpx
import pytest
import respx

from wikijs2rag.embedder import OpenAIEmbedder, default_prefixes

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


def _sent(route: respx.Route) -> list[str]:
    inputs: list[str] = json.loads(route.calls.last.request.content)["input"]
    return inputs


@respx.mock
def test_probe_has_no_prefix() -> None:
    route = _mock_embed()
    OpenAIEmbedder("m", base_url=URL, query_prefix="q: ", document_prefix="d: ")
    assert _sent(route) == ["dimension probe"]


@respx.mock
def test_encode_documents_and_queries_use_their_prefixes() -> None:
    route = _mock_embed()
    e = OpenAIEmbedder("m", base_url=URL, query_prefix="search_query: ", document_prefix="search_document: ")
    e.encode_documents(["a page", "another"])
    assert _sent(route) == ["search_document: a page", "search_document: another"]
    e.encode_queries(["door code"])
    assert _sent(route) == ["search_query: door code"]
    e.encode(["raw"])
    assert _sent(route) == ["raw"]


@respx.mock
def test_truncation_leaves_room_for_the_prefix() -> None:
    route = _mock_embed()
    e = OpenAIEmbedder("m", base_url=URL, context_length=8, document_prefix="search document: ")  # 6 words
    e.encode_documents(["one two three four five six seven eight"])
    assert _sent(route) == ["search document: one two three four"]


QWEN_QUERY = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("nomic-ai/modernbert-embed-base", ("search_query: ", "search_document: ")),
        ("lightonai/modernbert-embed-large", ("search_query: ", "search_document: ")),
        ("nomic-ai/nomic-embed-text-v1.5", ("search_query: ", "search_document: ")),
        ("intfloat/e5-large-v2", ("query: ", "passage: ")),
        ("mixedbread-ai/mxbai-embed-large-v1", ("Represent this sentence for searching relevant passages: ", "")),
        ("Qwen/Qwen3-Embedding-0.6B", (QWEN_QUERY, "")),
        ("text-embedding-3-small", ("", "")),
    ],
)
def test_default_prefixes(model: str, expected: tuple[str, str]) -> None:
    assert default_prefixes(model) == expected
