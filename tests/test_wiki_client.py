import json
import time
from collections.abc import Iterator

import httpx
import pytest
import respx
from bs4 import BeautifulSoup

from wikijs2rag import wiki_client
from wikijs2rag.wiki_client import JSON, WikiClient, WikiClientError, WikiPageForbiddenError

BASE = "http://wiki.test"
GQL = f"{BASE}/graphql"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _s: None)


@pytest.fixture
def client() -> Iterator[WikiClient]:
    with WikiClient(BASE + "/", api_key="dummy-key") as c:
        yield c


@respx.mock
def test_list_public_pages_filters_and_sends_auth(client: WikiClient) -> None:
    pages = [
        {"id": 1, "isPublished": True, "isPrivate": False},
        {"id": 2, "isPublished": False, "isPrivate": False},
        {"id": 3, "isPublished": True, "isPrivate": True},
    ]
    route = respx.post(GQL).mock(return_value=httpx.Response(200, json={"data": {"pages": {"list": pages}}}))
    assert [p["id"] for p in client.list_public_pages()] == [1]
    assert route.calls.last.request.headers["Authorization"] == "Bearer dummy-key"


def test_no_auth_header_without_key() -> None:
    with WikiClient(BASE) as c:
        assert "Authorization" not in c._client.headers


@respx.mock
def test_get_page_returns_single(client: WikiClient) -> None:
    page = {"id": 7, "content": "hi"}
    route = respx.post(GQL).mock(return_value=httpx.Response(200, json={"data": {"pages": {"single": page}}}))
    assert client.get_page(7) == page
    assert json.loads(route.calls.last.request.content)["variables"] == {"id": 7}


@respx.mock
def test_graphql_error_raises(client: WikiClient) -> None:
    respx.post(GQL).mock(return_value=httpx.Response(200, json={"errors": [{"message": "boom"}]}))
    with pytest.raises(WikiClientError, match="boom"):
        client.list_public_pages()


@respx.mock
@pytest.mark.parametrize(
    "err",
    [
        {"message": "x", "extensions": {"exception": {"code": 6013}}},
        {"message": "You are Not Authorized"},
    ],
)
def test_forbidden_without_meta_raises(client: WikiClient, err: JSON) -> None:
    route = respx.post(GQL).mock(return_value=httpx.Response(200, json={"errors": [err]}))
    with pytest.raises(WikiPageForbiddenError):
        client.get_page(1)
    assert route.call_count == 1  # not retried


@respx.mock
def test_forbidden_falls_back_to_scrape(client: WikiClient) -> None:
    respx.post(GQL).mock(return_value=httpx.Response(200, json={"errors": [{"message": "not authorized"}]}))
    html = "<html><body><nav>menu</nav><div class='contents'><p>Body text</p></div><footer>f</footer></body></html>"
    respx.get(f"{BASE}/docs/page").mock(return_value=httpx.Response(200, text=html))
    page = client.get_page(5, meta={"id": 5, "path": "/docs/page", "title": "T", "updatedAt": "2024"})
    assert page is not None
    assert page["contentType"] == "html"
    assert "Body text" in page["content"]
    assert "menu" not in page["content"]
    assert (page["title"], page["updatedAt"], page["id"]) == ("T", "2024", 5)


@respx.mock
def test_scrape_falls_back_to_body(client: WikiClient) -> None:
    respx.get(f"{BASE}/p").mock(return_value=httpx.Response(200, text="<html><body><p>Only</p></body></html>"))
    page = client._scrape_page("p", {})
    assert page is not None
    assert "Only" in page["content"]
    assert page["title"] == ""


@respx.mock
def test_scrape_returns_none_without_body(client: WikiClient, monkeypatch: pytest.MonkeyPatch) -> None:
    respx.get(f"{BASE}/p").mock(return_value=httpx.Response(200, text=""))

    def empty_soup(*_a: object, **_k: object) -> BeautifulSoup:
        soup = BeautifulSoup("", "html.parser")
        assert soup.body is None
        return soup

    monkeypatch.setattr(wiki_client, "BeautifulSoup", empty_soup)
    assert client._scrape_page("p", {}) is None


@respx.mock
def test_scrape_http_error(client: WikiClient) -> None:
    respx.get(f"{BASE}/p").mock(return_value=httpx.Response(404))
    with pytest.raises(WikiClientError, match="scrape failed"):
        client._scrape_page("p", {})


@respx.mock
def test_retries_on_5xx_then_succeeds(client: WikiClient) -> None:
    route = respx.post(GQL).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(429),
            httpx.Response(200, json={"data": {"pages": {"list": []}}}),
        ]
    )
    assert client.list_public_pages() == []
    assert route.call_count == 3


@respx.mock
def test_gives_up_after_max_retries_on_5xx(client: WikiClient) -> None:
    route = respx.post(GQL).mock(return_value=httpx.Response(500))
    with pytest.raises(WikiClientError):
        client.list_public_pages()
    assert route.call_count == 3


@respx.mock
def test_4xx_not_retried(client: WikiClient) -> None:
    route = respx.post(GQL).mock(return_value=httpx.Response(400))
    with pytest.raises(WikiClientError):
        client.list_public_pages()
    assert route.call_count == 1


@respx.mock
def test_request_error_retried_then_raises(client: WikiClient) -> None:
    route = respx.post(GQL).mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(WikiClientError, match="down"):
        client.list_public_pages()
    assert route.call_count == 3


def test_zero_retries_raises_exceeded() -> None:
    with WikiClient(BASE, max_retries=0) as c, pytest.raises(WikiClientError, match="Exceeded max retries"):
        c.list_public_pages()
