import threading
from collections import Counter
from dataclasses import replace
from typing import Any, ClassVar

import pytest

from wikijs2rag import ingest
from wikijs2rag.chunker import Chunk
from wikijs2rag.config import Settings
from wikijs2rag.wiki_client import JSON, WikiClientError

SETTINGS = Settings(wiki_url="http://wiki", page_delay=0.01)

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_build_context() -> None:
    c = Chunk(text="body", chunk_index=0, section_breadcrumb="A > B")
    assert ingest.build_context(c, "T", "D") == "[Page: T | Summary: D | Section: A > B]\n\nbody"
    assert ingest.build_context(Chunk(text="body", chunk_index=0), "", "") == "body"


def test_build_payloads_links_chunks() -> None:
    chunks = [Chunk(text=f"t{i}", chunk_index=i) for i in range(5)]
    page = {"path": "p", "title": "T", "tags": [{"tag": "a"}], "updatedAt": "u"}
    payloads = ingest.build_payloads(chunks, page, "http://w/p", parent_chunk_size=2)
    assert payloads[0]["prev_chunk_id"] is None
    assert payloads[-1]["next_chunk_id"] is None
    assert payloads[1]["prev_chunk_id"] == payloads[0]["chunk_id"]
    assert payloads[1]["next_chunk_id"] == payloads[2]["chunk_id"]
    assert [p["parent_chunk_index"] for p in payloads] == [0, 0, 2, 2, 4]
    assert payloads[0]["tags"] == ["a"]
    assert payloads[0]["total_chunks"] == 5


def test_build_wiki_metadata_full() -> None:
    pages: list[JSON] = [
        {"id": 1, "path": "b", "title": "B", "updatedAt": "2024-02-01T00:00:00Z"},
        {"id": 2, "path": "a", "title": "A", "updatedAt": "bad-date", "createdAt": "2023-01-01T00:00:00Z"},
        {"id": 3, "path": "c", "updatedAt": "2024-03-01T00:00:00Z"},
    ]
    extra: dict[int, JSON] = {1: {"authorName": "Ann", "createdAt": "2022-05-05T00:00:00Z"}, 3: {"authorName": "Bob"}}
    chunks = ingest.build_wiki_metadata(pages, "http://w", all_tags=Counter({"x": 2, "y": 1}), page_extra=extra)
    by_type = {c["meta_type"]: c["text"] for c in chunks}
    assert set(by_type) == {"overview", "contributors", "tags", "recent", "newest", "page_listing"}
    assert "Total public pages: 3" in by_type["overview"]
    assert 'Most recently updated page: "" on 2024-03-01' in by_type["overview"]
    assert 'Oldest page: "B" created on 2022-05-05' in by_type["overview"]
    assert "  - Ann: 1 page" in by_type["contributors"].splitlines()
    assert "x: 2 pages" in by_type["tags"]
    assert '"c" updated on 2024-03-01 by Bob' in by_type["recent"]
    assert '"B" created on 2022-05-05 by Ann' in by_type["newest"]
    assert '"A" created on 2023-01-01' in by_type["newest"]
    assert by_type["page_listing"].splitlines()[1:] == ["  - A (a)", "  - B (b)", "  - Untitled (c)"]
    assert chunks[0]["context"].startswith("[Wiki Metadata | Overview]\n\n")


def test_build_wiki_metadata_minimal() -> None:
    chunks = ingest.build_wiki_metadata([{"id": 1}], "http://w")
    assert [c["meta_type"] for c in chunks] == ["overview", "contributors", "page_listing"]


# ---------------------------------------------------------------------------
# Fakes for run(): Ollama, Qdrant and Wiki.js at their client boundary
# ---------------------------------------------------------------------------


class FakeEmbedder:
    dimension = 3

    def __init__(self, *_a: object, **_k: object) -> None:
        pass

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if any("EMBEDFAIL" in t for t in texts):
            raise RuntimeError("embed fail")
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeStore:
    instances: ClassVar[list[FakeStore]] = []
    fail_meta = False

    def __init__(self, *_a: object) -> None:
        self.pages: dict[int, list[JSON]] = {}
        self.meta: list[JSON] | None = None
        FakeStore.instances.append(self)

    def get_page_updated_at(self, page_id: int) -> str | None:
        return "same" if page_id == 10 else None

    def upsert_page_chunks(self, page_id: int, vectors: list[list[float]], payloads: list[JSON]) -> None:
        if payloads[0]["page_title"] == "StoreFail":
            raise RuntimeError("store fail")
        self.pages[page_id] = payloads

    def upsert_meta_chunks(self, vectors: list[list[float]], payloads: list[JSON]) -> None:
        if FakeStore.fail_meta:
            raise RuntimeError("meta fail")
        self.meta = payloads

    def collection_info(self) -> dict[str, Any]:  # mirrors VectorStore.collection_info
        return {"name": "wiki", "vectors_count": 1}


class FakeWiki:
    pages_list: ClassVar[list[JSON]] = []
    details: ClassVar[dict[int, JSON | Exception]] = {}
    list_error = False

    def __init__(self, *_a: object) -> None:
        pass

    def __enter__(self) -> FakeWiki:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def list_public_pages(self) -> list[JSON]:
        if FakeWiki.list_error:
            raise WikiClientError("nope")
        return FakeWiki.pages_list

    def get_page(self, page_id: int, meta: JSON | None = None) -> JSON:
        d = FakeWiki.details[page_id]
        if isinstance(d, Exception):
            raise d
        return d


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ingest, "OllamaEmbedder", FakeEmbedder)
    monkeypatch.setattr(ingest, "VectorStore", FakeStore)
    monkeypatch.setattr(ingest, "WikiClient", FakeWiki)
    FakeStore.instances = []
    FakeStore.fail_meta = False
    FakeWiki.list_error = False
    FakeWiki.pages_list = [
        {"id": 1, "path": "ok", "title": "OK", "updatedAt": "new"},
        {"id": 2, "path": "err"},
        {"id": 3, "path": "empty"},
        {"id": 4, "path": "embedfail"},
        {"id": 5, "path": "storefail"},
        {"id": 6, "path": "nochunks"},
        {"id": 10, "path": "unchanged", "updatedAt": "same"},
    ]
    FakeWiki.details = {
        1: {
            "path": "ok",
            "title": "OK",
            "content": "# H\nhello world",
            "tags": [{"tag": "t"}],
            "authorName": "Ann",
            "createdAt": "2024-01-01T00:00:00Z",
        },
        2: WikiClientError("fetch fail"),
        3: {"path": "empty", "content": "  "},
        4: {"path": "embedfail", "content": "EMBEDFAIL text", "tags": ["plain"]},
        5: {"path": "storefail", "title": "StoreFail", "content": "text"},
        6: {"path": "nochunks", "content": "<p></p>", "contentType": "html"},
    }


@pytest.mark.usefixtures("fakes")
def test_run_processes_pages() -> None:
    assert ingest.run(SETTINGS) is False  # some pages failed
    store = FakeStore.instances[0]
    assert list(store.pages) == [1]
    assert store.pages[1][0]["page_url"] == "http://wiki/ok"
    assert store.meta is not None
    assert {p["meta_type"] for p in store.meta} >= {"overview", "tags"}


@pytest.mark.usefixtures("fakes")
def test_run_force_reingest_and_clean_success() -> None:
    FakeWiki.pages_list = [FakeWiki.pages_list[0], {"id": 10, "path": "unchanged", "updatedAt": "same"}]
    FakeWiki.details[10] = {"path": "unchanged", "content": "again"}
    assert ingest.run(replace(SETTINGS, force_reingest=True, page_delay=0)) is True
    assert set(FakeStore.instances[0].pages) == {1, 10}


@pytest.mark.usefixtures("fakes")
def test_run_meta_failure_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    FakeWiki.pages_list = FakeWiki.pages_list[:1]
    FakeStore.fail_meta = True
    assert ingest.run(SETTINGS) is True
    assert FakeStore.instances[0].meta is None
    assert "Failed to ingest wiki metadata" in caplog.text


@pytest.mark.usefixtures("fakes")
def test_run_no_pages() -> None:
    FakeWiki.pages_list = []
    assert ingest.run(SETTINGS) is True


@pytest.mark.usefixtures("fakes")
def test_run_list_error_raises() -> None:
    FakeWiki.list_error = True
    with pytest.raises(ingest.IngestError, match="Failed to list pages"):
        ingest.run(SETTINGS)


@pytest.mark.usefixtures("fakes")
def test_run_stops_between_pages() -> None:
    stop = threading.Event()
    stop.set()
    assert ingest.run(SETTINGS, stop) is True
    assert FakeStore.instances[0].pages == {}


class InstantEvent(threading.Event):
    """An Event whose wait() returns immediately, recording the timeout."""

    def __init__(self) -> None:
        super().__init__()
        self.waits: list[float | None] = []

    def wait(self, timeout: float | None = None) -> bool:
        self.waits.append(timeout)
        return self.is_set()


@pytest.mark.usefixtures("fakes")
def test_connect_store_retries_then_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[int] = []

    def bad_store(*_a: object) -> FakeStore:
        attempts.append(1)
        raise ConnectionError("no qdrant")

    monkeypatch.setattr(ingest, "VectorStore", bad_store)
    stop = InstantEvent()
    with pytest.raises(ingest.IngestError, match="Cannot connect to Qdrant"):
        ingest.connect_store(SETTINGS, 3, stop)
    assert len(attempts) == ingest.QDRANT_CONNECT_ATTEMPTS
    assert stop.waits == [ingest.QDRANT_RETRY_SECONDS] * (ingest.QDRANT_CONNECT_ATTEMPTS - 1)


@pytest.mark.usefixtures("fakes")
def test_connect_store_retry_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def flaky(*a: object) -> FakeStore:
        calls.append(1)
        if len(calls) < 2:
            raise ConnectionError("not yet")
        return FakeStore(*a)

    monkeypatch.setattr(ingest, "VectorStore", flaky)
    assert isinstance(ingest.connect_store(SETTINGS, 3, InstantEvent()), FakeStore)
    assert len(calls) == 2


@pytest.mark.usefixtures("fakes")
def test_connect_store_gives_up_when_stopping(monkeypatch: pytest.MonkeyPatch) -> None:
    def bad_store(*_a: object) -> FakeStore:
        raise ConnectionError("no qdrant")

    monkeypatch.setattr(ingest, "VectorStore", bad_store)
    stop = InstantEvent()
    stop.set()
    with pytest.raises(ingest.IngestError):
        ingest.connect_store(SETTINGS, 3, stop)
    assert stop.waits == []
