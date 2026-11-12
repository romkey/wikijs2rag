import httpx
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from wikijs2rag.store import VectorStore


@pytest.fixture
def vs(memory_qdrant: QdrantClient) -> VectorStore:
    return VectorStore("unused", 0, "wiki", 3)


def test_creates_collection_once(memory_qdrant: QdrantClient) -> None:
    VectorStore("h", 1, "wiki", 3)
    VectorStore("h", 1, "wiki", 3)  # existing collection path
    assert [c.name for c in memory_qdrant.get_collections().collections] == ["wiki"]


def test_upsert_replaces_page_chunks(vs: VectorStore, memory_qdrant: QdrantClient) -> None:
    vs.upsert_page_chunks(1, [[1, 0, 0], [0, 1, 0]], [{"updated_at": "t1"}, {"updated_at": "t1"}])
    vs.upsert_page_chunks(2, [[0, 0, 1]], [{"updated_at": "x"}])
    vs.upsert_page_chunks(1, [[1, 1, 0]], [{"updated_at": "t2"}])
    assert memory_qdrant.count("wiki").count == 2
    assert vs.get_page_updated_at(1) == "t2"
    assert vs.get_page_updated_at(99) is None


def test_get_page_updated_at_missing_field(vs: VectorStore) -> None:
    vs.upsert_page_chunks(1, [[1, 0, 0]], [{}])
    assert vs.get_page_updated_at(1) is None


def test_upsert_length_mismatch(vs: VectorStore) -> None:
    with pytest.raises(ValueError, match="1 vectors but 0 payloads"):
        vs.upsert_page_chunks(1, [[1, 0, 0]], [])


def test_upsert_meta_chunks_replaces(vs: VectorStore, memory_qdrant: QdrantClient) -> None:
    vs.upsert_page_chunks(1, [[1, 0, 0]], [{}])
    vs.upsert_meta_chunks([[0, 1, 0], [0, 0, 1]], [{"meta_type": "a"}, {"meta_type": "b"}])
    vs.upsert_meta_chunks([[0, 1, 1]], [{"meta_type": "c"}])
    points, _ = memory_qdrant.scroll("wiki", limit=10)
    metas = [p.payload for p in points if p.payload and p.payload.get("is_meta") == "true"]
    assert [m["meta_type"] for m in metas if m] == ["c"]
    assert metas[0] is not None
    assert metas[0]["page_id"] == 0
    assert len(points) == 2


def test_collection_info(vs: VectorStore) -> None:
    vs.upsert_page_chunks(1, [[1, 0, 0]], [{}])
    info = vs.collection_info()
    assert info["name"] == "wiki"
    assert info["points_count"] == 1


def test_get_page_updated_at_swallows_errors(vs: VectorStore, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**_kw: object) -> None:
        raise RuntimeError("down")

    monkeypatch.setattr(vs._client, "scroll", boom)
    assert vs.get_page_updated_at(1) is None


def test_index_and_delete_errors_are_tolerated(vs: VectorStore, monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_unexpected(**_kw: object) -> None:
        raise UnexpectedResponse(status_code=400, reason_phrase="bad", content=b"", headers=httpx.Headers())

    monkeypatch.setattr(vs._client, "create_payload_index", raise_unexpected)
    vs._ensure_indexes()
    monkeypatch.setattr(vs._client, "delete", raise_unexpected)
    vs.delete_page(1)
    vs.upsert_meta_chunks([[1, 0, 0]], [{}])
