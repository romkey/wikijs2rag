from collections.abc import Iterator
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from wikijs2rag.staging import Staging
from wikijs2rag.store import StoreError, VectorStore, point_id


@pytest.fixture
def staging(tmp_path: Path) -> Iterator[Staging]:
    with Staging(f"sqlite:///{tmp_path / 's.db'}") as s:
        yield s


def _payload(doc: str, i: int, run: str) -> dict[str, object]:
    return {
        "id": point_id("wiki", doc, i),
        "source": "wiki",
        "doc_id": doc,
        "text": "t",
        "context": "c",
        "sync_run_id": run,
    }


def test_store_replace_stamp_delete(memory_qdrant: QdrantClient) -> None:
    store = VectorStore("h", 1, "m", 2)
    store.ensure_collection("c", "public")
    store.ensure_collection("c", "public")  # idempotent
    store.replace_doc("c", "wiki", "1", [[0.1, 0.2], [0.3, 0.4]], [_payload("1", 0, "r1"), _payload("1", 1, "r1")])
    store.replace_doc("c", "wiki", "2", [[0.1, 0.2]], [_payload("2", 0, "r1")])
    store.replace_doc("c", "wiki", "1", [[0.1, 0.2]], [_payload("1", 0, "r1")])  # shrinks
    assert store.count("c") == 2
    store.stamp_doc("c", "wiki", "1", "r2")
    assert store.delete_stale("c", "wiki", "r2") == 1
    assert store.delete_stale("c", "wiki", "r2") == 0
    assert [d for d, _ in store.scroll_texts("c")] == ["1"]
    store.replace_doc("c", "wiki", "1", [], [])
    assert store.count("c", "wiki") == 0
    store.replace_doc("c", "wiki", "3", [[0.1, 0.2]], [_payload("3", 0, "r")])
    store.delete_source("c", "wiki")
    assert store.count("c") == 0
    with pytest.raises(ValueError, match="vectors"):
        store.replace_doc("c", "wiki", "1", [], [_payload("1", 0, "r")])


def test_store_refuses_other_model(memory_qdrant: QdrantClient) -> None:
    VectorStore("h", 1, "m", 2).ensure_collection("c", "public")
    with pytest.raises(StoreError, match="embedding_model"):
        VectorStore("h", 1, "other", 2).ensure_collection("c", "public")
    memory_qdrant.create_collection("old", vectors_config=VectorParams(size=2, distance=Distance.COSINE))
    with pytest.raises(StoreError, match="predates"):
        VectorStore("h", 1, "m", 2).ensure_collection("old", "public")


def test_point_ids_are_stable() -> None:
    assert point_id("wiki", "1", 0) == point_id("wiki", "1", 0) != point_id("wiki", "1", 1)


def test_staging_runs_and_status(staging: Staging) -> None:
    assert staging.status() is None
    assert staging.last_full_finished_at() is None
    assert staging.last_full_docs_seen() == 0
    staging.start_run("r1", "full")
    staging.finish_run("r1", "full", True, {"seen": 7})
    staging.start_run("r2", "incremental")
    staging.finish_run("r2", "incremental", False, {"seen": 3}, "boom")
    status = staging.status()
    assert status is not None
    assert (status["last_run_id"], status["ok"], status["error"]) == ("r2", False, "boom")
    assert staging.last_full_docs_seen() == 7
    assert staging.last_full_finished_at() is not None
    assert [r["run_id"] for r in staging.recent_runs()] == ["r2", "r1"]


def test_staging_cache_review_quarantine(staging: Staging) -> None:
    assert staging.cache_get("k") is None
    staging.cache_put("k", "task", "m", "v1")
    staging.cache_put("k", "task", "m", "v2")
    assert staging.cache_get("k") == "v2"
    staging.queue_for_review("1", "h", "T", "u", "why")
    staging.queue_for_review("1", "h", "T", "u", "again")  # once per content hash
    assert [r["reason"] for r in staging.pending_reviews()] == ["why"]
    assert staging.review_status("1", "h") == "pending"
    assert staging.review_status("1", "other") is None
    staging.quarantine_hit("1", "page", "jwt", 3, "h")
    staging.quarantine_hit("1", "page", "jwt", 3, "h")
    assert len(staging.quarantine_report()) == 1


def test_source_lock_without_postgres(staging: Staging) -> None:
    with staging.source_lock() as got:
        assert got is True
    assert staging.is_postgres is False
    staging.clear_entity_documents([])
    staging.mark_seen([], "r")
