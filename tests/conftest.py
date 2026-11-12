import pytest
from qdrant_client import QdrantClient

from wikijs2rag import store


@pytest.fixture
def memory_qdrant(monkeypatch: pytest.MonkeyPatch) -> QdrantClient:
    """Make store.VectorStore use an in-process Qdrant instead of a server."""
    client = QdrantClient(location=":memory:")
    monkeypatch.setattr(store, "QdrantClient", lambda **_kw: client)
    return client
