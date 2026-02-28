"""
Qdrant vector store wrapper.

Each wiki page's chunks are stored as points in a single collection.
Re-running the ingestion for a page first deletes its old points so the
collection stays consistent (no stale chunks after a page is edited or
deleted).

The collection is created automatically on first use with COSINE distance,
which pairs well with normalized embeddings from sentence-transformers and
OpenAI.
"""

import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(
        self,
        host: str,
        port: int,
        collection: str,
        vector_size: int,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
    ):
        self._client = QdrantClient(
            host=host,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
        )
        self._collection = collection
        self._vector_size = vector_size
        self._ensure_collection()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection '%s' (dim=%d)", self._collection, self._vector_size)
        else:
            logger.debug("Using existing Qdrant collection '%s'", self._collection)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def delete_page(self, page_id: int) -> None:
        """Remove all chunks that belong to *page_id*."""
        try:
            self._client.delete(
                collection_name=self._collection,
                points_selector=Filter(
                    must=[FieldCondition(key="page_id", match=MatchValue(value=page_id))]
                ),
            )
        except UnexpectedResponse as exc:
            logger.warning("Could not delete page %d chunks: %s", page_id, exc)

    def upsert_page_chunks(
        self,
        page_id: int,
        vectors: list[list[float]],
        payloads: list[dict],
    ) -> None:
        """
        Replace all stored chunks for *page_id* with the new vectors.

        Args:
            page_id:  Wiki.js page ID (used to purge stale data first).
            vectors:  Embedding vectors, one per chunk.
            payloads: Metadata dicts, one per chunk.  ``page_id`` is
                      injected automatically.
        """
        assert len(vectors) == len(payloads), "vectors and payloads must have the same length"

        self.delete_page(page_id)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={**payload, "page_id": page_id},
            )
            for vec, payload in zip(vectors, payloads)
        ]

        self._client.upsert(collection_name=self._collection, points=points)
        logger.info("Stored %d chunks for page %d", len(points), page_id)

    def collection_info(self) -> dict:
        info = self._client.get_collection(self._collection)
        # vectors_count was made optional (and removed in some client versions);
        # fall back to points_count which is always present.
        vectors_count = getattr(info, "vectors_count", None) or getattr(info, "points_count", None)
        return {
            "name": self._collection,
            "vectors_count": vectors_count,
            "points_count": getattr(info, "points_count", None),
            "status": str(info.status),
        }
