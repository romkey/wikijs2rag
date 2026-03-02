"""
Qdrant vector store wrapper.

Each wiki page's chunks are stored as points in a single collection.
Re-running the ingestion for a page first deletes its old points so the
collection stays consistent (no stale chunks after a page is edited or
deleted).

Payload indexes are created on filterable fields (page_id, tags, page_path,
content_hash) so that filtered vector search and delete-by-filter are fast.
"""

import logging
import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    ScrollResult,
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
        self._ensure_indexes()

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

    def _ensure_indexes(self) -> None:
        """Create payload field indexes for efficient filtering."""
        index_fields = {
            "page_id":      PayloadSchemaType.INTEGER,
            "tags":         PayloadSchemaType.KEYWORD,
            "page_path":    PayloadSchemaType.KEYWORD,
            "content_hash": PayloadSchemaType.KEYWORD,
        }
        for field_name, schema_type in index_fields.items():
            try:
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field_name,
                    field_schema=schema_type,
                )
            except UnexpectedResponse:
                pass  # index already exists

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

    def get_page_updated_at(self, page_id: int) -> Optional[str]:
        """Return the updated_at value for the first stored chunk of *page_id*, or None."""
        try:
            result: ScrollResult = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="page_id", match=MatchValue(value=page_id))]
                ),
                limit=1,
                with_payload=["updated_at"],
                with_vectors=False,
            )
            points = result[0] if isinstance(result, tuple) else result.points
            if points:
                return points[0].payload.get("updated_at")
        except Exception:
            pass
        return None

    def upsert_page_chunks(
        self,
        page_id: int,
        vectors: list[list[float]],
        payloads: list[dict],
    ) -> None:
        """
        Replace all stored chunks for *page_id* with the new vectors.
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
        vectors_count = getattr(info, "vectors_count", None) or getattr(info, "points_count", None)
        return {
            "name": self._collection,
            "vectors_count": vectors_count,
            "points_count": getattr(info, "points_count", None),
            "status": str(info.status),
        }
