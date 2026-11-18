"""Qdrant collections: one per access tier, shared with the other sources.

Each point has a named dense vector (``dense``) and a named sparse keyword
vector (``sparse``, with Qdrant's IDF modifier). Point IDs are
uuid5(source:doc_id:chunk_index), so re-indexing overwrites. A document's
points are replaced as a unit: its old points are deleted, then the new ones
written. Collection metadata records the embedding model, its dimensions and
the sparse scheme; a mismatch stops the run, because mixing models in one
collection gives meaningless scores.
"""

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    Modifier,
    PayloadSchemaType,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

from wikijs2rag import sparse

logger = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "sparse"

INDEXES = {
    "source": PayloadSchemaType.KEYWORD,
    "tier": PayloadSchemaType.KEYWORD,
    "asset_ids": PayloadSchemaType.KEYWORD,
    "location_ids": PayloadSchemaType.KEYWORD,
    "topics": PayloadSchemaType.KEYWORD,
    "updated_at": PayloadSchemaType.DATETIME,
    "section": PayloadSchemaType.KEYWORD,
    "doc_id": PayloadSchemaType.KEYWORD,
    "sync_run_id": PayloadSchemaType.KEYWORD,
    "page_kind": PayloadSchemaType.KEYWORD,
    "chunk_kind": PayloadSchemaType.KEYWORD,
}


class StoreError(RuntimeError):
    pass


def point_id(source: str, doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{doc_id}:{chunk_index}"))


class VectorStore:
    def __init__(self, host: str, port: int, embedding_model: str, dimension: int) -> None:
        self.client = QdrantClient(host=host, port=port, prefer_grpc=False)
        self.embedding_model = embedding_model
        self.dimension = dimension

    def ensure_collection(self, name: str, tier: str) -> None:
        """Create *name* if missing; check its embedding model if not."""
        expected = {
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.dimension,
            "sparse_scheme": sparse.SCHEME,
            "tier": tier,
        }
        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config={DENSE: VectorParams(size=self.dimension, distance=Distance.COSINE)},
                sparse_vectors_config={SPARSE: SparseVectorParams(modifier=Modifier.IDF)},
                metadata=expected,
            )
            logger.info("Created Qdrant collection '%s' (dim=%d, tier=%s)", name, self.dimension, tier)
        else:
            info = self.client.get_collection(name)
            found = getattr(info.config, "metadata", None) or {}
            for key, value in expected.items():
                if key in found and found[key] != value:
                    raise StoreError(
                        f"Collection '{name}' has {key}={found[key]!r}, but this run uses {value!r}. "
                        "Every source in a collection must use the same embedding model; re-embed all of "
                        "them into new collections to change it."
                    )
            vectors = info.config.params.vectors
            if not isinstance(vectors, dict) or DENSE not in vectors:
                raise StoreError(f"Collection '{name}' has no '{DENSE}' vector; it predates wikijs2rag 3")
        for field_name, schema in INDEXES.items():
            try:
                self.client.create_payload_index(collection_name=name, field_name=field_name, field_schema=schema)
            except UnexpectedResponse as exc:
                logger.debug("Payload index %s not created (probably exists): %s", field_name, exc)

    @staticmethod
    def _doc_filter(source: str, doc_id: str) -> Filter:
        return Filter(
            must=[
                FieldCondition(key="source", match=MatchValue(value=source)),
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
            ]
        )

    def delete_doc(self, collection: str, source: str, doc_id: str) -> None:
        self.client.delete(collection_name=collection, points_selector=self._doc_filter(source, doc_id))

    def replace_doc(
        self,
        collection: str,
        source: str,
        doc_id: str,
        dense: list[list[float]],
        payloads: list[dict[str, Any]],  # JSON payloads stored as-is
    ) -> None:
        if len(dense) != len(payloads):
            raise ValueError(f"{len(dense)} vectors but {len(payloads)} payloads")
        self.delete_doc(collection, source, doc_id)
        if not payloads:
            return
        points = [
            PointStruct(
                id=payload["id"],
                vector={DENSE: vec, SPARSE: sparse.document_vector(f"{payload['context']}\n{payload['text']}")},
                payload=payload,
            )
            for vec, payload in zip(dense, payloads, strict=True)
        ]
        self.client.upsert(collection_name=collection, points=points)

    def stamp_doc(self, collection: str, source: str, doc_id: str, run_id: str) -> None:
        self.client.set_payload(
            collection_name=collection, payload={"sync_run_id": run_id}, points=self._doc_filter(source, doc_id)
        )

    def delete_stale(self, collection: str, source: str, run_id: str) -> int:
        """Delete *source*'s points not stamped with *run_id*; return how many."""
        stale = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source))],
            must_not=[FieldCondition(key="sync_run_id", match=MatchValue(value=run_id))],
        )
        count = self.client.count(collection_name=collection, count_filter=stale, exact=True).count
        if count:
            self.client.delete(collection_name=collection, points_selector=stale)
        return count

    def delete_source(self, collection: str, source: str) -> None:
        self.client.delete(
            collection_name=collection,
            points_selector=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))]),
        )

    def count(self, collection: str, source: str | None = None) -> int:
        flt = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))]) if source else None
        return self.client.count(collection_name=collection, count_filter=flt, exact=True).count

    def scroll_texts(self, collection: str) -> list[tuple[str, str]]:
        """(doc_id, text + context) of every point, for canary checks."""
        out: list[tuple[str, str]] = []
        offset: Any = None
        while True:
            points, offset = self.client.scroll(
                collection_name=collection,
                limit=256,
                offset=offset,
                with_payload=["doc_id", "text", "context", "title"],
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                joined = f"{payload.get('title', '')}\n{payload.get('context', '')}\n{payload.get('text', '')}"
                out.append((str(payload.get("doc_id", "")), joined))
            if offset is None:
                return out
