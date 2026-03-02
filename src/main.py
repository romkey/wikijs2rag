"""
wiki2rag – ingest Wiki.js public pages into a Qdrant vector store.

All configuration is via environment variables (see .env.example).
Run once manually, on a schedule, or via `docker compose run wiki2rag`.

Features:
  - Incremental ingestion: skips pages whose updatedAt hasn't changed
  - Enriched context per chunk for LLM grounding
  - Parent/prev/next chunk references for small-to-big retrieval
  - Content hash per chunk for deduplication
"""

import logging
import os
import sys
import time
import uuid

from chunker import Chunk, chunk_page
from embedder import build_embedder
from store import VectorStore
from version import __version__
from wiki_client import WikiClient, WikiClientError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("wiki2rag")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _require_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        logger.error("Required environment variable %s is not set.", name)
        sys.exit(1)
    return val


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context(
    chunk: Chunk,
    page_title: str,
    page_description: str,
) -> str:
    """
    Build an enriched context string for LLM consumption.

    Includes page title, description, and section breadcrumb so the LLM
    knows *where* this chunk comes from without the chatbot needing to
    reconstruct that from individual payload fields.
    """
    parts: list[str] = []

    if page_title:
        parts.append(f"Page: {page_title}")

    if page_description:
        parts.append(f"Summary: {page_description}")

    if chunk.section_breadcrumb:
        parts.append(f"Section: {chunk.section_breadcrumb}")

    if parts:
        header = " | ".join(parts)
        return f"[{header}]\n\n{chunk.text}"
    else:
        return chunk.text


# ---------------------------------------------------------------------------
# Payload builder with parent/prev/next references
# ---------------------------------------------------------------------------

def _build_payloads(
    chunks: list[Chunk],
    page: dict,
    page_url: str,
    parent_chunk_size: int,
) -> list[dict]:
    """
    Build payload dicts for each chunk.

    Assigns stable UUIDs per chunk and cross-references:
      - prev_chunk_id / next_chunk_id  for sequential traversal
      - parent_chunk_index             index of the "parent" chunk in a
                                       coarser-grained view (every Nth chunk)
    """
    page_title = page.get("title") or ""
    page_description = page.get("description") or ""
    tags = [t["tag"] for t in (page.get("tags") or [])]
    updated_at = page.get("updatedAt") or ""

    chunk_ids = [str(uuid.uuid4()) for _ in chunks]

    payloads: list[dict] = []
    for i, chunk in enumerate(chunks):
        context = _build_context(chunk, page_title, page_description)

        parent_idx = (i // parent_chunk_size) * parent_chunk_size

        payload = {
            "chunk_id":           chunk_ids[i],
            "text":               chunk.text,
            "context":            context,
            "chunk_index":        chunk.chunk_index,
            "section":            chunk.section,
            "section_breadcrumb": chunk.section_breadcrumb,
            "content_hash":       chunk.content_hash,
            "page_path":          page["path"],
            "page_title":         page_title,
            "page_url":           page_url,
            "description":        page_description,
            "tags":               tags,
            "updated_at":         updated_at,
            "prev_chunk_id":      chunk_ids[i - 1] if i > 0 else None,
            "next_chunk_id":      chunk_ids[i + 1] if i < len(chunks) - 1 else None,
            "parent_chunk_index": parent_idx,
            "total_chunks":       len(chunks),
        }
        payloads.append(payload)

    return payloads


# ---------------------------------------------------------------------------
# Main ingestion loop
# ---------------------------------------------------------------------------

def run() -> None:
    wiki_url       = _require_env("WIKI_URL")
    api_key        = os.environ.get("WIKI_API_KEY", "").strip() or None
    qdrant_host    = _env("QDRANT_HOST", "qdrant")
    qdrant_port    = int(_env("QDRANT_PORT", "6333"))
    collection     = _env("QDRANT_COLLECTION", "wiki")
    chunk_size     = int(_env("CHUNK_SIZE", "256"))
    chunk_overlap  = int(_env("CHUNK_OVERLAP", "50"))
    batch_size     = int(_env("EMBEDDING_BATCH_SIZE", "32"))
    page_delay     = float(_env("PAGE_DELAY_SECONDS", "0.1"))
    force_reingest = _env_bool("FORCE_REINGEST", False)
    parent_ratio   = int(_env("PARENT_CHUNK_RATIO", "4"))

    embedder = build_embedder()

    logger.info("Connecting to Qdrant at %s:%d …", qdrant_host, qdrant_port)
    for attempt in range(1, 7):
        try:
            store = VectorStore(qdrant_host, qdrant_port, collection, embedder.dimension)
            break
        except Exception as exc:
            if attempt == 6:
                logger.error("Cannot connect to Qdrant: %s", exc)
                sys.exit(1)
            logger.warning("Qdrant not ready (attempt %d/6): %s – retrying in 5 s…", attempt, exc)
            time.sleep(5)

    with WikiClient(wiki_url, api_key) as wiki:
        logger.info("wiki2rag %s  –  fetching public page list from %s …", __version__, wiki_url)
        try:
            pages = wiki.list_public_pages()
        except WikiClientError as exc:
            logger.error("Failed to list pages: %s", exc)
            sys.exit(1)

        if not pages:
            logger.warning("No public pages found – nothing to do.")
            return

        total = len(pages)
        ok = skipped = unchanged = errors = 0

        for i, meta in enumerate(pages, 1):
            page_id = meta["id"]
            title   = meta.get("title") or meta.get("path") or str(page_id)
            logger.info("[%d/%d] Processing page %d: %s", i, total, page_id, title)

            # --- Incremental: skip if updatedAt hasn't changed ---
            if not force_reingest:
                stored_ts = store.get_page_updated_at(page_id)
                wiki_ts = meta.get("updatedAt") or ""
                if stored_ts and wiki_ts and stored_ts == wiki_ts:
                    logger.debug("  Page %d unchanged (updatedAt=%s), skipping.", page_id, wiki_ts)
                    unchanged += 1
                    continue

            try:
                page = wiki.get_page(page_id, meta=meta)
            except WikiClientError as exc:
                logger.error("  Could not fetch page %d: %s", page_id, exc)
                errors += 1
                continue

            if not page or not (page.get("content") or "").strip():
                logger.debug("  Page %d has no content, skipping.", page_id)
                skipped += 1
                continue

            content_type = (page.get("contentType") or "markdown").lower()
            chunks = chunk_page(
                page["content"],
                content_type=content_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                page_title=page.get("title") or "",
                page_description=page.get("description") or "",
            )

            if not chunks:
                logger.debug("  Page %d produced no chunks, skipping.", page_id)
                skipped += 1
                continue

            texts = [c.text for c in chunks]
            try:
                vectors = embedder.encode(texts, batch_size=batch_size)
            except Exception as exc:
                logger.error("  Embedding failed for page %d: %s", page_id, exc)
                errors += 1
                continue

            page_url = f"{wiki_url.rstrip('/')}/{page['path'].lstrip('/')}"
            payloads = _build_payloads(chunks, page, page_url, parent_ratio)

            try:
                store.upsert_page_chunks(page_id, vectors, payloads)
            except Exception as exc:
                logger.error("  Store failed for page %d: %s", page_id, exc)
                errors += 1
                continue

            ok += 1
            if page_delay > 0:
                time.sleep(page_delay)

    info = store.collection_info()
    logger.info(
        "Done.  pages_ingested=%d  unchanged=%d  skipped=%d  errors=%d  | "
        "collection='%s'  total_chunks=%s",
        ok, unchanged, skipped, errors,
        info["name"], info["vectors_count"],
    )

    if errors:
        sys.exit(2)


if __name__ == "__main__":
    run()
