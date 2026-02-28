"""
wiki2rag – ingest Wiki.js public pages into a Qdrant vector store.

All configuration is via environment variables (see .env.example).
Run once manually, on a schedule, or via `docker compose run wiki2rag`.
"""

import logging
import os
import sys
import time

from chunker import chunk_page
from embedder import build_embedder
from store import VectorStore
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


# ---------------------------------------------------------------------------
# Main ingestion loop
# ---------------------------------------------------------------------------

def run() -> None:
    wiki_url    = _require_env("WIKI_URL")
    api_key     = os.environ.get("WIKI_API_KEY", "").strip() or None
    qdrant_host = _env("QDRANT_HOST", "qdrant")
    qdrant_port = int(_env("QDRANT_PORT", "6333"))
    collection  = _env("QDRANT_COLLECTION", "wiki")
    chunk_size  = int(_env("CHUNK_SIZE", "512"))
    chunk_overlap = int(_env("CHUNK_OVERLAP", "64"))
    batch_size  = int(_env("EMBEDDING_BATCH_SIZE", "32"))
    page_delay  = float(_env("PAGE_DELAY_SECONDS", "0.1"))

    # Build embedder (may download a model – this is the slow part)
    embedder = build_embedder()

    # Connect to Qdrant (retry a few times in case Qdrant is still starting)
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
        logger.info("Fetching public page list from %s …", wiki_url)
        try:
            pages = wiki.list_public_pages()
        except WikiClientError as exc:
            logger.error("Failed to list pages: %s", exc)
            sys.exit(1)

        if not pages:
            logger.warning("No public pages found – nothing to do.")
            return

        total = len(pages)
        ok = skipped = errors = 0

        for i, meta in enumerate(pages, 1):
            page_id = meta["id"]
            title   = meta.get("title") or meta.get("path") or str(page_id)
            logger.info("[%d/%d] Processing page %d: %s", i, total, page_id, title)

            try:
                page = wiki.get_page(page_id)
            except WikiClientError as exc:
                # Wiki.js lists some restricted pages as public but rejects
                # the content fetch.  Treat those as skips, not errors.
                if "not authorized" in str(exc).lower() or "6013" in str(exc):
                    logger.warning("  Page %d is access-restricted, skipping.", page_id)
                    skipped += 1
                else:
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
            payloads = [
                {
                    "text":        chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "section":     chunk.section,
                    "page_path":   page["path"],
                    "page_title":  page["title"] or "",
                    "page_url":    page_url,
                    "description": page.get("description") or "",
                    "tags":        [t["tag"] for t in (page.get("tags") or [])],
                    "updated_at":  page.get("updatedAt") or "",
                }
                for chunk in chunks
            ]

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
        "Done.  pages_ok=%d  skipped=%d  errors=%d  | "
        "collection='%s'  total_chunks=%s",
        ok, skipped, errors,
        info["name"], info["vectors_count"],
    )

    if errors:
        sys.exit(2)


if __name__ == "__main__":
    run()
