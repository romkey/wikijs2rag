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
from collections import Counter
from datetime import datetime

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
# Wiki metadata generation
# ---------------------------------------------------------------------------

def _build_wiki_metadata(
    pages: list[dict],
    wiki_url: str,
    all_tags: Counter | None = None,
    page_extra: dict | None = None,
) -> list[dict]:
    """
    Derive wiki-level metadata from the page list.

    *all_tags* is a Counter of tag→count collected during page ingestion
    (since the list query doesn't include tags).

    *page_extra* maps page_id → dict with keys like ``authorName`` and
    ``createdAt`` gathered from individual get_page calls (these fields
    aren't available on the list query).

    Returns a list of metadata chunk dicts, each with a 'text' field suitable
    for embedding and a 'context' field for LLM grounding.
    """
    if page_extra is None:
        page_extra = {}

    # Merge per-page details into the list entries
    for p in pages:
        extra = page_extra.get(p["id"], {})
        if "authorName" not in p and "authorName" in extra:
            p["authorName"] = extra["authorName"]
        if "createdAt" not in p and "createdAt" in extra:
            p["createdAt"] = extra["createdAt"]

    total_pages = len(pages)

    # Contributors (only counted for pages we actually fetched details for)
    authors = Counter(
        p.get("authorName")
        for p in pages
        if p.get("authorName")
    )
    num_contributors = len(authors)
    top_contributors = authors.most_common(10)

    # Tags (passed in from ingestion loop)
    if all_tags is None:
        all_tags = Counter()
    top_tags = all_tags.most_common(20)

    # Date analysis
    def _parse_dt(s: str) -> datetime | None:
        if not s:
            return None
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None

    pages_with_updated = [
        (p, _parse_dt(p.get("updatedAt", "")))
        for p in pages
    ]
    pages_with_updated = [(p, dt) for p, dt in pages_with_updated if dt]
    pages_with_updated.sort(key=lambda x: x[1], reverse=True)

    pages_with_created = [
        (p, _parse_dt(p.get("createdAt", "")))
        for p in pages
    ]
    pages_with_created = [(p, dt) for p, dt in pages_with_created if dt]
    pages_with_created.sort(key=lambda x: x[1])

    recently_updated = pages_with_updated[:10]
    newest_pages = sorted(pages_with_created, key=lambda x: x[1], reverse=True)[:10]
    oldest_pages = pages_with_created[:10]

    # --- Build text chunks ---
    chunks: list[dict] = []

    # Overview chunk
    overview_lines = [
        f"Wiki statistics and overview for {wiki_url}",
        f"Total public pages: {total_pages}",
        f"Number of contributors: {num_contributors}",
        f"Number of unique tags: {len(all_tags)}",
    ]
    if pages_with_updated:
        latest = pages_with_updated[0]
        overview_lines.append(
            f"Most recently updated page: \"{latest[0].get('title', '')}\" "
            f"on {latest[1].strftime('%Y-%m-%d')}"
        )
    if pages_with_created:
        oldest = pages_with_created[0]
        overview_lines.append(
            f"Oldest page: \"{oldest[0].get('title', '')}\" "
            f"created on {oldest[1].strftime('%Y-%m-%d')}"
        )

    overview_text = "\n".join(overview_lines)
    chunks.append({
        "text":        overview_text,
        "context":     f"[Wiki Metadata | Overview]\n\n{overview_text}",
        "page_title":  "Wiki Statistics",
        "page_path":   "_meta/overview",
        "page_url":    wiki_url,
        "section":     "Overview",
        "meta_type":   "overview",
    })

    # Contributors chunk
    contrib_lines = [
        f"Wiki contributors ({num_contributors} total):",
    ]
    for author, count in top_contributors:
        contrib_lines.append(f"  - {author}: {count} page{'s' if count != 1 else ''}")
    contrib_text = "\n".join(contrib_lines)
    chunks.append({
        "text":        contrib_text,
        "context":     f"[Wiki Metadata | Contributors]\n\n{contrib_text}",
        "page_title":  "Wiki Contributors",
        "page_path":   "_meta/contributors",
        "page_url":    wiki_url,
        "section":     "Contributors",
        "meta_type":   "contributors",
    })

    # Tags chunk
    if top_tags:
        tags_lines = [
            f"Wiki tags and topics ({len(all_tags)} unique tags):",
        ]
        for tag, count in top_tags:
            tags_lines.append(f"  - {tag}: {count} page{'s' if count != 1 else ''}")
        tags_text = "\n".join(tags_lines)
        chunks.append({
            "text":        tags_text,
            "context":     f"[Wiki Metadata | Tags and Topics]\n\n{tags_text}",
            "page_title":  "Wiki Tags and Topics",
            "page_path":   "_meta/tags",
            "page_url":    wiki_url,
            "section":     "Tags",
            "meta_type":   "tags",
        })

    # Recently updated chunk
    if recently_updated:
        recent_lines = ["Recently updated wiki pages:"]
        for p, dt in recently_updated:
            title = p.get("title") or p.get("path") or "Untitled"
            author = p.get("authorName") or ""
            line = f"  - \"{title}\" updated on {dt.strftime('%Y-%m-%d')}"
            if author:
                line += f" by {author}"
            recent_lines.append(line)
        recent_text = "\n".join(recent_lines)
        chunks.append({
            "text":        recent_text,
            "context":     f"[Wiki Metadata | Recently Updated]\n\n{recent_text}",
            "page_title":  "Recently Updated Pages",
            "page_path":   "_meta/recent",
            "page_url":    wiki_url,
            "section":     "Recently Updated",
            "meta_type":   "recent",
        })

    # Newest pages chunk
    if newest_pages:
        new_lines = ["Newest wiki pages (most recently created):"]
        for p, dt in newest_pages:
            title = p.get("title") or p.get("path") or "Untitled"
            author = p.get("authorName") or ""
            line = f"  - \"{title}\" created on {dt.strftime('%Y-%m-%d')}"
            if author:
                line += f" by {author}"
            new_lines.append(line)
        new_text = "\n".join(new_lines)
        chunks.append({
            "text":        new_text,
            "context":     f"[Wiki Metadata | Newest Pages]\n\n{new_text}",
            "page_title":  "Newest Wiki Pages",
            "page_path":   "_meta/newest",
            "page_url":    wiki_url,
            "section":     "Newest Pages",
            "meta_type":   "newest",
        })

    # Page listing chunk (all pages with paths, for "what pages exist" questions)
    listing_lines = [f"Complete list of all {total_pages} public wiki pages:"]
    for p in sorted(pages, key=lambda x: x.get("path", "")):
        title = p.get("title") or "Untitled"
        path = p.get("path") or ""
        listing_lines.append(f"  - {title} ({path})")
    listing_text = "\n".join(listing_lines)
    chunks.append({
        "text":        listing_text,
        "context":     f"[Wiki Metadata | Page Listing]\n\n{listing_text}",
        "page_title":  "All Wiki Pages",
        "page_path":   "_meta/pages",
        "page_url":    wiki_url,
        "section":     "Page Listing",
        "meta_type":   "page_listing",
    })

    return chunks


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
        collected_tags: Counter[str] = Counter()
        page_extra: dict[int, dict] = {}

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

            page_extra[page_id] = {
                "authorName": page.get("authorName") or "",
                "createdAt":  page.get("createdAt") or "",
            }

            for t in page.get("tags") or []:
                tag = t["tag"] if isinstance(t, dict) else t
                collected_tags[tag] += 1

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

    # --- Ingest wiki-level metadata ---
    logger.info("Building wiki metadata chunks…")
    meta_chunks = _build_wiki_metadata(pages, wiki_url, all_tags=collected_tags, page_extra=page_extra)
    if meta_chunks:
        meta_texts = [mc["text"] for mc in meta_chunks]
        try:
            meta_vectors = embedder.encode(meta_texts, batch_size=batch_size)
            meta_payloads = [
                {
                    "text":               mc["text"],
                    "context":            mc["context"],
                    "page_title":         mc["page_title"],
                    "page_path":          mc["page_path"],
                    "page_url":           mc["page_url"],
                    "section":            mc["section"],
                    "meta_type":          mc["meta_type"],
                    "tags":               [],
                    "updated_at":         datetime.utcnow().isoformat() + "Z",
                    "description":        "",
                    "section_breadcrumb": "",
                    "content_hash":       "",
                    "chunk_index":        0,
                    "total_chunks":       len(meta_chunks),
                }
                for mc in meta_chunks
            ]
            store.upsert_meta_chunks(meta_vectors, meta_payloads)
        except Exception as exc:
            logger.error("Failed to ingest wiki metadata: %s", exc)

    info = store.collection_info()
    logger.info(
        "Done.  pages_ingested=%d  unchanged=%d  skipped=%d  errors=%d  | "
        "metadata_chunks=%d  collection='%s'  total_chunks=%s",
        ok, unchanged, skipped, errors,
        len(meta_chunks) if meta_chunks else 0,
        info["name"], info["vectors_count"],
    )

    return errors == 0


if __name__ == "__main__":
    interval = int(_env("POLL_INTERVAL_SECONDS", "3600"))

    if interval <= 0:
        # One-shot mode (e.g. `docker compose run --rm wiki2rag`)
        if not run():
            sys.exit(2)
    else:
        logger.info("Running in continuous mode (poll every %d s).", interval)
        while True:
            try:
                run()
            except SystemExit:
                raise
            except Exception:
                logger.exception("Unexpected error during ingestion run.")

            logger.info("Sleeping %d s until next poll…", interval)
            time.sleep(interval)
