"""Ingestion: fetch public Wiki.js pages, chunk, embed and store them in Qdrant.

Features:
  - Incremental ingestion: skips pages whose updatedAt hasn't changed
  - Enriched context per chunk for LLM grounding
  - Parent/prev/next chunk references for small-to-big retrieval
  - Content hash per chunk for deduplication
  - Wiki-level metadata chunks (page count, contributors, tags, ...)
"""

import logging
import threading
import uuid
from collections import Counter
from datetime import UTC, datetime

from wikijs2rag import __version__
from wikijs2rag.chunker import Chunk, chunk_page
from wikijs2rag.config import Settings
from wikijs2rag.embedder import OllamaEmbedder
from wikijs2rag.store import VectorStore
from wikijs2rag.wiki_client import JSON, WikiClient, WikiClientError

logger = logging.getLogger(__name__)

QDRANT_CONNECT_ATTEMPTS = 6
QDRANT_RETRY_SECONDS = 5.0


class IngestError(Exception):
    """A failure that stops a whole ingestion run (Qdrant or the wiki unreachable)."""


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


def build_context(chunk: Chunk, page_title: str, page_description: str) -> str:
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
        return f"[{' | '.join(parts)}]\n\n{chunk.text}"
    return chunk.text


# ---------------------------------------------------------------------------
# Payload builder with parent/prev/next references
# ---------------------------------------------------------------------------


def build_payloads(chunks: list[Chunk], page: JSON, page_url: str, parent_chunk_size: int) -> list[JSON]:
    """
    Build payload dicts for each chunk.

    Assigns UUIDs per chunk and cross-references:
      - prev_chunk_id / next_chunk_id  for sequential traversal
      - parent_chunk_index             index of the "parent" chunk in a
                                       coarser-grained view (every Nth chunk)
    """
    page_title = page.get("title") or ""
    page_description = page.get("description") or ""
    tags = [t["tag"] for t in (page.get("tags") or [])]
    updated_at = page.get("updatedAt") or ""

    chunk_ids = [str(uuid.uuid4()) for _ in chunks]

    payloads: list[JSON] = []
    for i, chunk in enumerate(chunks):
        payloads.append(
            {
                "chunk_id": chunk_ids[i],
                "text": chunk.text,
                "context": build_context(chunk, page_title, page_description),
                "chunk_index": chunk.chunk_index,
                "section": chunk.section,
                "section_breadcrumb": chunk.section_breadcrumb,
                "content_hash": chunk.content_hash,
                "page_path": page["path"],
                "page_title": page_title,
                "page_url": page_url,
                "description": page_description,
                "tags": tags,
                "updated_at": updated_at,
                "prev_chunk_id": chunk_ids[i - 1] if i > 0 else None,
                "next_chunk_id": chunk_ids[i + 1] if i < len(chunks) - 1 else None,
                "parent_chunk_index": (i // parent_chunk_size) * parent_chunk_size,
                "total_chunks": len(chunks),
            }
        )
    return payloads


# ---------------------------------------------------------------------------
# Wiki metadata generation
# ---------------------------------------------------------------------------


def _parse_dt(s: str) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _meta_chunk(text: str, label: str, title: str, path: str, wiki_url: str, section: str, meta_type: str) -> JSON:
    return {
        "text": text,
        "context": f"[Wiki Metadata | {label}]\n\n{text}",
        "page_title": title,
        "page_path": path,
        "page_url": wiki_url,
        "section": section,
        "meta_type": meta_type,
    }


def _plural(count: int) -> str:
    return f"{count} page{'s' if count != 1 else ''}"


def build_wiki_metadata(
    pages: list[JSON],
    wiki_url: str,
    all_tags: Counter[str] | None = None,
    page_extra: dict[int, JSON] | None = None,
) -> list[JSON]:
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
    page_extra = page_extra or {}
    all_tags = all_tags or Counter()

    # Merge per-page details into the list entries
    for p in pages:
        extra = page_extra.get(p["id"], {})
        for key in ("authorName", "createdAt"):
            if key not in p and key in extra:
                p[key] = extra[key]

    total_pages = len(pages)

    authors = Counter(str(p["authorName"]) for p in pages if p.get("authorName"))
    num_contributors = len(authors)
    top_tags = all_tags.most_common(20)

    pages_with_updated = [(p, dt) for p in pages if (dt := _parse_dt(p.get("updatedAt", "")))]
    pages_with_updated.sort(key=lambda x: x[1], reverse=True)
    pages_with_created = [(p, dt) for p in pages if (dt := _parse_dt(p.get("createdAt", "")))]
    pages_with_created.sort(key=lambda x: x[1])

    recently_updated = pages_with_updated[:10]
    newest_pages = sorted(pages_with_created, key=lambda x: x[1], reverse=True)[:10]

    chunks: list[JSON] = []

    overview_lines = [
        f"Wiki statistics and overview for {wiki_url}",
        f"Total public pages: {total_pages}",
        f"Number of contributors: {num_contributors}",
        f"Number of unique tags: {len(all_tags)}",
    ]
    if pages_with_updated:
        page, dt = pages_with_updated[0]
        overview_lines.append(f'Most recently updated page: "{page.get("title", "")}" on {dt:%Y-%m-%d}')
    if pages_with_created:
        page, dt = pages_with_created[0]
        overview_lines.append(f'Oldest page: "{page.get("title", "")}" created on {dt:%Y-%m-%d}')
    chunks.append(
        _meta_chunk(
            "\n".join(overview_lines), "Overview", "Wiki Statistics", "_meta/overview", wiki_url, "Overview", "overview"
        )
    )

    contrib_lines = [f"Wiki contributors ({num_contributors} total):"]
    contrib_lines += [f"  - {author}: {_plural(count)}" for author, count in authors.most_common(10)]
    chunks.append(
        _meta_chunk(
            "\n".join(contrib_lines),
            "Contributors",
            "Wiki Contributors",
            "_meta/contributors",
            wiki_url,
            "Contributors",
            "contributors",
        )
    )

    if top_tags:
        tags_lines = [f"Wiki tags and topics ({len(all_tags)} unique tags):"]
        tags_lines += [f"  - {tag}: {_plural(count)}" for tag, count in top_tags]
        chunks.append(
            _meta_chunk(
                "\n".join(tags_lines), "Tags and Topics", "Wiki Tags and Topics", "_meta/tags", wiki_url, "Tags", "tags"
            )
        )

    def _dated_lines(header: str, verb: str, entries: list[tuple[JSON, datetime]]) -> str:
        lines = [header]
        for p, dt in entries:
            title = p.get("title") or p.get("path") or "Untitled"
            line = f'  - "{title}" {verb} on {dt:%Y-%m-%d}'
            if author := p.get("authorName"):
                line += f" by {author}"
            lines.append(line)
        return "\n".join(lines)

    if recently_updated:
        text = _dated_lines("Recently updated wiki pages:", "updated", recently_updated)
        chunks.append(
            _meta_chunk(
                text,
                "Recently Updated",
                "Recently Updated Pages",
                "_meta/recent",
                wiki_url,
                "Recently Updated",
                "recent",
            )
        )

    if newest_pages:
        text = _dated_lines("Newest wiki pages (most recently created):", "created", newest_pages)
        chunks.append(
            _meta_chunk(text, "Newest Pages", "Newest Wiki Pages", "_meta/newest", wiki_url, "Newest Pages", "newest")
        )

    # Page listing chunk (all pages with paths, for "what pages exist" questions)
    listing_lines = [f"Complete list of all {total_pages} public wiki pages:"]
    for p in sorted(pages, key=lambda x: str(x.get("path", ""))):
        listing_lines.append(f"  - {p.get('title') or 'Untitled'} ({p.get('path') or ''})")
    chunks.append(
        _meta_chunk(
            "\n".join(listing_lines),
            "Page Listing",
            "All Wiki Pages",
            "_meta/pages",
            wiki_url,
            "Page Listing",
            "page_listing",
        )
    )

    return chunks


# ---------------------------------------------------------------------------
# Main ingestion run
# ---------------------------------------------------------------------------


def connect_store(settings: Settings, dimension: int, stop: threading.Event) -> VectorStore:
    """Connect to Qdrant, retrying while it starts up."""
    q = settings.qdrant
    logger.info("Connecting to Qdrant at %s:%d …", q.host, q.port)
    for attempt in range(1, QDRANT_CONNECT_ATTEMPTS + 1):
        try:
            return VectorStore(q.host, q.port, q.collection, dimension)
        except Exception as exc:  # qdrant-client raises several transport errors
            if attempt == QDRANT_CONNECT_ATTEMPTS or stop.is_set():
                raise IngestError(f"Cannot connect to Qdrant: {exc}") from exc
            logger.warning(
                "Qdrant not ready (attempt %d/%d): %s - retrying in %.0f s…",
                attempt,
                QDRANT_CONNECT_ATTEMPTS,
                exc,
                QDRANT_RETRY_SECONDS,
            )
            stop.wait(QDRANT_RETRY_SECONDS)
    raise AssertionError("unreachable")  # pragma: no cover


def run(settings: Settings, stop: threading.Event | None = None) -> bool:
    """
    One ingestion pass. Returns True if every page was processed without
    error. Raises :class:`IngestError` if the run can't proceed at all.
    Stops early, between pages, once *stop* is set.
    """
    stop = stop or threading.Event()
    e = settings.embedding
    embedder = OllamaEmbedder(e.model, base_url=e.ollama_url, context_length=e.context_length, api_key=e.api_key)
    store = connect_store(settings, embedder.dimension, stop)
    wiki_url = settings.wiki_url

    with WikiClient(wiki_url, settings.wiki_api_key) as wiki:
        logger.info("wikijs2rag %s  -  fetching public page list from %s …", __version__, wiki_url)
        try:
            pages = wiki.list_public_pages()
        except WikiClientError as exc:
            raise IngestError(f"Failed to list pages: {exc}") from exc

        if not pages:
            logger.warning("No public pages found - nothing to do.")
            return True

        total = len(pages)
        ok = skipped = unchanged = errors = 0
        collected_tags: Counter[str] = Counter()
        page_extra: dict[int, JSON] = {}

        for i, meta in enumerate(pages, 1):
            if stop.is_set():
                logger.info("Stop requested; ending run after %d of %d pages.", i - 1, total)
                return errors == 0

            page_id: int = meta["id"]
            title = meta.get("title") or meta.get("path") or str(page_id)
            logger.info("[%d/%d] Processing page %d: %s", i, total, page_id, title)

            # --- Incremental: skip if updatedAt hasn't changed ---
            if not settings.force_reingest:
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
                "createdAt": page.get("createdAt") or "",
            }
            for t in page.get("tags") or []:
                collected_tags[t["tag"] if isinstance(t, dict) else t] += 1

            chunks = chunk_page(
                page["content"],
                content_type=(page.get("contentType") or "markdown").lower(),
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            if not chunks:
                logger.debug("  Page %d produced no chunks, skipping.", page_id)
                skipped += 1
                continue

            try:
                vectors = embedder.encode([c.text for c in chunks], batch_size=settings.batch_size)
            except Exception as exc:  # httpx or unexpected-response errors; keep going
                logger.error("  Embedding failed for page %d: %s", page_id, exc)
                errors += 1
                continue

            page_url = f"{wiki_url}/{page['path'].lstrip('/')}"
            payloads = build_payloads(chunks, page, page_url, settings.parent_ratio)

            try:
                store.upsert_page_chunks(page_id, vectors, payloads)
            except Exception as exc:  # qdrant-client transport errors; keep going
                logger.error("  Store failed for page %d: %s", page_id, exc)
                errors += 1
                continue

            ok += 1
            if settings.page_delay > 0:
                stop.wait(settings.page_delay)

    # --- Ingest wiki-level metadata ---
    logger.info("Building wiki metadata chunks…")
    meta_chunks = build_wiki_metadata(pages, wiki_url, all_tags=collected_tags, page_extra=page_extra)
    try:
        meta_vectors = embedder.encode([mc["text"] for mc in meta_chunks], batch_size=settings.batch_size)
        now = datetime.now(UTC).isoformat()
        meta_payloads = [
            {
                **mc,
                "tags": [],
                "updated_at": now,
                "description": "",
                "section_breadcrumb": "",
                "content_hash": "",
                "chunk_index": 0,
                "total_chunks": len(meta_chunks),
            }
            for mc in meta_chunks
        ]
        store.upsert_meta_chunks(meta_vectors, meta_payloads)
    except Exception:  # metadata is best-effort; page data is already stored
        logger.exception("Failed to ingest wiki metadata")

    info = store.collection_info()
    logger.info(
        "Done.  pages_ingested=%d  unchanged=%d  skipped=%d  errors=%d  | "
        "metadata_chunks=%d  collection='%s'  total_chunks=%s",
        ok,
        unchanged,
        skipped,
        errors,
        len(meta_chunks),
        info["name"],
        info["vectors_count"],
    )
    return errors == 0
