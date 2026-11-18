"""One sync run: two crawls, staging, sanitization, enrichment, indexing, reconcile.

    anonymous crawl ─┐                                   ┌─> hs_public (public tier only)
                     ├─> stage ─> sanitize ─> classify ──┤
    member crawl ────┘    (Postgres)   chunk, link, embed └─> hs_member (both tiers)

- Tier comes from which crawl can see a page: anything the anonymous crawl
  lists is ``public``; anything only the member crawl lists is ``member``.
  It is recomputed every run.
- Incremental runs refetch pages whose ``updatedAt`` or tier changed, or that
  were never indexed. Full runs (every RECONCILE_INTERVAL_SECONDS) refetch
  everything, stamp every document seen with the run ID, and delete this
  source's points with an older run ID, unless the run saw fewer than half
  the documents the previous full run saw.
- ``reindex`` rebuilds both collections from staging without touching the wiki.
"""

import hashlib
import logging
import threading
import uuid
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from wikijs2rag import __version__, attachments, content, sanitize
from wikijs2rag.chunker import chunk_markdown
from wikijs2rag.chunker import content_hash as text_hash
from wikijs2rag.config import Settings
from wikijs2rag.embedder import OpenAIEmbedder
from wikijs2rag.enrich import Enricher, authority, load_topics
from wikijs2rag.linker import Linker
from wikijs2rag.llm import LLM
from wikijs2rag.staging import SOURCE, Link, StagedAttachment, StagedPage, Staging
from wikijs2rag.store import StoreError, VectorStore, point_id
from wikijs2rag.wiki_client import JSON, WikiClient, WikiClientError

logger = logging.getLogger(__name__)

QDRANT_CONNECT_ATTEMPTS = 6
QDRANT_RETRY_SECONDS = 5.0
META_DOC_ID = "_meta"


class IngestError(Exception):
    """A failure that stops a whole run: a service unreachable, a credential
    that can see too much, or a canary where it must never be."""


@dataclass
class Counts:
    seen: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
    flagged: int = 0
    quarantined: int = 0
    skipped: int = 0
    errors: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass
class Listed:
    meta: JSON
    tier: str


def page_url(settings: Settings, locale: str, path: str) -> str:
    path = path.strip("/")
    if locale and locale != settings.wiki_default_locale:
        return f"{settings.wiki_url}/{locale}/{path}"
    return f"{settings.wiki_url}/{path}"


def is_excluded(settings: Settings, path: str) -> bool:
    path = path.strip("/")
    return any(path == p or path.startswith(p + "/") for p in settings.wiki_excluded_paths)


def list_tiers(settings: Settings, anon: WikiClient, member: WikiClient | None) -> dict[int, Listed]:
    """Every page either crawl can see, with its tier."""
    member_pages = member.list_pages() if member else []
    try:
        anon_pages = anon.list_pages()
    except WikiClientError as exc:
        if member is None:
            raise IngestError(f"Anonymous crawl can't list pages: {exc}") from exc
        logger.warning("Anonymous GraphQL listing failed (%s); checking each page as a guest instead.", exc)
        anon_pages = [
            p
            for p in member_pages
            if anon.guest_can_view(page_url(settings, p.get("locale") or "", p.get("path") or ""))
        ]
    listed: dict[int, Listed] = {}
    for p in anon_pages:
        listed[int(p["id"])] = Listed(p, "public")
    for p in member_pages:
        listed.setdefault(int(p["id"]), Listed(p, "member"))
    excluded = [pid for pid, item in listed.items() if is_excluded(settings, item.meta.get("path") or "")]
    for pid in excluded:
        del listed[pid]
    if excluded:
        logger.info("Skipped %d pages under excluded paths", len(excluded))
    return listed


def _check_canaries(settings: Settings, page: JSON, tier: str) -> None:
    body = f"{page.get('title', '')}\n{page.get('content', '')}\n{page.get('render', '')}"
    for canary in settings.canary_admin:
        if canary in body:
            raise IngestError(
                f"Admin canary found in page {page.get('id')} ({page.get('path')}) via the {tier} crawl: "
                "that crawl's credentials can read admin pages. Run stopped; the page was not staged."
            )
    if tier == "public":
        for canary in settings.canary_member:
            if canary in body:
                raise IngestError(
                    f"Member canary found in page {page.get('id')} ({page.get('path')}) via the anonymous crawl: "
                    "guests can read member pages. Run stopped; the page was not staged."
                )


def _tags(page: JSON, meta: JSON) -> list[str]:
    tags = page.get("tags") or meta.get("tags") or []
    return [str(t["tag"] if isinstance(t, dict) else t) for t in tags]


def _redact(text: str) -> str:
    return sanitize.scan_secrets(text)[0]


def build_staged(settings: Settings, page: JSON, meta: JSON, tier: str) -> tuple[StagedPage, list[Link], list[Any]]:
    """Normalize and sanitize a fetched page. Returns the staged page, its
    outbound links and the secret-scanner hits."""
    locale = page.get("locale") or meta.get("locale") or ""
    path = (page.get("path") or meta.get("path") or "").strip("/")
    content_type = (page.get("contentType") or "markdown").lower()
    raw = page.get("content") or ""
    render = page.get("render") or (raw if content_type == "html" else "")
    markdown = sanitize.scrub_network_identifiers(content.page_markdown(content_type, raw, render))
    markdown, hits = sanitize.scan_secrets(markdown)
    staged = StagedPage(
        source_id=str(page.get("id") or meta["id"]),
        locale=locale,
        path=path,
        parent_path=path.rsplit("/", 1)[0] if "/" in path else "",
        title=page.get("title") or meta.get("title") or path,
        description=page.get("description") or meta.get("description") or "",
        tags=_tags(page, meta),
        tier=tier,
        url=page_url(settings, locale, path),
        content_type=content_type,
        raw_content=_redact(raw),
        rendered_html=_redact(render),
        toc=page.get("toc") or "",
        text=markdown,
        content_hash=text_hash(markdown),
        flags=["secret-redacted"] if hits else [],
        created_at=page.get("createdAt") or meta.get("createdAt") or "",
        updated_at=page.get("updatedAt") or meta.get("updatedAt") or "",
    )
    if page.get("scraped"):
        staged.flags.append("html-scrape")
    return staged, content.extract_links(staged.rendered_html, settings.wiki_url), hits


def index_key(page: StagedPage, items: list[StagedAttachment]) -> str:
    parts = [page.content_hash, page.tier, *sorted(f"{a.url}:{a.tier}:{a.content_hash}" for a in items)]
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def _iso(value: str) -> str | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return None


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        staging: Staging,
        store: VectorStore,
        embedder: OpenAIEmbedder,
        llm: LLM | None,
        run_id: str,
    ) -> None:
        self.settings = settings
        self.staging = staging
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.run_id = run_id
        self.enricher = Enricher(llm, load_topics(settings.topics_file))
        self.linker = Linker(staging, llm)
        self.titles: dict[tuple[str, str], str] = {}

    # -- helpers ----------------------------------------------------------

    def collections(self) -> tuple[str, str]:
        q = self.settings.qdrant
        return q.public_collection, q.member_collection

    def _breadcrumb(self, page: StagedPage) -> list[str]:
        parts = page.path.split("/")
        crumbs = [self.titles.get((page.locale, "/".join(parts[: i + 1])), parts[i]) for i in range(len(parts) - 1)]
        return [*crumbs, page.title]

    def remove_doc(self, doc_id: str) -> None:
        for collection in self.collections():
            self.store.delete_doc(collection, SOURCE, doc_id)
        self.staging.clear_entity_documents([doc_id])

    def _gate(self, review_id: str, content_hash: str, page: StagedPage, title: str, text: str) -> str | None:
        """None if this member-tier text may be indexed, else why not."""
        status = self.staging.review_status(review_id, content_hash)
        if status == "approved":
            return None
        if status is not None:
            return f"in the review queue ({status})"
        reason = self.enricher.exclusion(title, text)  # LLMError propagates: fail closed
        if reason:
            self.staging.queue_for_review(review_id, content_hash, title, page.url, reason)
        return reason

    # -- indexing a staged page ---------------------------------------------

    def index_page(self, page: StagedPage) -> str:
        """Index one staged page into the collections. Returns "indexed" or
        "flagged". Raises on LLM, embedding or store failure."""
        items = self.staging.attachments_for(page.source_id)
        if page.tier == "member":
            reason = self._gate(page.source_id, page.content_hash, page, page.title, page.text)
            if reason:
                logger.warning("  Page %s held back: %s", page.source_id, reason)
                self.remove_doc(page.source_id)
                self.staging.mark_indexed(page.source_id, "")
                return "flagged"
        kept: list[StagedAttachment] = []
        for item in items:
            if not item.pages:
                continue
            if item.tier == "member":
                review_id = f"{page.source_id}:attachment:{item.url}"
                text = "\n\n".join(item.pages)
                reason = self._gate(
                    review_id, item.content_hash or text_hash(text), page, f"{page.title}: {item.url}", text
                )
                if reason:
                    logger.warning("  Attachment %s held back: %s", item.url, reason)
                    continue
            kept.append(item)

        kind = self.enricher.page_kind(page)
        if kind != page.page_kind:
            page.page_kind = kind
            self.staging.set_page_kind(page.source_id, kind)
        summary = self.enricher.summary(page)
        topics = self.enricher.page_topics(page, summary)
        page_links, roles = self.linker.link_page(kind, page.title, page.path, page.tags)
        crumbs = self._breadcrumb(page)
        now = datetime.now(UTC).isoformat()

        base: dict[str, Any] = {
            "source": SOURCE,
            "doc_id": page.source_id,
            "title": page.title,
            "tier": page.tier,
            "authority": authority(kind),
            "created_at": _iso(page.created_at),
            "updated_at": _iso(page.updated_at),
            "indexed_at": now,
            "stale_after": None,
            "superseded_by": None,
            "sync_run_id": self.run_id,
            "topics": topics,
            "lang": page.locale,
            "page": None,
            "page_kind": kind,
            "path": page.path,
            "locale": page.locale,
            "tags": page.tags,
            "description": page.description,
            "machine_generated": False,
        }

        def payload(chunk_kind: str, text: str, context: str, heading_path: str, url: str, **extra: Any) -> JSON:
            links = page_links.merged(self.linker.link(text))
            return {
                **base,
                "chunk_kind": chunk_kind,
                "text": text,
                "context": context,
                "heading_path": heading_path,
                "url": url,
                "content_hash": text_hash(text),
                **links.payload(),
                "section": self.linker.section(links),
                **extra,
            }

        payloads: list[JSON] = [
            payload(
                "summary",
                summary,
                f'Summary of the wiki page "{page.title}" ({kind}).',
                " > ".join(crumbs),
                page.url,
            )
        ]
        sizes = (
            self.settings.chunk_target_tokens,
            self.settings.chunk_max_tokens,
            self.settings.chunk_overlap_tokens,
        )
        for chunk in chunk_markdown(page.text, content.heading_anchors(page.rendered_html), *sizes):
            headings = chunk.headings
            if headings and headings[0].strip().lower() == page.title.strip().lower():
                headings = headings[1:]
            heading_path = " > ".join(crumbs + headings)
            url = f"{page.url}#{chunk.anchor}" if chunk.anchor else page.url
            context = self.enricher.context(page, summary, heading_path, chunk.text)
            payloads.append(payload("chunk", chunk.text, context, heading_path, url))

        for item in kept:
            name = item.url.rsplit("/", 1)[-1]
            title = f"{page.title}: {name}"
            if item.kind == "image":
                label = " ".join(x for x in (item.alt, item.caption) if x)
                text = f"[Image{': ' + label if label else ''}, machine-generated description]\n{item.pages[0]}"
                context = f'An image on the wiki page "{page.title}", described by a vision model.'
                payloads.append(
                    payload(
                        "image", text, context, " > ".join(crumbs), item.url, tier=item.tier, machine_generated=True
                    )
                )
                continue
            for number, page_text in enumerate(item.pages, 1):
                text = sanitize.scan_secrets(page_text)[0]
                for chunk in chunk_markdown(text, None, *sizes):
                    where = f"page {number}" if item.kind == "pdf" else ""
                    heading_path = " > ".join([*crumbs, name, *([where] if where else [])])
                    url = f"{item.url}#page={number}" if item.kind == "pdf" else item.url
                    context = f'From {name}{", " + where if where else ""}, attached to the wiki page "{page.title}".'
                    payloads.append(
                        payload(
                            "attachment",
                            chunk.text,
                            context,
                            heading_path,
                            url,
                            tier=item.tier,
                            title=title,
                            page=number if item.kind == "pdf" else None,
                        )
                    )

        for i, p in enumerate(payloads):
            p["chunk_index"] = i
            p["id"] = point_id(SOURCE, page.source_id, i)

        vectors = self.embedder.encode_documents(
            [f"{p['context']}\n\n{p['text']}" for p in payloads], batch_size=self.settings.batch_size
        )
        public, member = self.collections()
        if page.tier == "public":
            keep = [(v, p) for v, p in zip(vectors, payloads, strict=True) if p["tier"] == "public"]
            self.store.replace_doc(public, SOURCE, page.source_id, [v for v, _ in keep], [p for _, p in keep])
        else:
            self.store.delete_doc(public, SOURCE, page.source_id)
        self.store.replace_doc(member, SOURCE, page.source_id, vectors, payloads)
        self.staging.set_entity_documents(page.source_id, page.url, roles)
        self.staging.mark_indexed(page.source_id, index_key(page, items))
        logger.info("  Indexed %d points (%s, %s)", len(payloads), page.tier, kind)
        return "indexed"

    # -- wiki-level metadata ------------------------------------------------

    def index_metadata(self) -> None:
        pages = self.staging.live_pages()
        public, member = self.collections()
        for collection, tier_pages, tier in (
            (public, [p for p in pages if p.tier == "public"], "public"),
            (member, pages, "member"),
        ):
            payloads = build_wiki_metadata(tier_pages, self.settings.wiki_url, tier, self.run_id)
            vectors = self.embedder.encode_documents(
                [f"{p['context']}\n\n{p['text']}" for p in payloads], batch_size=self.settings.batch_size
            )
            self.store.replace_doc(collection, SOURCE, META_DOC_ID, vectors, payloads)


def _plural(count: int) -> str:
    return f"{count} page{'s' if count != 1 else ''}"


def build_wiki_metadata(pages: list[StagedPage], wiki_url: str, tier: str, run_id: str) -> list[JSON]:
    """Points describing the wiki itself (counts, tags, recent and new pages,
    a page listing), built only from pages of the collection's tier. No
    names of people: user lists are never ingested."""
    audience = "public" if tier == "public" else "public and member-only"
    dated = [(p, dt) for p in pages if (dt := _parse_dt(p.updated_at))]
    created = [(p, dt) for p in pages if (dt := _parse_dt(p.created_at))]
    tags: Counter[str] = Counter(t for p in pages for t in p.tags)
    texts: list[tuple[str, str]] = []

    overview = [
        f"Wiki statistics and overview for {wiki_url}",
        f"Total {audience} pages: {len(pages)}",
        f"Number of unique tags: {len(tags)}",
    ]
    if dated:
        p, dt = max(dated, key=lambda x: x[1])
        overview.append(f'Most recently updated page: "{p.title}" on {dt:%Y-%m-%d}')
    if created:
        p, dt = min(created, key=lambda x: x[1])
        overview.append(f'Oldest page: "{p.title}" created on {dt:%Y-%m-%d}')
    texts.append(("overview", "\n".join(overview)))
    if tags:
        lines = [f"Wiki tags and topics ({len(tags)} unique tags):"]
        lines += [f"  - {tag}: {_plural(n)}" for tag, n in tags.most_common(20)]
        texts.append(("tags", "\n".join(lines)))
    if dated:
        lines = ["Recently updated wiki pages:"]
        lines += [f'  - "{p.title}" updated on {dt:%Y-%m-%d}' for p, dt in sorted(dated, key=lambda x: x[1])[::-1][:10]]
        texts.append(("recent", "\n".join(lines)))
    if created:
        lines = ["Newest wiki pages (most recently created):"]
        lines += [
            f'  - "{p.title}" created on {dt:%Y-%m-%d}' for p, dt in sorted(created, key=lambda x: x[1])[::-1][:10]
        ]
        texts.append(("newest", "\n".join(lines)))
    listing = [f"Complete list of all {len(pages)} {audience} wiki pages:"]
    listing += [f"  - {p.title or 'Untitled'} ({p.path})" for p in sorted(pages, key=lambda x: (x.locale, x.path))]
    texts.append(("page_listing", "\n".join(listing)))

    now = datetime.now(UTC).isoformat()
    return [
        {
            "id": point_id(SOURCE, META_DOC_ID, i),
            "source": SOURCE,
            "doc_id": META_DOC_ID,
            "chunk_index": i,
            "chunk_kind": "meta",
            "meta_type": meta_type,
            "title": f"Wiki {meta_type.replace('_', ' ')}",
            "heading_path": "Wiki metadata",
            "text": text,
            "context": f"Statistics about the wiki itself ({meta_type.replace('_', ' ')}).",
            "url": wiki_url,
            "tier": tier,
            "authority": 4,
            "created_at": None,
            "updated_at": now,
            "indexed_at": now,
            "stale_after": None,
            "superseded_by": None,
            "content_hash": text_hash(text),
            "sync_run_id": run_id,
            "asset_ids": [],
            "location_ids": [],
            "work_ids": [],
            "event_ids": [],
            "training_ids": [],
            "topics": [],
            "section": None,
            "page": None,
            "lang": "",
        }
        for i, (meta_type, text) in enumerate(texts)
    ]


def _parse_dt(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def connect_store(settings: Settings, embedder: OpenAIEmbedder, stop: threading.Event) -> VectorStore:
    """Connect to Qdrant, retrying while it starts up, and check both collections."""
    q = settings.qdrant
    logger.info("Connecting to Qdrant at %s:%d …", q.host, q.port)
    for attempt in range(1, QDRANT_CONNECT_ATTEMPTS + 1):
        try:
            store = VectorStore(q.host, q.port, settings.embedding.model, embedder.dimension)
            store.ensure_collection(q.public_collection, "public")
            store.ensure_collection(q.member_collection, "member")
            return store
        except StoreError as exc:
            raise IngestError(str(exc)) from exc
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


def _new_run_id() -> str:
    return f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"


def full_run_due(settings: Settings, staging: Staging) -> bool:
    last = staging.last_full_finished_at()
    if last is None:
        return True
    return (datetime.now(UTC) - last).total_seconds() >= settings.reconcile_interval


def run(
    settings: Settings,
    stop: threading.Event | None = None,
    full: bool | None = None,
    reindex: bool = False,
) -> bool:
    """One sync run. Returns True if every page was processed without error.
    Raises :class:`IngestError` if the run can't proceed at all. *full* None
    means a full reconcile if one is due. Stops early, between pages, once
    *stop* is set."""
    stop = stop or threading.Event()
    staging = Staging(settings.database_url)
    try:
        with staging.source_lock() as locked:
            if not locked:
                logger.warning("Another wikijs2rag run holds the wiki lock; skipping this run.")
                return True
            if full is None:
                full = settings.force_reingest or full_run_due(settings, staging)
            kind = "reindex" if reindex else "full" if full else "incremental"
            run_id = _new_run_id()
            staging.start_run(run_id, kind)
            counts = Counts()
            try:
                ok = _run(settings, staging, run_id, kind, stop, counts)
            except Exception as exc:
                staging.finish_run(run_id, kind, False, counts.as_dict(), str(exc))
                raise
            staging.finish_run(run_id, kind, ok, counts.as_dict())
            logger.info("Run %s (%s) done: %s", run_id, kind, " ".join(f"{k}={v}" for k, v in counts.as_dict().items()))
            return ok
    finally:
        staging.close()


def _run(settings: Settings, staging: Staging, run_id: str, kind: str, stop: threading.Event, counts: Counts) -> bool:
    embedder = settings.embedding.embedder()
    store = connect_store(settings, embedder, stop)
    llm = (
        LLM(staging, settings.llm.base_url, settings.llm.model, settings.llm.api_key, settings.llm.vision_model)
        if settings.llm.model
        else None
    )
    pipeline = Pipeline(settings, staging, store, embedder, llm, run_id)
    try:
        if kind == "reindex":
            completed = _reindex(pipeline, stop, counts)
        else:
            completed = _crawl(pipeline, kind == "full", stop, counts)
        if not completed:
            return False
        pipeline.index_metadata()
        if kind != "incremental":
            _reconcile(pipeline, kind, counts)
    finally:
        if llm:
            llm.close()
    return counts.errors == 0


def _reindex(pipeline: Pipeline, stop: threading.Event, counts: Counts) -> bool:
    pages = pipeline.staging.live_pages()
    pipeline.titles = {(p.locale, p.path): p.title for p in pages}
    logger.info("wikijs2rag %s - rebuilding %d pages from staging", __version__, len(pages))
    for page in pages:
        if stop.is_set():
            return False
        counts.seen += 1
        _index(pipeline, page, counts, "updated")
    return True


def _index(pipeline: Pipeline, page: StagedPage, counts: Counts, label: str) -> None:
    try:
        result = pipeline.index_page(page)
    except Exception as exc:  # LLM, embeddings or Qdrant failure: retried next run
        logger.error("  Could not index page %s: %s", page.source_id, exc)
        counts.errors += 1
        return
    if result == "flagged":
        counts.flagged += 1
    elif label == "added":
        counts.added += 1
    else:
        counts.updated += 1


def _crawl(pipeline: Pipeline, full: bool, stop: threading.Event, counts: Counts) -> bool:
    settings, staging = pipeline.settings, pipeline.staging
    key = settings.wiki_member_api_key
    with WikiClient(settings.wiki_url) as anon, WikiClient(settings.wiki_url, key) if key else nullcontext() as member:
        for client in (anon, member):
            if client is not None and client.has_admin_rights():
                raise IngestError(
                    f"The {'member' if client.authenticated else 'anonymous'} credentials have admin rights "
                    "(they can list API keys). Use a key scoped to the members group; never a full-access key."
                )
        logger.info("wikijs2rag %s - %s crawl of %s", __version__, "full" if full else "incremental", settings.wiki_url)
        try:
            listed = list_tiers(settings, anon, member)
        except WikiClientError as exc:
            raise IngestError(f"Failed to list pages: {exc}") from exc
        pipeline.titles = {
            (item.meta.get("locale") or "", (item.meta.get("path") or "").strip("/")): item.meta.get("title") or ""
            for item in listed.values()
        }
        states = staging.page_states()
        seen: list[str] = []
        total = len(listed)
        for n, (page_id, item) in enumerate(sorted(listed.items()), 1):
            if stop.is_set():
                logger.info("Stop requested; ending run after %d of %d pages.", n - 1, total)
                staging.mark_seen(seen, pipeline.run_id)
                return False
            sid = str(page_id)
            seen.append(sid)
            counts.seen += 1
            state = states.get(sid)
            if state and state[1] == "public" and item.tier == "member":
                # Restricted at the source: out of the public collection now,
                # whatever happens to the rest of this page's processing.
                pipeline.store.delete_doc(settings.qdrant.public_collection, SOURCE, sid)
            changed = (
                settings.force_reingest
                or full
                or state is None
                or state[3]  # deleted
                or state[0] != (item.meta.get("updatedAt") or "")
                or state[1] != item.tier
                or not state[2]  # never indexed, or held back
            )
            if not changed:
                counts.unchanged += 1
                continue
            logger.info("[%d/%d] %s page %s: %s", n, total, item.tier, sid, item.meta.get("path"))
            # Member-tier pages only come from the member crawl, so member is set.
            client = anon if item.tier == "public" or member is None else member
            try:
                page = client.get_page(page_id, meta=item.meta)
            except WikiClientError as exc:
                logger.error("  Could not fetch page %s: %s", sid, exc)
                counts.errors += 1
                continue
            if not page or not ((page.get("content") or "").strip() or (page.get("render") or "").strip()):
                counts.skipped += 1
                pipeline.remove_doc(sid)
                continue
            _check_canaries(settings, page, item.tier)
            staged, links, hits = build_staged(settings, page, item.meta, item.tier)
            crawls: list[tuple[WikiClient, str]] = [(client, item.tier)]
            if item.tier == "public" and member is not None:
                crawls.append((member, "member"))
            files = attachments.collect(
                crawls, staged.rendered_html, settings.wiki_url, settings.attachment_max_bytes, pipeline.llm
            )
            for f in files:
                _check_canaries(settings, {"id": sid, "path": f.url, "content": "\n".join(f.pages)}, f.tier)
                cleaned = [sanitize.scan_secrets(t) for t in f.pages]
                f.pages = [c[0] for c in cleaned]
                for _text, file_hits in cleaned:
                    for hit in file_hits:
                        staging.quarantine_hit(sid, f.url, hit.rule, hit.line, f.content_hash)
                        counts.quarantined += 1
            for hit in hits:
                staging.quarantine_hit(sid, "page", hit.rule, hit.line, staged.content_hash)
                counts.quarantined += 1
            if hits:
                logger.warning("  Redacted %d secret(s) in page %s; see the quarantine report", len(hits), sid)
            previous = staging.get_page(sid)
            staging.stage_page(staged, pipeline.run_id, links)
            staging.stage_attachments(sid, files)
            staged.indexed_key = previous.indexed_key if previous else ""
            if previous and previous.indexed_key == index_key(staged, files) and not settings.force_reingest:
                counts.unchanged += 1
                continue
            _index(pipeline, staged, counts, "added" if state is None else "updated")
            if settings.page_delay > 0:
                stop.wait(settings.page_delay)
        staging.mark_seen(seen, pipeline.run_id)
    return True


def _reconcile(pipeline: Pipeline, kind: str, counts: Counts) -> None:
    """Delete this source's points and staged pages that this run didn't see."""
    staging, store = pipeline.staging, pipeline.store
    previous = staging.last_full_docs_seen()
    if kind == "full" and previous and counts.seen < previous / 2:
        logger.error(
            "ALERT: this full run saw %d pages but the previous one saw %d. Not deleting anything; "
            "check the wiki and its credentials.",
            counts.seen,
            previous,
        )
        counts.errors += 1
        return
    gone = staging.mark_unseen_deleted(pipeline.run_id) if kind == "full" else []
    staging.clear_entity_documents(gone)
    live = [p for p in staging.live_pages() if p.indexed_key]
    public, member = pipeline.collections()
    stamped = 0
    for collection, docs in ((public, [p for p in live if p.tier == "public"]), (member, live)):
        for page in docs:
            store.stamp_doc(collection, SOURCE, page.source_id, pipeline.run_id)
            stamped += 1
        counts.deleted += store.delete_stale(collection, SOURCE, pipeline.run_id)
    logger.info(
        "Reconcile: stamped %d docs, deleted %d points, %d pages gone from the wiki", stamped, counts.deleted, len(gone)
    )
