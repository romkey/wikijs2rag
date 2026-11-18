"""Staging store and entity registry (Postgres in production, SQLite for tests).

Every fetched page is kept here in normalized form, so the Qdrant collections
can be rebuilt (re-chunked, re-embedded) without refetching the wiki. The same
database holds:

- ``links``: outbound links per page, internal and external.
- ``attachments``: files and images a page links to, with extracted text.
- ``llm_cache``: every ingest-time LLM result, keyed by a hash of its input.
- ``review_queue``: pages the exclusion classifier flagged; humans approve or
  reject them by setting ``status``.
- ``quarantine``: secret-scanner hits (never the secret itself).
- ``sync_status`` / ``sync_runs``: one status row per source, plus history.
- ``entities`` / ``entity_aliases`` / ``entity_documents``: the shared entity
  registry used to correlate sources. Humans edit aliases; this ingester
  reads them and writes which wiki page documents which entity.

Tables are created if missing. The schema is shared with the other source
ingesters, so add columns rather than rename them.
"""

import json
import logging
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Engine,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    and_,
    create_engine,
    delete,
    func,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)

SOURCE = "wiki"

metadata = MetaData()

documents = Table(
    "documents",
    metadata,
    Column("source", String(32), primary_key=True),
    Column("source_id", String(255), primary_key=True),
    Column("locale", String(16), nullable=False, default=""),
    Column("path", Text, nullable=False, default=""),
    Column("parent_path", Text, nullable=False, default=""),
    Column("title", Text, nullable=False, default=""),
    Column("description", Text, nullable=False, default=""),
    Column("tags", Text, nullable=False, default="[]"),  # JSON list
    Column("url", Text, nullable=False, default=""),
    Column("content_type", String(32), nullable=False, default=""),
    Column("raw_content", Text, nullable=False, default=""),
    Column("rendered_html", Text, nullable=False, default=""),
    Column("toc", Text, nullable=False, default=""),
    Column("text", Text, nullable=False, default=""),  # clean, sanitized markdown
    Column("content_hash", String(64), nullable=False, default=""),
    Column("tier", String(16), nullable=False),
    Column("flags", Text, nullable=False, default="[]"),  # JSON list
    Column("page_kind", String(32), nullable=False, default=""),
    Column("source_created_at", String(64), nullable=False, default=""),
    Column("source_updated_at", String(64), nullable=False, default=""),
    Column("fetched_at", DateTime(timezone=True), nullable=False),
    Column("last_seen_run", String(64), nullable=False, default=""),
    Column("indexed_key", String(64), nullable=False, default=""),  # hash+tier last indexed
    Column("deleted_at", DateTime(timezone=True), nullable=True),
)

links = Table(
    "links",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("source", String(32), nullable=False),
    Column("source_id", String(255), nullable=False, index=True),
    Column("kind", String(16), nullable=False),  # internal | external
    Column("target", Text, nullable=False),  # internal: locale/path; external: URL
    Column("anchor", Text, nullable=False, default=""),
    Column("label", Text, nullable=False, default=""),
)

attachments = Table(
    "attachments",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("source", String(32), nullable=False),
    Column("source_id", String(255), nullable=False, index=True),
    Column("url", Text, nullable=False),
    Column("kind", String(16), nullable=False),  # pdf | text | image
    Column("tier", String(16), nullable=False),
    Column("alt", Text, nullable=False, default=""),
    Column("caption", Text, nullable=False, default=""),
    Column("pages", Text, nullable=False, default="[]"),  # JSON list of page texts
    Column("machine_generated", Boolean, nullable=False, default=False),
    Column("content_hash", String(64), nullable=False, default=""),
    Column("fetched_at", DateTime(timezone=True), nullable=False),
)

llm_cache = Table(
    "llm_cache",
    metadata,
    Column("key", String(64), primary_key=True),
    Column("task", String(32), nullable=False),
    Column("model", Text, nullable=False),
    Column("value", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

review_queue = Table(
    "review_queue",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("source", String(32), nullable=False),
    Column("source_id", String(255), nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("title", Text, nullable=False, default=""),
    Column("url", Text, nullable=False, default=""),
    Column("reason", Text, nullable=False),
    Column("status", String(16), nullable=False, default="pending"),  # pending | approved | rejected
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("decided_at", DateTime(timezone=True), nullable=True),
    UniqueConstraint("source", "source_id", "content_hash"),
)

quarantine = Table(
    "quarantine",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("source", String(32), nullable=False),
    Column("source_id", String(255), nullable=False),
    Column("location", Text, nullable=False),  # page, or attachment URL
    Column("rule", String(64), nullable=False),
    Column("line", Integer, nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("source", "source_id", "location", "rule", "line", "content_hash"),
)

sync_status = Table(
    "sync_status",
    metadata,
    Column("source", String(32), primary_key=True),
    Column("last_run_id", String(64), nullable=False),
    Column("last_kind", String(16), nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    Column("ok", Boolean, nullable=False, default=False),
    Column("counts", Text, nullable=False, default="{}"),  # JSON
    Column("error", Text, nullable=False, default=""),
    Column("last_full_run_id", String(64), nullable=False, default=""),
    Column("last_full_finished_at", DateTime(timezone=True), nullable=True),
    Column("last_full_docs_seen", Integer, nullable=False, default=0),
)

sync_runs = Table(
    "sync_runs",
    metadata,
    Column("run_id", String(64), primary_key=True),
    Column("source", String(32), nullable=False),
    Column("kind", String(16), nullable=False),  # incremental | full | reindex
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    Column("ok", Boolean, nullable=False, default=False),
    Column("counts", Text, nullable=False, default="{}"),
    Column("error", Text, nullable=False, default=""),
)

entities = Table(
    "entities",
    metadata,
    Column("entity_type", String(32), primary_key=True),  # asset | location | work | event | series | training
    Column("entity_id", String(255), primary_key=True),
    Column("name", Text, nullable=False, default=""),
    Column("attributes", Text, nullable=False, default="{}"),  # JSON, e.g. {"section": "2"} for locations
)

entity_aliases = Table(
    "entity_aliases",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("entity_type", String(32), nullable=False),
    Column("entity_id", String(255), nullable=False),
    Column("alias", Text, nullable=False),
)

entity_documents = Table(
    "entity_documents",
    metadata,
    Column("entity_type", String(32), primary_key=True),
    Column("entity_id", String(255), primary_key=True),
    Column("role", String(32), primary_key=True),  # how-to-use | training | build-instructions | area
    Column("source", String(32), primary_key=True),
    Column("doc_id", String(255), primary_key=True),
    Column("url", Text, nullable=False, default=""),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)


def now() -> datetime:
    return datetime.now(UTC)


@dataclass
class StagedPage:
    """A page as staged: what the indexer needs, with no wiki API in sight."""

    source_id: str
    locale: str
    path: str
    title: str
    tier: str
    url: str
    text: str
    content_hash: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    parent_path: str = ""
    content_type: str = "markdown"
    raw_content: str = ""
    rendered_html: str = ""
    toc: str = ""
    flags: list[str] = field(default_factory=list)
    page_kind: str = ""
    created_at: str = ""
    updated_at: str = ""
    indexed_key: str = ""


@dataclass
class StagedAttachment:
    url: str
    kind: str  # pdf | text | image
    tier: str
    alt: str = ""
    caption: str = ""
    pages: list[str] = field(default_factory=list)
    machine_generated: bool = False
    content_hash: str = ""


@dataclass
class Link:
    kind: str  # internal | external
    target: str
    anchor: str = ""
    label: str = ""


@dataclass
class Entity:
    entity_type: str
    entity_id: str
    attributes: dict[str, Any]


class Staging:
    def __init__(self, url: str) -> None:
        self.url = url
        self.engine: Engine = create_engine(url, future=True)
        metadata.create_all(self.engine)

    def close(self) -> None:
        self.engine.dispose()

    def __enter__(self) -> Staging:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def is_postgres(self) -> bool:
        return self.engine.dialect.name == "postgresql"

    @contextmanager
    def begin(self) -> Iterator[Connection]:
        with self.engine.begin() as conn:
            yield conn

    # -- locking --------------------------------------------------------

    @contextmanager
    def source_lock(self, source: str = SOURCE) -> Iterator[bool]:
        """Hold a per-source lock for the duration of a run. Yields False if
        another process holds it. Postgres advisory lock; SQLite has none."""
        if not self.is_postgres:
            yield True
            return
        key = zlib.crc32(f"wikijs2rag:{source}".encode())
        with self.engine.connect() as conn:
            got = bool(conn.execute(text("SELECT pg_try_advisory_lock(CAST(:k AS bigint))"), {"k": key}).scalar())
            try:
                yield got
            finally:
                if got:
                    conn.execute(text("SELECT pg_advisory_unlock(CAST(:k AS bigint))"), {"k": key})
                    conn.commit()

    # -- documents ------------------------------------------------------

    def get_page(self, source_id: str) -> StagedPage | None:
        with self.begin() as conn:
            row = conn.execute(
                select(documents).where(and_(documents.c.source == SOURCE, documents.c.source_id == source_id))
            ).first()
        return _page_from_row(row._mapping) if row else None

    def page_states(self) -> dict[str, tuple[str, str, str, bool]]:
        """source_id -> (source_updated_at, tier, indexed_key, deleted) for every staged page."""
        with self.begin() as conn:
            rows = conn.execute(
                select(
                    documents.c.source_id,
                    documents.c.source_updated_at,
                    documents.c.tier,
                    documents.c.indexed_key,
                    documents.c.deleted_at,
                ).where(documents.c.source == SOURCE)
            ).all()
        return {r[0]: (r[1], r[2], r[3], r[4] is not None) for r in rows}

    def live_pages(self) -> list[StagedPage]:
        with self.begin() as conn:
            rows = conn.execute(
                select(documents).where(and_(documents.c.source == SOURCE, documents.c.deleted_at.is_(None)))
            ).all()
        return [_page_from_row(r._mapping) for r in rows]

    def stage_page(self, page: StagedPage, run_id: str, page_links: list[Link]) -> None:
        values = {
            "locale": page.locale,
            "path": page.path,
            "parent_path": page.parent_path,
            "title": page.title,
            "description": page.description,
            "tags": json.dumps(page.tags),
            "url": page.url,
            "content_type": page.content_type,
            "raw_content": page.raw_content,
            "rendered_html": page.rendered_html,
            "toc": page.toc,
            "text": page.text,
            "content_hash": page.content_hash,
            "tier": page.tier,
            "flags": json.dumps(page.flags),
            "page_kind": page.page_kind,
            "source_created_at": page.created_at,
            "source_updated_at": page.updated_at,
            "fetched_at": now(),
            "last_seen_run": run_id,
            "deleted_at": None,
        }
        key = and_(documents.c.source == SOURCE, documents.c.source_id == page.source_id)
        with self.begin() as conn:
            exists = conn.execute(select(documents.c.source_id).where(key)).first()
            if exists:
                conn.execute(update(documents).where(key).values(**values))
            else:
                conn.execute(insert(documents).values(source=SOURCE, source_id=page.source_id, **values))
            conn.execute(delete(links).where(and_(links.c.source == SOURCE, links.c.source_id == page.source_id)))
            if page_links:
                conn.execute(
                    insert(links),
                    [
                        {
                            "source": SOURCE,
                            "source_id": page.source_id,
                            "kind": link.kind,
                            "target": link.target,
                            "anchor": link.anchor,
                            "label": link.label,
                        }
                        for link in page_links
                    ],
                )

    def set_page_kind(self, source_id: str, page_kind: str) -> None:
        self._update_doc(source_id, page_kind=page_kind)

    def mark_seen(self, source_ids: list[str], run_id: str) -> None:
        if not source_ids:
            return
        with self.begin() as conn:
            for i in range(0, len(source_ids), 500):
                conn.execute(
                    update(documents)
                    .where(and_(documents.c.source == SOURCE, documents.c.source_id.in_(source_ids[i : i + 500])))
                    .values(last_seen_run=run_id, deleted_at=None)
                )

    def mark_indexed(self, source_id: str, indexed_key: str) -> None:
        self._update_doc(source_id, indexed_key=indexed_key)

    def mark_unseen_deleted(self, run_id: str) -> list[str]:
        """Mark staged pages not seen in *run_id* as deleted; return their IDs."""
        with self.begin() as conn:
            gone = [
                r[0]
                for r in conn.execute(
                    select(documents.c.source_id).where(
                        and_(
                            documents.c.source == SOURCE,
                            documents.c.last_seen_run != run_id,
                            documents.c.deleted_at.is_(None),
                        )
                    )
                ).all()
            ]
            if gone:
                conn.execute(
                    update(documents)
                    .where(and_(documents.c.source == SOURCE, documents.c.source_id.in_(gone)))
                    .values(deleted_at=now(), indexed_key="")
                )
        return gone

    def _update_doc(self, source_id: str, **values: Any) -> None:
        with self.begin() as conn:
            conn.execute(
                update(documents)
                .where(and_(documents.c.source == SOURCE, documents.c.source_id == source_id))
                .values(**values)
            )

    def links_for(self, source_id: str) -> list[Link]:
        with self.begin() as conn:
            rows = conn.execute(
                select(links.c.kind, links.c.target, links.c.anchor, links.c.label).where(
                    and_(links.c.source == SOURCE, links.c.source_id == source_id)
                )
            ).all()
        return [Link(*r) for r in rows]

    def all_internal_links(self) -> list[tuple[str, str, str]]:
        """(source_id, target, label) for every internal link of a live page."""
        with self.begin() as conn:
            rows = conn.execute(
                select(links.c.source_id, links.c.target, links.c.label)
                .join(
                    documents,
                    and_(documents.c.source == links.c.source, documents.c.source_id == links.c.source_id),
                )
                .where(and_(links.c.source == SOURCE, links.c.kind == "internal", documents.c.deleted_at.is_(None)))
            ).all()
        return [(r[0], r[1], r[2]) for r in rows]

    # -- attachments ----------------------------------------------------

    def stage_attachments(self, source_id: str, items: list[StagedAttachment]) -> None:
        with self.begin() as conn:
            conn.execute(
                delete(attachments).where(and_(attachments.c.source == SOURCE, attachments.c.source_id == source_id))
            )
            if items:
                conn.execute(
                    insert(attachments),
                    [
                        {
                            "source": SOURCE,
                            "source_id": source_id,
                            "url": a.url,
                            "kind": a.kind,
                            "tier": a.tier,
                            "alt": a.alt,
                            "caption": a.caption,
                            "pages": json.dumps(a.pages),
                            "machine_generated": a.machine_generated,
                            "content_hash": a.content_hash,
                            "fetched_at": now(),
                        }
                        for a in items
                    ],
                )

    def attachments_for(self, source_id: str) -> list[StagedAttachment]:
        with self.begin() as conn:
            rows = conn.execute(
                select(attachments)
                .where(and_(attachments.c.source == SOURCE, attachments.c.source_id == source_id))
                .order_by(attachments.c.id)
            ).all()
        return [
            StagedAttachment(
                url=r.url,
                kind=r.kind,
                tier=r.tier,
                alt=r.alt,
                caption=r.caption,
                pages=json.loads(r.pages),
                machine_generated=bool(r.machine_generated),
                content_hash=r.content_hash,
            )
            for r in rows
        ]

    # -- LLM cache ------------------------------------------------------

    def cache_get(self, key: str) -> str | None:
        with self.begin() as conn:
            row = conn.execute(select(llm_cache.c.value).where(llm_cache.c.key == key)).first()
        return row[0] if row else None

    def cache_put(self, key: str, task: str, model: str, value: str) -> None:
        with self.begin() as conn:
            if conn.execute(select(llm_cache.c.key).where(llm_cache.c.key == key)).first():
                conn.execute(update(llm_cache).where(llm_cache.c.key == key).values(value=value, model=model))
            else:
                conn.execute(insert(llm_cache).values(key=key, task=task, model=model, value=value, created_at=now()))

    # -- review queue and quarantine -------------------------------------

    def review_status(self, source_id: str, content_hash: str) -> str | None:
        with self.begin() as conn:
            row = conn.execute(
                select(review_queue.c.status).where(
                    and_(
                        review_queue.c.source == SOURCE,
                        review_queue.c.source_id == source_id,
                        review_queue.c.content_hash == content_hash,
                    )
                )
            ).first()
        return row[0] if row else None

    def queue_for_review(self, source_id: str, content_hash: str, title: str, url: str, reason: str) -> None:
        if self.review_status(source_id, content_hash) is not None:
            return
        with self.begin() as conn:
            conn.execute(
                insert(review_queue).values(
                    source=SOURCE,
                    source_id=source_id,
                    content_hash=content_hash,
                    title=title,
                    url=url,
                    reason=reason,
                    status="pending",
                    created_at=now(),
                )
            )

    def pending_reviews(self) -> list[dict[str, Any]]:
        with self.begin() as conn:
            rows = conn.execute(
                select(review_queue)
                .where(and_(review_queue.c.source == SOURCE, review_queue.c.status == "pending"))
                .order_by(review_queue.c.created_at)
            ).all()
        return [dict(r._mapping) for r in rows]

    def quarantine_hit(self, source_id: str, location: str, rule: str, line: int, content_hash: str) -> None:
        with self.begin() as conn:
            exists = conn.execute(
                select(quarantine.c.id).where(
                    and_(
                        quarantine.c.source == SOURCE,
                        quarantine.c.source_id == source_id,
                        quarantine.c.location == location,
                        quarantine.c.rule == rule,
                        quarantine.c.line == line,
                        quarantine.c.content_hash == content_hash,
                    )
                )
            ).first()
            if not exists:
                conn.execute(
                    insert(quarantine).values(
                        source=SOURCE,
                        source_id=source_id,
                        location=location,
                        rule=rule,
                        line=line,
                        content_hash=content_hash,
                        created_at=now(),
                    )
                )

    def quarantine_report(self) -> list[dict[str, Any]]:
        with self.begin() as conn:
            rows = conn.execute(
                select(quarantine).where(quarantine.c.source == SOURCE).order_by(quarantine.c.created_at)
            ).all()
        return [dict(r._mapping) for r in rows]

    # -- sync status ----------------------------------------------------

    def start_run(self, run_id: str, kind: str) -> None:
        started = now()
        with self.begin() as conn:
            conn.execute(insert(sync_runs).values(run_id=run_id, source=SOURCE, kind=kind, started_at=started))
            values = {
                "last_run_id": run_id,
                "last_kind": kind,
                "started_at": started,
                "finished_at": None,
                "ok": False,
                "error": "",
            }
            if conn.execute(select(sync_status.c.source).where(sync_status.c.source == SOURCE)).first():
                conn.execute(update(sync_status).where(sync_status.c.source == SOURCE).values(**values))
            else:
                conn.execute(insert(sync_status).values(source=SOURCE, counts="{}", **values))

    def finish_run(
        self,
        run_id: str,
        kind: str,
        ok: bool,
        counts: dict[str, int],
        error: str = "",
        full_completed: bool = False,
        reconciled: bool = False,
    ) -> None:
        """Record the end of a run. A full run that got through every page
        resets the reconcile clock even if some pages failed, so a persistent
        failure doesn't make every poll a full run. Only a run that actually
        deleted stale points sets the baseline for the safety valve."""
        finished = now()
        values: dict[str, Any] = {"finished_at": finished, "ok": ok, "counts": json.dumps(counts), "error": error}
        status_values = dict(values)
        if kind == "full" and full_completed:
            status_values["last_full_finished_at"] = finished
        if kind == "full" and reconciled:
            status_values.update(last_full_run_id=run_id, last_full_docs_seen=counts.get("seen", 0))
        with self.begin() as conn:
            conn.execute(update(sync_runs).where(sync_runs.c.run_id == run_id).values(**values))
            conn.execute(update(sync_status).where(sync_status.c.source == SOURCE).values(**status_values))

    def status(self) -> dict[str, Any] | None:
        with self.begin() as conn:
            row = conn.execute(select(sync_status).where(sync_status.c.source == SOURCE)).first()
        if not row:
            return None
        result = dict(row._mapping)
        result["counts"] = json.loads(result["counts"] or "{}")
        return result

    def recent_runs(self, limit: int = 10) -> list[dict[str, Any]]:
        with self.begin() as conn:
            rows = conn.execute(
                select(sync_runs)
                .where(sync_runs.c.source == SOURCE)
                .order_by(sync_runs.c.started_at.desc())
                .limit(limit)
            ).all()
        return [dict(r._mapping) for r in rows]

    def last_full_docs_seen(self) -> int:
        status = self.status()
        return int(status["last_full_docs_seen"]) if status else 0

    def last_full_finished_at(self) -> datetime | None:
        status = self.status()
        if not status or status["last_full_finished_at"] is None:
            return None
        value: datetime = status["last_full_finished_at"]
        return value if value.tzinfo else value.replace(tzinfo=UTC)

    # -- entity registry --------------------------------------------------

    def aliases(self) -> list[tuple[str, str, str]]:
        """(entity_type, entity_id, alias) for every alias in the registry."""
        with self.begin() as conn:
            rows = conn.execute(
                select(entity_aliases.c.entity_type, entity_aliases.c.entity_id, entity_aliases.c.alias)
            ).all()
        return [(r[0], r[1], r[2]) for r in rows]

    def entity_attributes(self) -> dict[tuple[str, str], dict[str, Any]]:
        with self.begin() as conn:
            rows = conn.execute(select(entities.c.entity_type, entities.c.entity_id, entities.c.attributes)).all()
        return {(r[0], r[1]): json.loads(r[2] or "{}") for r in rows}

    def set_entity_documents(self, doc_id: str, url: str, links_: list[tuple[str, str, str]]) -> None:
        """Replace the (entity_type, entity_id, role) rows this wiki page provides."""
        with self.begin() as conn:
            conn.execute(
                delete(entity_documents).where(
                    and_(entity_documents.c.source == SOURCE, entity_documents.c.doc_id == doc_id)
                )
            )
            if links_:
                conn.execute(
                    insert(entity_documents),
                    [
                        {
                            "entity_type": t,
                            "entity_id": i,
                            "role": role,
                            "source": SOURCE,
                            "doc_id": doc_id,
                            "url": url,
                            "updated_at": now(),
                        }
                        for t, i, role in sorted(set(links_))
                    ],
                )

    def clear_entity_documents(self, doc_ids: list[str]) -> None:
        if not doc_ids:
            return
        with self.begin() as conn:
            conn.execute(
                delete(entity_documents).where(
                    and_(entity_documents.c.source == SOURCE, entity_documents.c.doc_id.in_(doc_ids))
                )
            )

    def count_pages(self) -> int:
        with self.begin() as conn:
            return int(
                conn.execute(
                    select(func.count())
                    .select_from(documents)
                    .where(and_(documents.c.source == SOURCE, documents.c.deleted_at.is_(None)))
                ).scalar()
                or 0
            )


def _page_from_row(row: Any) -> StagedPage:  # a SQLAlchemy RowMapping
    return StagedPage(
        source_id=row["source_id"],
        locale=row["locale"],
        path=row["path"],
        title=row["title"],
        tier=row["tier"],
        url=row["url"],
        text=row["text"],
        content_hash=row["content_hash"],
        description=row["description"],
        tags=json.loads(row["tags"] or "[]"),
        parent_path=row["parent_path"],
        content_type=row["content_type"],
        raw_content=row["raw_content"],
        rendered_html=row["rendered_html"],
        toc=row["toc"],
        flags=json.loads(row["flags"] or "[]"),
        page_kind=row["page_kind"],
        created_at=row["source_created_at"],
        updated_at=row["source_updated_at"],
        indexed_key=row["indexed_key"],
    )
