"""Reports for humans, and the canary check: `wikijs2rag-report`.

    wikijs2rag-report status      last run, counts, recent runs
    wikijs2rag-report stale       pages not updated in 18 months (--months N)
    wikijs2rag-report flagged     pages and files held for review; redacted secrets
    wikijs2rag-report links       internal links to pages that don't exist
    wikijs2rag-report canaries    exit 1 if a canary is anywhere it must not be

Reads the staging database (DATABASE_URL); `canaries` also reads Qdrant.
"""

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import TextIO

from wikijs2rag import __version__
from wikijs2rag.config import ConfigError, Settings, database_url
from wikijs2rag.content import normalize_heading
from wikijs2rag.staging import Staging
from wikijs2rag.store import VectorStore


def status(staging: Staging, out: TextIO) -> int:
    row = staging.status()
    if not row:
        out.write("No runs yet.\n")
        return 0
    state = "running" if row["finished_at"] is None else "ok" if row["ok"] else "FAILED"
    out.write(f"Source wiki: last run {row['last_run_id']} ({row['last_kind']}) {state}\n")
    out.write(f"  started {row['started_at']}  finished {row['finished_at']}\n")
    out.write(f"  counts {json.dumps(row['counts'], sort_keys=True)}\n")
    if row["error"]:
        out.write(f"  error: {row['error']}\n")
    out.write(f"  last full reconcile {row['last_full_run_id'] or 'never'} at {row['last_full_finished_at']}\n")
    out.write(f"  pages staged {staging.count_pages()}  quarantine entries {len(staging.quarantine_report())}\n")
    out.write(f"  pending reviews {len(staging.pending_reviews())}\n\nRecent runs:\n")
    for run in staging.recent_runs():
        ok = "ok" if run["ok"] else ("running" if run["finished_at"] is None else "FAILED")
        out.write(f"  {run['run_id']}  {run['kind']:<11} {ok:<7} {run['counts']}\n")
    return 0


def _parse(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def stale(staging: Staging, out: TextIO, months: int) -> int:
    cutoff = datetime.now(UTC) - timedelta(days=round(months * 30.44))
    rows = []
    for page in staging.live_pages():
        updated = _parse(page.updated_at)
        if updated and updated < cutoff:
            rows.append((updated, page))
    out.write(f"{len(rows)} pages not updated in {months} months:\n")
    for updated, page in sorted(rows, key=lambda r: r[0]):
        out.write(f"  {updated:%Y-%m-%d}  [{page.tier}]  {page.title}  {page.url}\n")
    return 0


def flagged(staging: Staging, out: TextIO) -> int:
    reviews = staging.pending_reviews()
    out.write(f"{len(reviews)} items waiting for review (set review_queue.status to approved or rejected):\n")
    for r in reviews:
        out.write(f"  #{r['id']}  {r['title']}  {r['url']}\n      {r['reason']}\n")
    hits = staging.quarantine_report()
    out.write(f"\n{len(hits)} secrets redacted (fix them at the source):\n")
    for h in hits:
        out.write(f"  page {h['source_id']}  {h['location']}  line {h['line']}  {h['rule']}\n")
    redacted = [p for p in staging.live_pages() if p.flags]
    out.write(f"\n{len(redacted)} staged pages with flags:\n")
    for p in redacted:
        out.write(f"  {', '.join(p.flags)}  {p.title}  {p.url}\n")
    return 0


def broken_links(staging: Staging, out: TextIO) -> int:
    pages = staging.live_pages()
    paths = {(p.locale, p.path) for p in pages} | {("", p.path) for p in pages}
    by_id = {p.source_id: p for p in pages}
    broken = []
    for source_id, target, label in staging.all_internal_links():
        page = by_id[source_id]
        parts = target.split("/", 1)
        candidates = {(page.locale, target), ("", target)}
        if len(parts) == 2 and len(parts[0]) in (2, 5):  # a locale prefix, e.g. en/ or pt-br/
            candidates.add((parts[0], parts[1]))
        if not candidates & paths:
            broken.append((page, target, label))
    out.write(f"{len(broken)} internal links to pages neither crawl can see (missing, or restricted to admins):\n")
    for page, target, label in sorted(broken, key=lambda b: (b[0].path, b[1])):
        out.write(f"  {page.url}  ->  /{target}  ({normalize_heading(label) or 'no label'})\n")
    return 0


def canaries(settings: Settings, staging: Staging, out: TextIO) -> int:
    """Admin canaries must be nowhere; member canaries must not be public."""
    problems: list[str] = []
    for page in staging.live_pages():
        blob = f"{page.title}\n{page.raw_content}\n{page.text}"
        problems += [f"admin canary {c!r} staged in page {page.source_id}" for c in settings.canary_admin if c in blob]
        if page.tier == "public":
            problems += [
                f"member canary {c!r} staged as public in page {page.source_id}"
                for c in settings.canary_member
                if c in blob
            ]
    q = settings.qdrant
    store = VectorStore(q.host, q.port, settings.embedding.model, 0)
    for collection in (q.public_collection, q.member_collection):
        if not store.client.collection_exists(collection):
            continue
        for doc_id, text in store.scroll_texts(collection):
            problems += [f"admin canary {c!r} in {collection} doc {doc_id}" for c in settings.canary_admin if c in text]
            if collection == q.public_collection:
                problems += [
                    f"member canary {c!r} in {collection} doc {doc_id}" for c in settings.canary_member if c in text
                ]
    if not settings.canary_admin and not settings.canary_member:
        out.write("No canaries configured (CANARY_ADMIN, CANARY_MEMBER).\n")
        return 0
    for problem in problems:
        out.write(f"FAIL: {problem}\n")
    if not problems:
        out.write("Canaries OK.\n")
    return 1 if problems else 0


def main(argv: Sequence[str] | None = None, out: TextIO | None = None, env: Mapping[str, str] = os.environ) -> int:
    out = out or sys.stdout
    parser = argparse.ArgumentParser(prog="wikijs2rag-report", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("report", choices=["status", "stale", "flagged", "links", "canaries"])
    parser.add_argument("--months", type=int, default=18, help="For stale: months without an update (default 18)")
    args = parser.parse_args(argv)
    try:
        if args.report == "canaries":
            settings = Settings.from_env(env)
            url = settings.database_url
        else:
            url = database_url(env, Settings.database_url)
    except ConfigError as exc:
        sys.stderr.write(f"wikijs2rag-report: {exc}\n")
        return 2
    staging = Staging(url)
    try:
        if args.report == "status":
            return status(staging, out)
        if args.report == "stale":
            return stale(staging, out, args.months)
        if args.report == "flagged":
            return flagged(staging, out)
        if args.report == "links":
            return broken_links(staging, out)
        return canaries(settings, staging, out)
    finally:
        staging.close()


if __name__ == "__main__":
    sys.exit(main())
