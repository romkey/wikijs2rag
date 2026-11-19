"""End-to-end sync runs against a fake Wiki.js that enforces permissions by
credential, a fake LLM gateway, in-memory Qdrant and SQLite staging."""

import json
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from wikijs2rag import config, ingest
from wikijs2rag.config import LLMSettings, Settings
from wikijs2rag.staging import StagedPage, Staging, entities, entity_aliases, entity_documents, review_queue

WIKI = "http://wiki.test"
GQL = f"{WIKI}/graphql"
LLM_URL = "http://llm.test/v1/chat/completions"
MEMBER_KEY = "member-key-for-tests"
ADMIN_CANARY = "CANARY-ADMIN-7f3a"
MEMBER_CANARY = "CANARY-MEMBER-91bc"


_open: list[Staging] = []


def _staging(settings: Settings) -> Staging:
    staging = Staging(settings.database_url)
    _open.append(staging)
    return staging


@pytest.fixture(autouse=True)
def close_staging() -> Iterator[None]:
    yield
    while _open:
        _open.pop().close()


class FakeEmbedder:
    dimension = 3
    calls = 0

    def __init__(self, *_a: object, **_k: object) -> None:
        pass

    def encode_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        FakeEmbedder.calls += 1
        if any("EMBEDFAIL" in t for t in texts):
            raise RuntimeError("embed fail")
        return [[0.1, 0.2, 0.3] for _ in texts]


def _page(pid: int, path: str, title: str, content: str, render: str = "", **extra: Any) -> dict[str, Any]:
    return {
        "id": pid,
        "path": path,
        "locale": "en",
        "title": title,
        "description": "",
        "isPublished": True,
        "isPrivate": False,
        "contentType": "markdown",
        "createdAt": "2020-01-01T00:00:00Z",
        "updatedAt": "2024-01-01T00:00:00Z",
        "tags": [],
        "content": content,
        "render": render,
        **extra,
    }


class FakeWiki:
    """Pages have a tier: public (guests read), member (the member key reads)
    or admin (nobody this pipeline uses can read)."""

    def __init__(self) -> None:
        self.pages: dict[int, tuple[str, dict[str, Any]]] = {}
        self.files: dict[str, tuple[str, bytes, str]] = {}  # path -> (tier, bytes, type)
        self.member_key_is_admin = False
        self.anon_list_broken = False

    def add(self, tier: str, page: dict[str, Any]) -> None:
        self.pages[page["id"]] = (tier, page)

    def _can(self, request: httpx.Request, tier: str) -> bool:
        auth = request.headers.get("Authorization", "")
        if tier == "public":
            return True
        return tier == "member" and auth == f"Bearer {MEMBER_KEY}"

    def graphql(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        q = body["query"]
        auth = request.headers.get("Authorization", "")
        if "apiKeys" in q:
            if self.member_key_is_admin and auth:
                return httpx.Response(200, json={"data": {"authentication": {"apiKeys": []}}})
            return httpx.Response(200, json={"errors": [{"message": "Forbidden"}]})
        if "list" in q:
            if self.anon_list_broken and not auth:
                return httpx.Response(200, json={"errors": [{"message": "You are not authorized"}]})
            listed = [
                {k: v for k, v in p.items() if k not in ("content", "render")}
                for tier, p in self.pages.values()
                if self._can(request, tier) or (tier == "admin" and self.member_key_is_admin and auth)
            ]
            return httpx.Response(200, json={"data": {"pages": {"list": listed}}})
        pid = body["variables"]["id"]
        tier, page = self.pages[pid]
        if not self._can(request, tier) and not (self.member_key_is_admin and auth):
            return httpx.Response(200, json={"errors": [{"message": "x", "extensions": {"exception": {"code": 6013}}}]})
        single = {**page, "tags": [{"tag": t} for t in page["tags"]]}
        return httpx.Response(200, json={"data": {"pages": {"single": single}}})

    def get(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path in self.files:
            tier, data, media_type = self.files[path]
            if self._can(request, tier):
                return httpx.Response(200, content=data, headers={"content-type": media_type})
            return httpx.Response(403)
        for tier, page in self.pages.values():
            if path.strip("/") in (page["path"], f"en/{page['path']}"):
                if tier == "public":
                    return httpx.Response(200, text="<html>page</html>")
                return httpx.Response(302, headers={"location": f"{WIKI}/login"})
        if path == "/login":
            return httpx.Response(200, text="<form>login</form>")
        return httpx.Response(404)


def fake_llm(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    messages = body["messages"]
    system = messages[0]["content"] if messages[0]["role"] == "system" else ""
    user = messages[-1]["content"]
    user_text = user if isinstance(user, str) else user[0]["text"]
    if not system:  # vision
        reply = "A sign reading: LASER CUTTER - WEAR GOGGLES"
    elif "topics from this list" in system:
        reply = '{"topics": ["lasers", "not-a-topic"]}'
    elif "classify" in system:
        reply = '{"kind": "equipment"}' if "Laser" in user_text else '{"kind": "procedure"}'
    elif "You screen" in system:
        flagged = "fob log" in user_text.lower()
        reply = json.dumps({"exclude": flagged, "reason": "member records" if flagged else ""})
    elif "could mean" in system:
        reply = '{"entity": "asset:laser-1"}'
    elif "Summarize" in system:
        reply = "A short summary."
    else:
        reply = "This excerpt is about the page."
    return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})


@pytest.fixture
def wiki() -> Iterator[FakeWiki]:
    fake = FakeWiki()
    fake.add(
        "public",
        _page(
            1,
            "equipment/laser",
            "Laser Cutter",
            "# Laser Cutter\n\nThe big laser.\n\n## Safety\n\n### Eye protection\n\nWear goggles near the laser.\n\n"
            "```yaml\nhost: 10.0.0.5\nwifi_password: Zq8#kLm2pX9w\n```\n",
            render='<h1 id="laser-cutter">Laser Cutter</h1><h2 id="safety">Safety</h2>'
            '<h3 id="eye-protection">Eye protection</h3>'
            '<a href="/equipment/manual.pdf">manual</a> <a href="/rules">rules</a> <a href="/missing-page">x</a>'
            ' <a href="https://example.com/ext">ext</a> <img src="/signs/laser.png" alt="sign">'
            ' <a href="/member-notes.txt">notes</a>',
            tags=["lasers"],
        ),
    )
    fake.add("public", _page(2, "rules", "Rules", "Be excellent. Read the [laser](/equipment/laser) page."))
    fake.add("member", _page(3, "members/door", "Door procedure", f"Use your fob at the door. {MEMBER_CANARY}"))
    fake.add("member", _page(4, "members/fobs", "Fob audit", "The fob log for March shows entries."))
    fake.add("admin", _page(5, "infra/zigbee", "Zigbee admin", f"Remove a sensor. {ADMIN_CANARY}"))
    fake.add("public", _page(6, "admin/secret", "Excluded", "Under an excluded path"))
    fake.files["/equipment/manual.pdf"] = ("public", _pdf("Laser manual page text"), "application/pdf")
    fake.files["/signs/laser.png"] = ("public", b"\x89PNG fake", "image/png")
    fake.files["/member-notes.txt"] = ("member", b"Notes only members may read.", "text/plain")
    with respx.mock(assert_all_called=False) as mock:
        mock.post(GQL).mock(side_effect=fake.graphql)
        mock.get(url__startswith=WIKI).mock(side_effect=fake.get)
        mock.post(LLM_URL).mock(side_effect=fake_llm)
        yield fake


def _pdf(text: str) -> bytes:
    """A minimal one-page PDF with *text*."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, memory_qdrant: QdrantClient) -> Settings:
    monkeypatch.setattr(config, "OpenAIEmbedder", FakeEmbedder)
    db = f"sqlite:///{tmp_path / 'staging.db'}"
    staging = Staging(db)
    with staging.begin() as conn:
        conn.execute(entities.insert().values(entity_type="location", entity_id="loc-2", attributes='{"section": "2"}'))
        conn.execute(
            entity_aliases.insert(),
            [
                {"entity_type": "asset", "entity_id": "laser-1", "alias": "big laser"},
                {"entity_type": "asset", "entity_id": "laser-1", "alias": "Laser Cutter"},
                {"entity_type": "location", "entity_id": "loc-2", "alias": "the door"},
                {"entity_type": "asset", "entity_id": "laser-1", "alias": "goggles"},
                {"entity_type": "asset", "entity_id": "goggle-rack", "alias": "goggles"},
            ],
        )
    staging.close()
    return Settings(
        wiki_url=WIKI,
        wiki_member_api_key=MEMBER_KEY,
        database_url=db,
        llm=LLMSettings(base_url="http://llm.test/v1", model="gen", vision_model="vis"),
        page_delay=0,
        canary_admin=(ADMIN_CANARY,),
        canary_member=(MEMBER_CANARY,),
    )


def _points(client: QdrantClient, collection: str, doc_id: str | None = None) -> list[dict[str, Any]]:
    flt = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]) if doc_id else None
    points, _ = client.scroll(collection, scroll_filter=flt, limit=1000, with_payload=True)
    return [p.payload or {} for p in points]


def _docs(client: QdrantClient, collection: str) -> set[str]:
    return {p["doc_id"] for p in _points(client, collection)}


def test_two_crawls_tier_pages(wiki: FakeWiki, settings: Settings, memory_qdrant: QdrantClient) -> None:
    assert ingest.run(settings) is True
    public, member = _docs(memory_qdrant, "hs_public"), _docs(memory_qdrant, "hs_member")
    assert public == {"1", "2", "_meta"}
    assert member == {"1", "2", "3", "_meta"}  # 4 is held for review; 5 is admin; 6 excluded path
    # The member canary is in hs_member and not hs_public; the admin canary is nowhere.
    texts = {c: " ".join(p["text"] for p in _points(memory_qdrant, c)) for c in ("hs_public", "hs_member")}
    assert MEMBER_CANARY in texts["hs_member"]
    assert MEMBER_CANARY not in texts["hs_public"]
    assert all(ADMIN_CANARY not in t for t in texts.values())
    # Every public point came from the anonymous crawl.
    assert {p["tier"] for p in _points(memory_qdrant, "hs_public")} == {"public"}

    staging = _staging(settings)
    assert {p.source_id for p in staging.live_pages()} == {"1", "2", "3", "4"}  # admin page never staged
    assert staging.status()["ok"] is True  # type: ignore[index]
    assert staging.status()["last_kind"] == "full"  # type: ignore[index]
    reviews = staging.pending_reviews()
    assert [r["source_id"] for r in reviews] == ["4"]
    assert reviews[0]["reason"] == "member records"


def test_payload_shape_anchor_and_links(wiki: FakeWiki, settings: Settings, memory_qdrant: QdrantClient) -> None:
    ingest.run(settings)
    points = sorted(_points(memory_qdrant, "hs_member", "1"), key=lambda p: p["chunk_index"])
    summary = points[0]
    assert summary["chunk_kind"] == "summary"
    assert summary["text"] == "A short summary."
    deep = next(p for p in points if "Wear goggles" in p["text"])
    assert deep["url"] == f"{WIKI}/equipment/laser#eye-protection"
    assert deep["heading_path"] == "equipment > Laser Cutter > Safety > Eye protection"
    assert deep["context"] == "This excerpt is about the page."
    assert deep["authority"] == 4
    assert deep["page_kind"] == "equipment"
    assert deep["topics"] == ["lasers"]
    assert deep["asset_ids"] == ["laser-1"]  # page-level link, and "goggles" resolved by the LLM
    assert deep["source"] == "wiki"
    assert deep["tier"] == "public"
    assert deep["lang"] == "en"
    assert set(deep) >= {
        "id", "doc_id", "chunk_index", "title", "created_at", "updated_at", "indexed_at", "stale_after",
        "superseded_by", "content_hash", "sync_run_id", "location_ids", "work_ids", "event_ids",
        "training_ids", "section", "page",
    }  # fmt: skip
    code = next(p for p in points if "```yaml" in p["text"])
    assert "10.0.0.5" not in code["text"]
    assert "<IP>" in code["text"]
    assert "Zq8#kLm2pX9w" not in code["text"]
    assert "[REDACTED]" in code["text"]
    pdf = next(p for p in points if p["chunk_kind"] == "attachment" and p["page"] == 1)
    assert "Laser manual page text" in pdf["text"]
    assert pdf["url"] == f"{WIKI}/equipment/manual.pdf#page=1"
    image = next(p for p in points if p["chunk_kind"] == "image")
    assert image["machine_generated"] is True
    assert "WEAR GOGGLES" in image["text"]
    notes = next(p for p in points if "Notes only members" in p["text"])
    assert notes["tier"] == "member"  # a member-only file on a public page
    assert "Notes only members" not in " ".join(p["text"] for p in _points(memory_qdrant, "hs_public"))

    door = _points(memory_qdrant, "hs_member", "3")
    assert all(p["location_ids"] == ["loc-2"] and p["section"] == "2" for p in door if p["chunk_kind"] == "chunk")

    staging = _staging(settings)
    links = {(link.kind, link.target) for link in staging.links_for("1")}
    assert links == {("internal", "rules"), ("internal", "missing-page"), ("external", "https://example.com/ext")}
    assert [h["rule"] for h in staging.quarantine_report()] == ["assigned-secret"]
    assert "Zq8#kLm2pX9w" not in (staging.get_page("1") or StagedPage("", "", "", "", "", "", "", "")).raw_content
    with staging.begin() as conn:
        rows = conn.execute(entity_documents.select()).all()
    assert [(r.entity_id, r.role, r.doc_id) for r in rows] == [("laser-1", "how-to-use", "1")]


def test_incremental_skips_unchanged_and_follows_tier_changes(
    wiki: FakeWiki, settings: Settings, memory_qdrant: QdrantClient
) -> None:
    ingest.run(settings)
    calls = FakeEmbedder.calls
    assert ingest.run(settings, full=False) is True
    assert FakeEmbedder.calls == calls + 2  # only the metadata points, one batch per collection
    # Page 2 becomes member-only: out of hs_public on the next run.
    _tier, page = wiki.pages[2]
    wiki.pages[2] = ("member", {**page, "updatedAt": "2025-01-01T00:00:00Z"})
    assert ingest.run(settings, full=False) is True
    assert "2" not in _docs(memory_qdrant, "hs_public")
    assert {p["tier"] for p in _points(memory_qdrant, "hs_member", "2")} == {"member"}


def test_full_reconcile_deletes_and_safety_valve(
    wiki: FakeWiki, settings: Settings, memory_qdrant: QdrantClient
) -> None:
    ingest.run(settings)
    del wiki.pages[3]
    assert ingest.run(settings, full=True) is True
    assert "3" not in _docs(memory_qdrant, "hs_member")
    assert "3" not in {p.source_id for p in _staging(settings).live_pages()}
    # Most pages vanish (credentials broken?): nothing is deleted, the run fails.
    for pid in (1, 2):
        del wiki.pages[pid]
    assert ingest.run(settings, full=True) is False
    assert {"1", "2"} <= _docs(memory_qdrant, "hs_member")


def test_review_approval_indexes_page(wiki: FakeWiki, settings: Settings, memory_qdrant: QdrantClient) -> None:
    ingest.run(settings)
    staging = _staging(settings)
    with staging.begin() as conn:
        conn.execute(review_queue.update().values(status="approved"))
    ingest.run(settings, full=False)
    assert "4" in _docs(memory_qdrant, "hs_member")
    assert "4" not in _docs(memory_qdrant, "hs_public")


def test_admin_canary_stops_run(wiki: FakeWiki, settings: Settings) -> None:
    wiki.add("member", _page(7, "members/leak", "Leak", f"Copied from admin: {ADMIN_CANARY}"))
    with pytest.raises(ingest.IngestError, match="Admin canary"):
        ingest.run(settings)
    assert "7" not in {p.source_id for p in _staging(settings).live_pages()}
    assert _staging(settings).status()["ok"] is False  # type: ignore[index]


def test_member_canary_in_public_stops_run(wiki: FakeWiki, settings: Settings) -> None:
    wiki.add("public", _page(8, "oops", "Oops", f"{MEMBER_CANARY}"))
    with pytest.raises(ingest.IngestError, match="Member canary"):
        ingest.run(settings)


def test_admin_credential_refused(wiki: FakeWiki, settings: Settings) -> None:
    wiki.member_key_is_admin = True
    with pytest.raises(ingest.IngestError, match="admin rights"):
        ingest.run(settings)


def test_guest_fallback_when_anonymous_api_disabled(
    wiki: FakeWiki, settings: Settings, memory_qdrant: QdrantClient
) -> None:
    wiki.anon_list_broken = True
    assert ingest.run(replace(settings, canary_member=())) is True
    assert _docs(memory_qdrant, "hs_public") == {"1", "2", "_meta"}


def test_public_only_without_member_key(wiki: FakeWiki, settings: Settings, memory_qdrant: QdrantClient) -> None:
    plain = replace(settings, wiki_member_api_key=None, llm=LLMSettings())
    assert ingest.run(plain) is True
    assert _docs(memory_qdrant, "hs_member") == {"1", "2", "_meta"}
    point = next(p for p in _points(memory_qdrant, "hs_public", "2") if p["chunk_kind"] == "chunk")
    assert point["context"] == 'From the wiki page "Rules".'
    # Without an LLM the ambiguous alias "goggles" is dropped, but the unambiguous one isn't.
    laser = [p for p in _points(memory_qdrant, "hs_public", "1") if "goggles" in p["text"]]
    assert all(p["asset_ids"] == ["laser-1"] for p in laser)


def test_errors_are_counted_and_retried(wiki: FakeWiki, settings: Settings, memory_qdrant: QdrantClient) -> None:
    tier, page = wiki.pages[2]
    wiki.pages[2] = (tier, {**page, "content": "EMBEDFAIL"})
    assert ingest.run(settings) is False
    assert "2" not in _docs(memory_qdrant, "hs_public")
    wiki.pages[2] = (tier, page)
    assert ingest.run(settings, full=False) is True
    assert "2" in _docs(memory_qdrant, "hs_public")


def test_reindex_from_staging(wiki: FakeWiki, settings: Settings, memory_qdrant: QdrantClient) -> None:
    ingest.run(settings)
    before = _docs(memory_qdrant, "hs_member")
    wiki.pages.clear()  # the wiki isn't consulted
    assert ingest.run(settings, reindex=True) is True
    assert _docs(memory_qdrant, "hs_member") == before


def test_stop_ends_run_early(wiki: FakeWiki, settings: Settings) -> None:
    stop = threading.Event()
    stop.set()
    assert ingest.run(settings, stop) is False


def test_run_skipped_while_locked_returns_none(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    @contextmanager
    def held(self: Staging, source: str = "wiki") -> Iterator[bool]:
        yield False

    monkeypatch.setattr(Staging, "source_lock", held)
    assert ingest.run(settings) is None


def test_page_url_and_exclusions() -> None:
    s = Settings(wiki_url=WIKI, wiki_excluded_paths=("admin", "board/private"))
    assert ingest.page_url(s, "en", "/a/b") == f"{WIKI}/a/b"
    assert ingest.page_url(s, "fr", "a") == f"{WIKI}/fr/a"
    assert ingest.is_excluded(s, "admin")
    assert ingest.is_excluded(s, "board/private/x")
    assert not ingest.is_excluded(s, "administrivia")


def test_metadata_is_per_tier_and_names_no_people() -> None:
    pages = [
        StagedPage("1", "en", "a", "A", "public", "u", "t", "h", updated_at="2024-02-01T00:00:00Z"),
        StagedPage("2", "en", "b", "B", "member", "u", "t", "h", created_at="2023-01-01T00:00:00Z"),
    ]
    payloads = ingest.build_wiki_metadata(pages, WIKI, "member", "run")
    by_type = {p["meta_type"]: p["text"] for p in payloads}
    assert set(by_type) == {"overview", "recent", "newest", "page_listing"}
    assert "Total public and member-only pages: 2" in by_type["overview"]
    assert re.search(r"contributor", " ".join(by_type.values()), re.I) is None


def test_approved_attachment_gets_indexed(wiki: FakeWiki, settings: Settings, memory_qdrant: QdrantClient) -> None:
    wiki.files["/member-notes.txt"] = ("member", b"The fob log export for March.", "text/plain")
    ingest.run(settings)
    assert "fob log export" not in " ".join(p["text"] for p in _points(memory_qdrant, "hs_member", "1"))
    staging = _staging(settings)
    review_id = "1:attachment:http://wiki.test/member-notes.txt"
    assert review_id in [r["source_id"] for r in staging.pending_reviews()]
    assert ingest.run(settings, full=False) is True  # still held: no change
    with staging.begin() as conn:
        conn.execute(review_queue.update().where(review_queue.c.source_id == review_id).values(status="approved"))
    assert ingest.run(settings, full=False) is True
    notes = [p for p in _points(memory_qdrant, "hs_member", "1") if "fob log export" in p["text"]]
    assert [p["tier"] for p in notes] == ["member"]


def test_failed_update_is_retried_and_old_points_kept(
    wiki: FakeWiki, settings: Settings, memory_qdrant: QdrantClient
) -> None:
    ingest.run(settings)
    tier, page = wiki.pages[2]
    wiki.pages[2] = (tier, {**page, "content": "EMBEDFAIL", "updatedAt": "2025-02-02T00:00:00Z"})
    assert ingest.run(settings, full=False) is False
    assert "2" in _docs(memory_qdrant, "hs_public")  # the old points survive the failure...
    assert ingest.run(settings, full=True) is False
    assert "2" in _docs(memory_qdrant, "hs_public")  # ...and a full reconcile
    wiki.pages[2] = (tier, {**page, "content": "Fixed text.", "updatedAt": "2025-02-02T00:00:00Z"})
    assert ingest.run(settings, full=False) is True  # same updatedAt, still retried
    assert any("Fixed text." in p["text"] for p in _points(memory_qdrant, "hs_public", "2"))


def test_metadata_leaves_out_held_pages(wiki: FakeWiki, settings: Settings, memory_qdrant: QdrantClient) -> None:
    ingest.run(settings)
    meta = " ".join(p["text"] for p in _points(memory_qdrant, "hs_member", "_meta"))
    assert "Door procedure" in meta
    assert "Fob audit" not in meta  # held for review


def test_failed_full_run_still_resets_reconcile_clock(wiki: FakeWiki, settings: Settings) -> None:
    tier, page = wiki.pages[2]
    wiki.pages[2] = (tier, {**page, "content": "EMBEDFAIL"})
    assert ingest.run(settings) is False  # the first run is full
    assert ingest.run(settings) is False
    status = _staging(settings).status()
    assert status is not None
    assert status["last_kind"] == "incremental"
    assert status["last_full_docs_seen"] == 4  # reconciled despite the page error
