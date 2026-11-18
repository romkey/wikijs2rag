import io
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from wikijs2rag import report
from wikijs2rag.staging import Link, StagedPage, Staging
from wikijs2rag.store import VectorStore, point_id


@pytest.fixture
def db(tmp_path: Path) -> str:
    url = f"sqlite:///{tmp_path / 's.db'}"
    with Staging(url) as staging:
        old = StagedPage(
            "1", "en", "old", "Old page", "public", "http://w/old", "t", "h", updated_at="2020-01-01T00:00:00Z"
        )
        new = StagedPage(
            "2", "en", "new", "New page", "member", "http://w/new", "MEMBER-CANARY", "h2", flags=["secret-redacted"]
        )
        staging.stage_page(old, "r", [Link("internal", "new", "", "New"), Link("internal", "gone", "", "Gone")])
        staging.stage_page(new, "r", [Link("internal", "en/old", "", "")])
        staging.queue_for_review("2", "h2", "New page", "http://w/new", "member records")
        staging.quarantine_hit("2", "page", "jwt", 4, "h2")
        staging.start_run("r", "full")
        staging.finish_run("r", "full", True, {"seen": 2}, "")
    return url


def _run(argv: list[str], env: dict[str, str]) -> tuple[int, str]:
    out = io.StringIO()
    return report.main(argv, out, env), out.getvalue()


def test_status(db: str) -> None:
    code, text = _run(["status"], {"DATABASE_URL": db})
    assert code == 0
    assert "last run r (full) ok" in text
    assert "pages staged 2" in text
    assert "pending reviews 1" in text


def test_status_without_runs(tmp_path: Path) -> None:
    assert _run(["status"], {"DATABASE_URL": f"sqlite:///{tmp_path / 'e.db'}"}) == (0, "No runs yet.\n")


def test_stale_flagged_links(db: str) -> None:
    env = {"DATABASE_URL": db}
    assert _run(["stale"], env)[0] == 0
    text = _run(["stale"], env)[1]
    assert "1 pages not updated in 18 months" in text
    assert "Old page" in text
    text = _run(["flagged"], env)[1]
    assert "member records" in text
    assert "jwt" in text
    assert "secret-redacted" in text
    text = _run(["links"], env)[1]
    assert text.startswith("1 internal links")
    assert "/gone" in text


def test_canaries(db: str, memory_qdrant: QdrantClient) -> None:
    store = VectorStore("h", 1, "nomic-ai/modernbert-embed-base", 2)
    store.ensure_collection("hs_public", "public")
    payload = {"id": point_id("wiki", "2", 0), "source": "wiki", "doc_id": "2", "text": "MEMBER-CANARY", "context": ""}
    store.replace_doc("hs_public", "wiki", "2", [[0.1, 0.2]], [payload])
    env = {"DATABASE_URL": db, "WIKI_URL": "http://w", "CANARY_MEMBER": "MEMBER-CANARY", "CANARY_ADMIN": "ADMIN-X"}
    code, text = _run(["canaries"], env)
    assert code == 1
    assert "FAIL: member canary 'MEMBER-CANARY' in hs_public doc 2" in text
    store.delete_doc("hs_public", "wiki", "2")
    assert _run(["canaries"], env) == (0, "Canaries OK.\n")
    assert _run(["canaries"], {"DATABASE_URL": db, "WIKI_URL": "http://w"})[1].startswith("No canaries")


def test_bad_config(db: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["canaries"], {"DATABASE_URL": db})[0] == 2
    assert "WIKI_URL" in capsys.readouterr().err
