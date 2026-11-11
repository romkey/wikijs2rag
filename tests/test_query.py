import io
from types import SimpleNamespace
from typing import ClassVar

import pytest
from qdrant_client.models import ScoredPoint

from wikijs2rag import __version__, query
from wikijs2rag.query import Style

PLAIN = Style(color=False)


def _hit(score: float = 0.9, **payload: object) -> ScoredPoint:
    return ScoredPoint(id=1, version=0, score=score, payload=payload)


class TtyOut(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_style_color_and_plain() -> None:
    assert Style(color=True).bold("x") == "\033[1mx\033[0m"
    assert Style(color=True).cyan("x") == "\033[36mx\033[0m"
    assert PLAIN.yellow("x") == "x"


@pytest.mark.parametrize(("score", "code"), [(0.9, "32"), (0.6, "33"), (0.2, "31")])
def test_score_bar_colors(score: float, code: str) -> None:
    bar = Style(color=True).score_bar(score, width=10)
    assert bar.startswith(f"\033[{code}m")
    assert bar.count("█") == round(score * 10)


def test_format_result_full() -> None:
    hit = _hit(page_title="T", page_url="http://u", section="S", tags=["a", "b"], text="line1\nline2")
    out = query.format_result(1, hit, show_text=True, text_width=40, style=PLAIN)
    assert out.splitlines()[0] == "#1  T  ›  S"  # noqa: RUF001 -- the glyph under test
    assert "http://u" in out
    assert "0.900" in out
    assert "tags: a, b" in out
    assert "│ line1 line2" in out


def test_format_result_minimal() -> None:
    out = query.format_result(2, _hit(page_path="p/x"), show_text=True, text_width=40, style=PLAIN)
    assert out.splitlines()[0] == "#2  p/x"
    assert "tags" not in out
    assert "│" not in out
    assert query.format_result(3, _hit(), False, 40, PLAIN).startswith("#3  Untitled")
    no_payload = ScoredPoint(id=1, version=0, score=0.5, payload=None)
    assert query.format_result(4, no_payload, False, 40, PLAIN).startswith("#4  Untitled")


class FakeEmbedder:
    error: Exception | None = None

    def __init__(self, *_a: object, **_k: object) -> None:
        if FakeEmbedder.error:
            raise FakeEmbedder.error

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 1.0]]


class FakeClient:
    hits: ClassVar[list[ScoredPoint]] = []
    error: Exception | None = None
    host: tuple[str, int] = ("", 0)
    last_kwargs: ClassVar[dict[str, object]] = {}

    def __init__(self, host: str, port: int) -> None:
        FakeClient.host = (host, port)

    def query_points(self, **kwargs: object) -> SimpleNamespace:
        FakeClient.last_kwargs = kwargs
        if FakeClient.error:
            raise FakeClient.error
        return SimpleNamespace(points=FakeClient.hits)


@pytest.fixture(autouse=True)
def fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(query, "OllamaEmbedder", FakeEmbedder)
    monkeypatch.setattr(query, "QdrantClient", FakeClient)
    for var in ("QUERY_LIMIT", "QUERY_MIN_SCORE", "QUERY_SHOW_TEXT", "QUERY_WIDTH", "NO_COLOR", "QDRANT_COLLECTION"):
        monkeypatch.delenv(var, raising=False)
    FakeEmbedder.error = None
    FakeClient.hits = []
    FakeClient.error = None


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        query.main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"wikijs2rag-query {__version__}\n"


def test_prints_results_with_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_HOST", "qh")
    monkeypatch.setenv("QDRANT_PORT", "1234")
    FakeClient.hits = [_hit(page_title="One", text="body"), _hit(page_title="Two")]
    out = io.StringIO()
    argv = ["--limit", "2", "--min-score", "0.5", "-c", "coll", "--show-text", "--width", "30", "hello", "world"]
    assert query.main(argv, out) == 0
    text = out.getvalue()
    assert "Query: hello world" in text
    assert "#1  One" in text
    assert "#2  Two" in text
    assert "│ body" in text
    assert "\033[" not in text  # not a terminal
    assert FakeClient.host == ("qh", 1234)
    assert FakeClient.last_kwargs["collection_name"] == "coll"
    assert FakeClient.last_kwargs["limit"] == 2
    assert FakeClient.last_kwargs["score_threshold"] == 0.5


def test_environment_defaults_and_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUERY_LIMIT", "7")
    monkeypatch.setenv("QDRANT_COLLECTION", "envcoll")
    FakeClient.hits = [_hit(page_title="One")]
    out = TtyOut()
    assert query.main(["q"], out) == 0
    assert "\033[1m" in out.getvalue()
    assert FakeClient.last_kwargs["limit"] == 7
    assert FakeClient.last_kwargs["collection_name"] == "envcoll"
    assert FakeClient.last_kwargs["score_threshold"] is None


def test_no_results() -> None:
    out = io.StringIO()
    assert query.main(["nothing"], out) == 0
    assert "No results found." in out.getvalue()


def test_bad_config(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("QUERY_LIMIT", "lots")
    assert query.main(["q"]) == 2
    assert "QUERY_LIMIT" in capsys.readouterr().err


def test_qdrant_error(capsys: pytest.CaptureFixture[str]) -> None:
    FakeClient.error = RuntimeError("down")
    assert query.main(["q"], io.StringIO()) == 1
    assert "Qdrant error: down" in capsys.readouterr().err


def test_embedder_error(capsys: pytest.CaptureFixture[str]) -> None:
    FakeEmbedder.error = RuntimeError("no ollama")
    assert query.main(["q"], io.StringIO()) == 1
    assert "Embedding error: no ollama" in capsys.readouterr().err
