import signal
import threading
from collections.abc import Callable

import pytest

from wikijs2rag import __main__ as cli
from wikijs2rag import __version__
from wikijs2rag.config import Settings
from wikijs2rag.ingest import IngestError

type Handler = Callable[[int, object], None]


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"wikijs2rag {__version__}\n"


def test_bad_config_exits_2(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("WIKI_URL", raising=False)
    with pytest.raises(SystemExit) as exit_info:
        cli.main([])
    assert exit_info.value.code == 2
    assert "WIKI_URL is required" in capsys.readouterr().err


@pytest.fixture
def handlers(monkeypatch: pytest.MonkeyPatch) -> dict[int, Handler]:
    """Configure via env and capture signal handlers instead of installing them."""
    monkeypatch.setenv("WIKI_URL", "http://wiki")
    found: dict[int, Handler] = {}
    monkeypatch.setattr(signal, "signal", found.__setitem__)
    return found


def _fake_run(
    monkeypatch: pytest.MonkeyPatch, results: list[bool | BaseException], on_call: Callable[[], None] | None = None
) -> list[threading.Event]:
    """Replace ingest.run with one returning/raising *results* in turn."""
    seen: list[threading.Event] = []
    it = iter(results)

    def fake(settings: Settings, stop: threading.Event) -> bool:
        seen.append(stop)
        if on_call:
            on_call()
        r = next(it)
        if isinstance(r, BaseException):
            raise r
        return r

    monkeypatch.setattr(cli, "run", fake)
    return seen


@pytest.mark.usefixtures("handlers")
@pytest.mark.parametrize(("result", "code"), [(True, cli.OK), (False, cli.PAGE_ERRORS), (IngestError("x"), cli.FATAL)])
def test_one_shot(monkeypatch: pytest.MonkeyPatch, result: bool | BaseException, code: int) -> None:
    monkeypatch.setenv("POLL_INTERVAL_SECONDS", "0")
    _fake_run(monkeypatch, [result])
    assert cli.main([]) == code


def test_continuous_polls_until_sigterm(monkeypatch: pytest.MonkeyPatch, handlers: dict[int, Handler]) -> None:
    monkeypatch.setenv("POLL_INTERVAL_SECONDS", "5")
    waits: list[float | None] = []
    calls = 0

    def on_call() -> None:
        nonlocal calls
        calls += 1
        if calls == 3:  # `docker stop` during the third run
            handlers[signal.SIGTERM](signal.SIGTERM, None)

    seen = _fake_run(monkeypatch, [True, RuntimeError("transient"), True], on_call)

    def fake_wait(self: threading.Event, timeout: float | None = None) -> bool:
        waits.append(timeout)
        return self.is_set()

    monkeypatch.setattr(threading.Event, "wait", fake_wait)

    assert cli.main([]) == cli.OK
    assert calls == 3
    assert waits == [5, 5]  # slept after runs 1 and 2, not after the stop
    assert seen[0].is_set()
    assert signal.SIGINT in handlers


@pytest.mark.usefixtures("handlers")
def test_continuous_fatal_error_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLL_INTERVAL_SECONDS", "5")
    _fake_run(monkeypatch, [IngestError("qdrant down")])
    assert cli.main([]) == cli.FATAL
