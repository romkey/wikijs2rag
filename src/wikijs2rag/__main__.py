"""Ingestion service: `wikijs2rag` or `python -m wikijs2rag`.

Syncs the wiki into the staging database and the tiered Qdrant collections,
then (unless POLL_INTERVAL_SECONDS is 0) sleeps and repeats until stopped.
Each run is incremental unless a full reconcile is due
(RECONCILE_INTERVAL_SECONDS) or --full is given. --reindex rebuilds the
collections from staging without fetching anything, then exits.
REINGEST_ON_START=true makes the first run re-fetch and re-index every page.
"""

import argparse
import logging
import signal
import sys
import threading
from collections.abc import Sequence
from dataclasses import replace
from types import FrameType

from wikijs2rag import __version__
from wikijs2rag.config import ConfigError, Settings
from wikijs2rag.ingest import IngestError, run

logger = logging.getLogger("wikijs2rag")

# Exit codes
OK = 0
FATAL = 1  # Qdrant or the wiki unreachable; bad configuration exits 2 via argparse
PAGE_ERRORS = 2  # one-shot run finished, but some pages failed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wikijs2rag", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--full", action="store_true", help="Make the first run a full reconcile")
    parser.add_argument("--reindex", action="store_true", help="Rebuild the collections from staging, then exit")
    args = parser.parse_args(argv)

    try:
        settings = Settings.from_env()
    except ConfigError as e:
        parser.exit(2, f"wikijs2rag: {e}\n")

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )

    # `docker stop` sends SIGTERM: finish the current page, then exit.
    stop = threading.Event()

    def on_signal(signum: int, frame: FrameType | None) -> None:
        logger.info("Received signal %d, stopping after the current page.", signum)
        stop.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    interval = settings.poll_interval
    # The first run (until one finishes) re-indexes everything if asked to.
    first = replace(settings, force_reingest=True) if settings.reingest_on_start else settings
    if interval <= 0 or args.reindex:
        # One-shot mode (e.g. `docker compose run --rm app`)
        try:
            ok = run(first, stop, full=True if args.full else None, reindex=args.reindex)
        except IngestError as exc:
            logger.error("%s", exc)
            return FATAL
        return OK if ok else PAGE_ERRORS

    logger.info("wikijs2rag %s running in continuous mode (poll every %d s).", __version__, interval)
    full: bool | None = True if args.full else None
    while not stop.is_set():
        try:
            run(first, stop, full=full)
            full = None
            first = settings
        except IngestError as exc:
            logger.error("%s", exc)
            return FATAL
        except Exception:  # keep the service alive; the next poll may succeed
            logger.exception("Unexpected error during ingestion run.")

        if not stop.is_set():
            logger.info("Sleeping %d s until next poll…", interval)
            stop.wait(interval)

    logger.info("Stopped.")
    return OK


if __name__ == "__main__":
    sys.exit(main())
