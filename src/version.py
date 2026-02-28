"""
Single source of version truth for Python code.

The VERSION file lives next to this module in the Docker image (/app/VERSION)
and at the repo root locally.  We look in both places so it works everywhere.
"""

from pathlib import Path


def _read_version() -> str:
    for candidate in [
        Path(__file__).with_name("VERSION"),       # /app/VERSION  (Docker)
        Path(__file__).parent.parent / "VERSION",  # repo root     (local dev)
    ]:
        if candidate.exists():
            return candidate.read_text().strip()
    return "unknown"


__version__ = _read_version()
