"""Search the wikijs2rag Qdrant collection from the command line.

Usage:
    wikijs2rag-query "how do I reset my password?"
    wikijs2rag-query --limit 10 --collection wiki "event manager setup"
    wikijs2rag-query --show-text "access control list"

Docker:
    docker compose -f docker-compose.prod.yaml run --rm query "how do I reset my password?"
"""

import argparse
import logging
import sys
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TextIO

from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint

from wikijs2rag import __version__
from wikijs2rag.config import ConfigError, QuerySettings

# ── Styling ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Style:
    """ANSI styling, or plain text when *color* is off."""

    color: bool

    def _c(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def bold(self, t: str) -> str:
        return self._c("1", t)

    def dim(self, t: str) -> str:
        return self._c("2", t)

    def cyan(self, t: str) -> str:
        return self._c("36", t)

    def yellow(self, t: str) -> str:
        return self._c("33", t)

    def score_bar(self, score: float, width: int = 20) -> str:
        filled = round(score * width)
        bar = "█" * filled + "░" * (width - filled)
        color = "32" if score >= 0.7 else ("33" if score >= 0.5 else "31")
        return self._c(color, bar)


# ── Formatting ────────────────────────────────────────────────────────────────


def format_result(rank: int, hit: ScoredPoint, show_text: bool, text_width: int, style: Style) -> str:
    p = hit.payload or {}
    score = hit.score
    title = p.get("page_title") or p.get("page_path") or "Untitled"
    url = p.get("page_url", "")
    section = p.get("section", "")
    tags = p.get("tags") or []
    text = p.get("text", "")

    lines = [
        f"{style.bold(f'#{rank}')}  {style.bold(title)}"
        + (f"  {style.dim('›')}  {style.dim(section)}" if section else ""),  # noqa: RUF001 -- intended glyph
        f"   {style.cyan(url)}",
        f"   score {style.score_bar(score)} {style.yellow(f'{score:.3f}')}",
    ]

    if tags:
        lines.append(f"   {style.dim('tags:')} {style.dim(', '.join(tags))}")

    if show_text and text:
        wrapped = textwrap.fill(
            text.replace("\n", " "),
            width=text_width,
            initial_indent="   │ ",
            subsequent_indent="   │ ",
        )
        lines += ["", style.dim(wrapped)]

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────


def _parser() -> argparse.ArgumentParser:
    """Flags default to None, meaning "use the environment" (QuerySettings)."""
    parser = argparse.ArgumentParser(
        prog="wikijs2rag-query",
        description="Search the wikijs2rag Qdrant collection.",
        epilog="Defaults come from environment variables; see docs/configuration.md.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("query", nargs="+", help="Search query text")
    parser.add_argument("-l", "--limit", type=int, metavar="N", help="Number of results (QUERY_LIMIT, default 5)")
    parser.add_argument(
        "-c", "--collection", metavar="NAME", help="Qdrant collection (QDRANT_COLLECTION, default wiki)"
    )
    parser.add_argument(
        "--show-text", action="store_true", default=None, help="Print the matched chunk text (QUERY_SHOW_TEXT)"
    )
    parser.add_argument(
        "--min-score", type=float, metavar="FLOAT", help="Hide results below this similarity, 0-1 (QUERY_MIN_SCORE)"
    )
    parser.add_argument("--width", type=int, metavar="N", help="Text wrapping width (QUERY_WIDTH, default 100)")
    return parser


def main(argv: Sequence[str] | None = None, out: TextIO | None = None) -> int:
    out = out or sys.stdout
    args = _parser().parse_args(argv)
    try:
        settings = QuerySettings.from_env()
    except ConfigError as exc:
        sys.stderr.write(f"wikijs2rag-query: {exc}\n")
        return 2

    limit: int = settings.limit if args.limit is None else args.limit
    collection: str = args.collection or settings.qdrant.collection
    show_text: bool = settings.show_text if args.show_text is None else args.show_text
    min_score: float = settings.min_score if args.min_score is None else args.min_score
    width: int = settings.width if args.width is None else args.width

    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    style = Style(color=settings.color and out.isatty())
    query_text = " ".join(args.query)

    # ── Embed the query ───────────────────────────────────────────────────────
    try:
        embedder = settings.embedding.embedder()
        query_vector = embedder.encode([query_text])[0]
    except Exception as exc:  # httpx transport/HTTP errors or a bad embeddings response
        sys.stderr.write(f"Embedding error: {exc}\n")
        return 1

    # ── Query Qdrant ──────────────────────────────────────────────────────────
    q = settings.qdrant
    try:
        client = QdrantClient(host=q.host, port=q.port)
        hits = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=limit,
            with_payload=True,
            score_threshold=min_score if min_score > 0 else None,
        ).points
    except Exception as exc:  # qdrant-client raises several transport errors
        sys.stderr.write(f"Qdrant error: {exc}\n")
        return 1

    # ── Print results ─────────────────────────────────────────────────────────
    header = f"collection={collection}  model={settings.embedding.model}  v{__version__}"
    out.write(f"\n  {style.bold('Query:')} {query_text}\n  {style.dim(header)}\n\n")

    if not hits:
        out.write(style.dim("  No results found.") + "\n\n")
        return 0

    for rank, hit in enumerate(hits, 1):
        out.write(format_result(rank, hit, show_text, width, style) + "\n\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
