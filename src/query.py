"""
wiki2rag query tool

Search the Qdrant collection from the command line.

Usage:
    python query.py "how do I reset my password?"
    python query.py --limit 10 --collection wiki "event manager setup"
    python query.py --show-text "access control list"

Docker:
    docker compose run --rm query "how do I reset my password?"
"""

import argparse
import os
import sys
import textwrap

from embedder import build_embedder
from qdrant_client import QdrantClient


# ── ANSI helpers ──────────────────────────────────────────────────────────────

def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

_USE_COLOR = _supports_color()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def bold(t: str)   -> str: return _c("1", t)
def dim(t: str)    -> str: return _c("2", t)
def cyan(t: str)   -> str: return _c("36", t)
def yellow(t: str) -> str: return _c("33", t)
def green(t: str)  -> str: return _c("32", t)


# ── Formatting ────────────────────────────────────────────────────────────────

def _score_bar(score: float, width: int = 20) -> str:
    filled = round(score * width)
    bar = "█" * filled + "░" * (width - filled)
    color = "32" if score >= 0.7 else ("33" if score >= 0.5 else "31")
    return _c(color, bar)


def _format_result(rank: int, hit, show_text: bool, text_width: int) -> str:
    p     = hit.payload
    score = hit.score
    title   = p.get("page_title") or p.get("page_path") or "Untitled"
    url     = p.get("page_url", "")
    section = p.get("section", "")
    tags    = p.get("tags") or []
    text    = p.get("text", "")

    lines = [
        f"{bold(f'#{rank}')}  {bold(title)}"
        + (f"  {dim('›')}  {dim(section)}" if section else ""),
        f"   {cyan(url)}",
        f"   score {_score_bar(score)} {yellow(f'{score:.3f}')}",
    ]

    if tags:
        lines.append(f"   {dim('tags:')} {dim(', '.join(tags))}")

    if show_text and text:
        wrapped = textwrap.fill(
            text.replace("\n", " "),
            width=text_width,
            initial_indent="   │ ",
            subsequent_indent="   │ ",
        )
        lines.append("")
        lines.append(dim(wrapped))

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search the wiki2rag Qdrant collection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Environment variables (override with flags or .env):
              QDRANT_HOST        default: localhost
              QDRANT_PORT        default: 6333
              QDRANT_COLLECTION  default: wiki
              EMBEDDING_BACKEND  default: local
              EMBEDDING_MODEL    default: all-MiniLM-L6-v2
        """),
    )
    parser.add_argument("query", nargs="+", help="Search query text")
    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=int(os.environ.get("QUERY_LIMIT", "5")),
        metavar="N",
        help="Number of results to return (default: 5)",
    )
    parser.add_argument(
        "-c", "--collection",
        default=os.environ.get("QDRANT_COLLECTION", "wiki"),
        metavar="NAME",
        help="Qdrant collection name (default: wiki)",
    )
    parser.add_argument(
        "--show-text",
        action="store_true",
        default=os.environ.get("QUERY_SHOW_TEXT", "").lower() in ("1", "true", "yes"),
        help="Print the matched chunk text under each result",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=float(os.environ.get("QUERY_MIN_SCORE", "0.0")),
        metavar="FLOAT",
        help="Hide results below this cosine similarity (0–1)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=int(os.environ.get("QUERY_WIDTH", "100")),
        metavar="N",
        help="Terminal width for text wrapping (default: 100)",
    )
    args = parser.parse_args()

    query_text = " ".join(args.query)

    # ── Embed the query ───────────────────────────────────────────────────────
    try:
        embedder = build_embedder()
    except Exception as exc:
        print(f"Error loading embedding model: {exc}", file=sys.stderr)
        sys.exit(1)

    vectors = embedder.encode([query_text])
    query_vector = vectors[0]

    # ── Query Qdrant ──────────────────────────────────────────────────────────
    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", "6333"))

    try:
        client = QdrantClient(host=host, port=port)
        # query_points() is the current API (qdrant-client >= 1.10);
        # fall back to the legacy search() for older installs.
        threshold = args.min_score if args.min_score > 0 else None
        if hasattr(client, "query_points"):
            response = client.query_points(
                collection_name=args.collection,
                query=query_vector,
                limit=args.limit,
                with_payload=True,
                score_threshold=threshold,
            )
            hits = response.points
        else:
            hits = client.search(
                collection_name=args.collection,
                query_vector=query_vector,
                limit=args.limit,
                with_payload=True,
                score_threshold=threshold,
            )
    except Exception as exc:
        print(f"Qdrant error: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Print results ─────────────────────────────────────────────────────────
    print()
    print(f"  {bold('Query:')} {query_text}")
    model_name = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    print(f"  {dim(f'collection={args.collection}  model={model_name}')}")
    print()

    if not hits:
        print(dim("  No results found."))
        print()
        sys.exit(0)

    for rank, hit in enumerate(hits, 1):
        print(_format_result(rank, hit, args.show_text, args.width))
        print()


if __name__ == "__main__":
    main()
