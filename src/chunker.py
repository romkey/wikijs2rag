"""
Text chunking for wiki pages.

Splits markdown/HTML content into retrieval-friendly chunks with:
  - Header-based section splitting (preserves document structure)
  - Atomic block detection (tables, ordered lists, code blocks kept whole)
  - Sliding word-window with overlap for remaining prose
  - Enriched context string per chunk (title + section breadcrumb) for LLM use
  - Content hashing for downstream deduplication
"""

import hashlib
import re
from dataclasses import dataclass
from html.parser import HTMLParser


@dataclass
class Chunk:
    text: str
    chunk_index: int
    section: str = ""
    section_breadcrumb: str = ""
    content_hash: str = ""


# ---------------------------------------------------------------------------
# Content cleaning
# ---------------------------------------------------------------------------

def _strip_html(text: str) -> str:
    class _Extractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts: list[str] = []

        def handle_data(self, data: str) -> None:
            self.parts.append(data)

    parser = _Extractor()
    parser.feed(text)
    return " ".join(parser.parts)


def _clean_markdown(text: str) -> str:
    text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"```[^\n]*\n(.*?)```", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Section splitting with breadcrumb tracking
# ---------------------------------------------------------------------------

def _split_by_headers(text: str) -> list[tuple[str, str, str]]:
    """
    Split on markdown headers; returns [(header, body, breadcrumb), ...].

    The breadcrumb is a " > "-joined path like "Setup > Installation > Docker"
    that tracks the H1/H2/H3 nesting hierarchy.
    """
    header_re = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    sections: list[tuple[str, str, str]] = []

    # Track headers at each level for breadcrumb
    level_headers: dict[int, str] = {}
    current_header = ""
    current_level = 0
    current_lines: list[str] = []

    def _breadcrumb() -> str:
        parts = []
        for lvl in sorted(level_headers):
            if level_headers[lvl]:
                parts.append(level_headers[lvl])
        return " > ".join(parts)

    for line in text.splitlines():
        m = header_re.match(line)
        if m:
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_header, body, _breadcrumb()))

            level = len(m.group(1))
            title = m.group(2).strip()
            current_header = title
            current_level = level
            level_headers[level] = title
            for lvl in list(level_headers):
                if lvl > level:
                    del level_headers[lvl]
            current_lines = []
        else:
            current_lines.append(line)

    body = "\n".join(current_lines).strip()
    if body:
        sections.append((current_header, body, _breadcrumb()))

    if not sections:
        sections = [("", text.strip(), "")]

    return sections


# ---------------------------------------------------------------------------
# Atomic block extraction (tables, ordered lists, code blocks)
# ---------------------------------------------------------------------------

_TABLE_RE = re.compile(
    r"((?:^\|.+\|[ \t]*\n)+)",
    re.MULTILINE,
)

_ORDERED_LIST_RE = re.compile(
    r"((?:^\d+[.)]\s+.+\n(?:[ \t]+.+\n)*)+)",
    re.MULTILINE,
)

_CODE_BLOCK_RE = re.compile(
    r"(^(?:    |\t).+(?:\n(?:    |\t).+)*)",
    re.MULTILINE,
)


def _extract_atomic_blocks(body: str) -> list[tuple[str, bool]]:
    """
    Split *body* into segments, tagging each as atomic (keep whole) or prose
    (eligible for word-window splitting).

    Returns [(text, is_atomic), ...] in document order.
    """
    atomic_spans: list[tuple[int, int]] = []
    for pattern in (_TABLE_RE, _ORDERED_LIST_RE, _CODE_BLOCK_RE):
        for m in pattern.finditer(body):
            atomic_spans.append((m.start(), m.end()))

    if not atomic_spans:
        return [(body, False)]

    # Merge overlapping spans and sort
    atomic_spans.sort()
    merged: list[tuple[int, int]] = [atomic_spans[0]]
    for start, end in atomic_spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    segments: list[tuple[str, bool]] = []
    pos = 0
    for start, end in merged:
        if pos < start:
            prose = body[pos:start].strip()
            if prose:
                segments.append((prose, False))
        block = body[start:end].strip()
        if block:
            segments.append((block, True))
        pos = end

    if pos < len(body):
        trailing = body[pos:].strip()
        if trailing:
            segments.append((trailing, False))

    return segments


# ---------------------------------------------------------------------------
# Sliding-window word chunker
# ---------------------------------------------------------------------------

def _window_chunks(
    words: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split *words* into overlapping windows, returning text strings."""
    if not words:
        return []

    chunks: list[str] = []
    pos = 0
    while pos < len(words):
        end = min(pos + chunk_size, len(words))
        chunks.append(" ".join(words[pos:end]))
        if end == len(words):
            break
        pos += chunk_size - chunk_overlap

    return chunks


def _content_hash(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_page(
    content: str,
    content_type: str = "markdown",
    chunk_size: int = 256,
    chunk_overlap: int = 50,
    page_title: str = "",
    page_description: str = "",
) -> list[Chunk]:
    """
    Split a wiki page's raw content into overlapping text chunks.

    Args:
        content:          Raw page content (markdown or HTML).
        content_type:     ``"markdown"`` or ``"html"`` (case-insensitive).
        chunk_size:       Maximum words per chunk.
        chunk_overlap:    Words shared between consecutive chunks.
        page_title:       Page title (used to build enriched context).
        page_description: Page description/excerpt (included in context).

    Returns:
        Ordered list of :class:`Chunk` objects.
    """
    if content_type.lower() == "html":
        text = _strip_html(content)
        text = re.sub(r"\s{2,}", " ", text).strip()
        sections = [("", text, "")]
    else:
        text = _clean_markdown(content)
        sections = _split_by_headers(text)

    all_chunks: list[Chunk] = []

    for header, body, breadcrumb in sections:
        segments = _extract_atomic_blocks(body)
        prefix = f"{header}\n\n" if header else ""

        for segment_text, is_atomic in segments:
            if is_atomic:
                chunk_text = f"{prefix}{segment_text}".strip()
                all_chunks.append(Chunk(
                    text=chunk_text,
                    chunk_index=len(all_chunks),
                    section=header,
                    section_breadcrumb=breadcrumb,
                    content_hash=_content_hash(chunk_text),
                ))
            else:
                words = segment_text.split()
                windows = _window_chunks(words, chunk_size, chunk_overlap)
                for window_text in windows:
                    chunk_text = f"{prefix}{window_text}".strip()
                    all_chunks.append(Chunk(
                        text=chunk_text,
                        chunk_index=len(all_chunks),
                        section=header,
                        section_breadcrumb=breadcrumb,
                        content_hash=_content_hash(chunk_text),
                    ))

    return all_chunks
