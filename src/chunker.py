"""
Text chunking for wiki pages.

Splits markdown/HTML content first by header sections, then by a sliding
word-window so that no chunk exceeds `chunk_size` words. Adjacent chunks
share `chunk_overlap` words to preserve context across boundaries.
"""

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Optional


@dataclass
class Chunk:
    text: str
    chunk_index: int
    section: str = ""


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
    # Strip YAML front-matter
    text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, flags=re.DOTALL)
    # Strip inline HTML
    text = re.sub(r"<[^>]+>", " ", text)
    # Image links -> alt text only
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # Hyperlinks -> link text only
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # Code fences -> keep content, drop the fence lines
    text = re.sub(r"```[^\n]*\n(.*?)```", r"\1", text, flags=re.DOTALL)
    # Collapse excess blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

def _split_by_headers(text: str) -> list[tuple[str, str]]:
    """Split on H1/H2/H3 markdown headers; returns [(header, body), ...]."""
    header_re = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    sections: list[tuple[str, str]] = []
    current_header = ""
    current_lines: list[str] = []

    for line in text.splitlines():
        m = header_re.match(line)
        if m:
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_header, body))
            current_header = m.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    body = "\n".join(current_lines).strip()
    if body:
        sections.append((current_header, body))

    # If no headers were found, treat the whole text as one section
    if not sections:
        sections = [("", text.strip())]

    return sections


# ---------------------------------------------------------------------------
# Sliding-window word chunker
# ---------------------------------------------------------------------------

def _window_chunks(
    header: str,
    body: str,
    chunk_size: int,
    chunk_overlap: int,
    start_index: int,
) -> list[Chunk]:
    words = body.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    idx = start_index
    pos = 0

    while pos < len(words):
        end = min(pos + chunk_size, len(words))
        window = " ".join(words[pos:end])
        prefix = f"{header}\n\n" if header else ""
        chunks.append(Chunk(text=f"{prefix}{window}".strip(), chunk_index=idx, section=header))
        idx += 1
        if end == len(words):
            break
        pos += chunk_size - chunk_overlap

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_page(
    content: str,
    content_type: str = "markdown",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """
    Split a wiki page's raw content into overlapping text chunks.

    Args:
        content:      Raw page content (markdown or HTML).
        content_type: ``"markdown"`` or ``"html"`` (case-insensitive).
        chunk_size:   Maximum words per chunk.
        chunk_overlap: Words shared between consecutive chunks.

    Returns:
        Ordered list of :class:`Chunk` objects.
    """
    if content_type.lower() == "html":
        text = _strip_html(content)
        # Minimal clean-up after HTML extraction
        text = re.sub(r"\s{2,}", " ", text).strip()
        sections = [("", text)]
    else:
        text = _clean_markdown(content)
        sections = _split_by_headers(text)

    all_chunks: list[Chunk] = []
    for header, body in sections:
        new_chunks = _window_chunks(header, body, chunk_size, chunk_overlap, len(all_chunks))
        all_chunks.extend(new_chunks)

    return all_chunks
