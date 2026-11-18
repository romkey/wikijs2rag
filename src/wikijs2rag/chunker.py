"""Structure-first chunking of markdown.

The page is split on headings first, then each section is packed into chunks
of about ``target_tokens``. Blocks are never split mid-block when they fit
under ``max_tokens``; code blocks (and so YAML) are never split at all.
Oversized prose is split into overlapping windows, oversized lists by item,
and oversized tables by rows with the caption and header row repeated on each
piece. Each chunk knows its heading chain and the anchor of its nearest
heading, so a citation can open the page at that heading.
"""

import hashlib
import re
from dataclasses import dataclass, field

from wikijs2rag.content import Anchor, normalize_heading, slugify

_FENCE = re.compile(r"^\s*(```+|~~~+)")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_TOKEN = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    """A tokenizer-free estimate: words and punctuation marks."""
    return len(_TOKEN.findall(text))


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass
class Chunk:
    text: str
    chunk_index: int
    headings: list[str] = field(default_factory=list)
    anchor: str = ""

    @property
    def content_hash(self) -> str:
        return content_hash(self.text)


@dataclass
class _Block:
    kind: str  # para | code | table | list
    text: str


def _clean_inline(text: str) -> str:
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", lambda m: f"[image: {m.group(1)}]" if m.group(1) else "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    return re.sub(r"<(?!/?(?:kbd|sup|sub)\b)[^>\n]+>", "", text)


def _blocks(lines: list[str]) -> list[_Block]:
    blocks: list[_Block] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if m := _FENCE.match(line):
            fence = m.group(1)
            end = i + 1
            while end < len(lines) and not lines[end].strip().startswith(fence):
                end += 1
            blocks.append(_Block("code", "\n".join(lines[i : end + 1]).strip()))
            i = end + 1
        elif line.lstrip().startswith("|"):
            end = i
            while end < len(lines) and lines[end].lstrip().startswith("|"):
                end += 1
            blocks.append(_Block("table", _clean_inline("\n".join(lines[i:end]))))
            i = end
        elif _LIST_ITEM.match(line):
            end = i + 1
            while end < len(lines):
                nxt = lines[end]
                if (
                    _LIST_ITEM.match(nxt)
                    or (nxt.startswith((" ", "\t")) and nxt.strip())
                    or (not nxt.strip() and end + 1 < len(lines) and lines[end + 1].startswith((" ", "\t")))
                ):
                    end += 1
                else:
                    break
            blocks.append(_Block("list", _clean_inline("\n".join(lines[i:end]).rstrip())))
            i = end
        else:
            end = i + 1
            while (
                end < len(lines)
                and lines[end].strip()
                and not _FENCE.match(lines[end])
                and not lines[end].lstrip().startswith("|")
                and not _LIST_ITEM.match(lines[end])
            ):
                end += 1
            para = _clean_inline("\n".join(lines[i:end])).strip()
            if para:
                blocks.append(_Block("para", para))
            i = end
    return blocks


@dataclass
class _Section:
    headings: list[str]
    anchor: str
    lines: list[str]


def _sections(markdown: str, anchors: list[Anchor]) -> list[_Section]:
    sections = [_Section([], "", [])]
    chain: dict[int, str] = {}
    next_anchor = 0
    in_fence: str | None = None
    for line in markdown.splitlines():
        if m := _FENCE.match(line):
            fence = m.group(1)
            if in_fence is None:
                in_fence = fence[:3]
            elif line.strip().startswith(in_fence):
                in_fence = None
        heading = None if in_fence else _HEADING.match(line)
        if not heading:
            sections[-1].lines.append(line)
            continue
        level = len(heading.group(1))
        title = _clean_inline(heading.group(2)).strip()
        chain = {lvl: t for lvl, t in chain.items() if lvl < level}
        chain[level] = title
        wanted = normalize_heading(title)
        anchor = slugify(title)
        for j in range(next_anchor, len(anchors)):
            if anchors[j].text == wanted:
                anchor = anchors[j].id
                next_anchor = j + 1
                break
        sections.append(_Section([chain[k] for k in sorted(chain)], anchor, []))
    return sections


def _split_prose(text: str, max_tokens: int, overlap: int) -> list[str]:
    words = text.split()
    # Words are ~1.3 tokens; work in words, sized to stay under max_tokens.
    size = max(1, int(max_tokens / 1.3))
    step = max(1, size - int(overlap / 1.3))
    pieces = []
    pos = 0
    while pos < len(words):
        pieces.append(" ".join(words[pos : pos + size]))
        if pos + size >= len(words):
            break
        pos += step
    return pieces


def _split_lines(lines: list[str], header: list[str], max_tokens: int) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    for line in lines:
        if current and count_tokens("\n".join(header + current + [line])) > max_tokens:
            pieces.append("\n".join(header + current))
            current = []
        current.append(line)
    if current:
        pieces.append("\n".join(header + current))
    return pieces


def _split_table(text: str, caption: str, max_tokens: int) -> list[str]:
    rows = text.splitlines()
    header = rows[:2] if len(rows) > 1 and re.match(r"^\s*\|[\s:|-]+\|\s*$", rows[1]) else rows[:1]
    if caption:
        header = [caption, *header]
    return _split_lines(rows[len(header) - (1 if caption else 0) :], header, max_tokens)


def _split_list(text: str, max_tokens: int) -> list[str]:
    items: list[str] = []
    for line in text.splitlines():
        if _LIST_ITEM.match(line) or not items:
            items.append(line)
        else:
            items[-1] += "\n" + line
    return _split_lines(items, [], max_tokens)


def _section_pieces(blocks: list[_Block], target: int, max_tokens: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            pieces.append("\n\n".join(current))
            current.clear()

    for n, block in enumerate(blocks):
        tokens = count_tokens(block.text)
        if tokens > max_tokens and block.kind != "code":
            flush()
            if block.kind == "table":
                prev = blocks[n - 1] if n else None
                caption = prev.text if prev and prev.kind == "para" and count_tokens(prev.text) < 60 else ""
                pieces.extend(_split_table(block.text, caption, max_tokens))
            elif block.kind == "list":
                pieces.extend(_split_list(block.text, max_tokens))
            else:
                pieces.extend(_split_prose(block.text, max_tokens, overlap))
            continue
        if current and count_tokens("\n\n".join([*current, block.text])) > target:
            flush()
        current.append(block.text)
    flush()
    return pieces


def chunk_markdown(
    markdown: str,
    anchors: list[Anchor] | None = None,
    target_tokens: int = 400,
    max_tokens: int = 800,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    """Split *markdown* into chunks; see the module docstring."""
    chunks: list[Chunk] = []
    for section in _sections(markdown, anchors or []):
        for piece in _section_pieces(_blocks(section.lines), target_tokens, max_tokens, overlap_tokens):
            chunks.append(Chunk(piece, len(chunks), section.headings, section.anchor))
    return chunks
