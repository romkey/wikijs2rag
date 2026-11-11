from wikijs2rag.chunker import (
    _clean_markdown,
    _content_hash,
    _extract_atomic_blocks,
    _split_by_headers,
    _strip_html,
    _window_chunks,
    chunk_page,
)


def test_strip_html_extracts_text() -> None:
    assert _strip_html("<p>Hello <b>world</b></p>") == "Hello  world"


def test_clean_markdown_removes_frontmatter_links_images_and_html() -> None:
    md = "---\ntitle: x\n---\n# H\n![alt](img.png) [link](http://x) <span>t</span>\n\n\n\nend"
    out = _clean_markdown(md)
    assert "title: x" not in out
    assert "alt" in out
    assert "img.png" not in out
    assert "link" in out
    assert "http://x" not in out
    assert "<span>" not in out
    assert "\n\n\n" not in out


def test_clean_markdown_unwraps_fenced_code() -> None:
    assert _clean_markdown("```python\nprint(1)\n```") == "print(1)"


def test_split_by_headers_builds_breadcrumbs() -> None:
    text = "intro\n# A\na body\n## B\nb body\n### C\nc body\n## D\nd body"
    sections = _split_by_headers(text)
    assert sections == [
        ("", "intro", ""),
        ("A", "a body", "A"),
        ("B", "b body", "A > B"),
        ("C", "c body", "A > B > C"),
        ("D", "d body", "A > D"),
    ]


def test_split_by_headers_skips_empty_sections_and_handles_empty_text() -> None:
    assert _split_by_headers("# A\n# B\nbody") == [("B", "body", "B")]
    assert _split_by_headers("   ") == [("", "", "")]


def test_extract_atomic_blocks_no_atomic() -> None:
    assert _extract_atomic_blocks("just prose") == [("just prose", False)]


def test_extract_atomic_blocks_table_list_code() -> None:
    body = "before\n| a | b |\n| - | - |\n| 1 | 2 |\nmiddle\n1. one\n2. two\n    code line\n    more code\nafter"
    segs = _extract_atomic_blocks(body)
    assert segs[0] == ("before", False)
    assert segs[1][1] is True
    assert segs[1][0].startswith("| a | b |")
    assert segs[2] == ("middle", False)
    assert any(atomic and "1. one" in text for text, atomic in segs)
    assert any(atomic and "code line" in text for text, atomic in segs)
    assert segs[-1] == ("after", False)


def test_extract_atomic_blocks_merges_overlapping_spans() -> None:
    # An ordered list item with an indented continuation is matched by both
    # the list and code-block patterns; the spans must merge into one block.
    body = "1. step\n    indented detail\n"
    segs = _extract_atomic_blocks(body)
    assert segs == [("1. step\n    indented detail", True)]


def test_window_chunks() -> None:
    words = [str(i) for i in range(10)]
    assert _window_chunks([], 4, 1) == []
    assert _window_chunks(words, 4, 1) == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]
    assert _window_chunks(words, 20, 5) == [" ".join(words)]


def test_content_hash_normalizes_case_and_whitespace() -> None:
    assert _content_hash("Hello   World") == _content_hash("hello world")
    assert len(_content_hash("x")) == 16


def test_chunk_page_markdown() -> None:
    md = "# Title\n" + " ".join(f"w{i}" for i in range(30)) + "\n## Table\n| a |\n| b |\n"
    chunks = chunk_page(md, chunk_size=10, chunk_overlap=2)
    prose = [c for c in chunks if c.section == "Title"]
    assert len(prose) == 4
    assert all(c.text.startswith("Title\n\n") for c in prose)
    table = [c for c in chunks if c.section == "Table"]
    assert len(table) == 1
    assert table[0].section_breadcrumb == "Title > Table"
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.content_hash for c in chunks)


def test_chunk_page_html() -> None:
    chunks = chunk_page("<div><p>Hello</p>   <p>there</p></div>", content_type="HTML")
    assert len(chunks) == 1
    assert chunks[0].text == "Hello there"
    assert chunks[0].section == ""


def test_chunk_page_empty() -> None:
    assert chunk_page("") == []


def test_atomic_blocks_at_end_of_section_stay_whole() -> None:
    # Regression: the final row of a table / item of a list at the very end of
    # a section has no trailing newline and used to be split off as prose.
    assert _extract_atomic_blocks("| a |\n| b |") == [("| a |\n| b |", True)]
    assert _extract_atomic_blocks("1. one\n2. two") == [("1. one\n2. two", True)]
