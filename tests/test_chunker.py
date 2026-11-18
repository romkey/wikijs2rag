from wikijs2rag.chunker import chunk_markdown, count_tokens
from wikijs2rag.content import Anchor


def test_splits_by_heading_with_anchor_and_chain() -> None:
    md = "Intro.\n\n# Top\n\nA.\n\n## Mid\n\nB.\n\n### Deep\n\nC.\n\n## Other\n\nD."
    anchors = [Anchor(1, "top", "top"), Anchor(2, "mid", "mid-1"), Anchor(3, "deep", "deep")]
    chunks = chunk_markdown(md, anchors)
    assert [(c.text, c.headings, c.anchor) for c in chunks] == [
        ("Intro.", [], ""),
        ("A.", ["Top"], "top"),
        ("B.", ["Top", "Mid"], "mid-1"),
        ("C.", ["Top", "Mid", "Deep"], "deep"),
        ("D.", ["Top", "Other"], "other"),  # no rendered anchor: slugified
    ]
    assert [c.chunk_index for c in chunks] == [0, 1, 2, 3, 4]


def test_code_blocks_are_never_split_and_hide_headings() -> None:
    code = "```yaml\n# not a heading\n" + "\n".join(f"key{i}: value" for i in range(400)) + "\n```"
    chunks = chunk_markdown(f"# H\n\n{code}\n\nAfter.", max_tokens=100, target_tokens=60, overlap_tokens=10)
    assert chunks[0].text == code
    assert chunks[0].headings == ["H"]
    assert chunks[-1].text == "After."


def test_small_blocks_are_packed_to_target() -> None:
    md = "\n\n".join(f"Paragraph {i} has some words." for i in range(30))
    chunks = chunk_markdown(md, target_tokens=60, max_tokens=100, overlap_tokens=5)
    assert 1 < len(chunks) < 30
    assert all(count_tokens(c.text) <= 100 for c in chunks)


def test_long_prose_splits_with_overlap() -> None:
    words = [f"w{i}" for i in range(300)]
    chunks = chunk_markdown(" ".join(words), target_tokens=60, max_tokens=100, overlap_tokens=20)
    assert len(chunks) > 3
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert set(first) & set(second)  # overlap
    assert all(count_tokens(c.text) <= 100 for c in chunks)


def test_long_table_repeats_caption_and_header() -> None:
    rows = "\n".join(f"| item {i} | value {i} |" for i in range(80))
    md = f"Table 1: inventory\n\n| Name | Value |\n|---|---|\n{rows}"
    chunks = chunk_markdown(md, target_tokens=60, max_tokens=120, overlap_tokens=5)
    table_chunks = [c for c in chunks if "| Name | Value |" in c.text]
    assert len(table_chunks) > 1
    assert all(c.text.startswith("Table 1: inventory\n| Name | Value |\n|---|---|") for c in table_chunks)


def test_long_list_splits_by_item() -> None:
    md = "\n".join(f"- item {i} with a few words\n  continued" for i in range(60))
    chunks = chunk_markdown(md, target_tokens=60, max_tokens=100, overlap_tokens=5)
    assert len(chunks) > 1
    assert all(c.text.startswith("- item") for c in chunks)


def test_inline_markup_cleaned() -> None:
    [chunk] = chunk_markdown("See [the docs](/docs) and ![a sign](/s.png) <span>x</span> <kbd>K</kbd>.")
    assert chunk.text == "See the docs and [image: a sign] x <kbd>K</kbd>."


def test_empty_and_heading_only() -> None:
    assert chunk_markdown("") == []
    assert chunk_markdown("# Only a heading") == []
    assert chunk_markdown("# T\n\n1. one\n2. two")[0].text == "1. one\n2. two"


def test_nested_fences_stay_one_block() -> None:
    inner = "````markdown\n# Not a heading\n```\ncode\n```\nstill inside\n````"
    chunks = chunk_markdown(f"# Real\n\n{inner}\n\nAfter.")
    assert [c.headings for c in chunks] == [["Real"]]
    assert chunks[0].text == f"{inner}\n\nAfter."
    [chunk] = chunk_markdown("```\n```python\nnot a close\n```")
    assert chunk.text == "```\n```python\nnot a close\n```"
