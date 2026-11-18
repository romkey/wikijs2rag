from wikijs2rag import content
from wikijs2rag.staging import Link

BASE = "https://wiki.test"

RENDER = """
<h1 id="intro"><a class="toc-anchor" href="#intro">¶</a> Intro <em>here</em></h1>
<h2 id="h-2-step">2 Step</h2><h3>No id</h3>
<a href="/other/page#part">Other</a>
<a href="https://wiki.test/fr/autre">Autre</a>
<a href="https://example.com/x">Ext</a>
<a href="#local">local</a> <a href="mailto:a@b.c">mail</a> <a href="ftp://f">ftp</a>
<a href="/files/manual.pdf">PDF</a> <a href="/files/manual.pdf#p2">PDF again</a> <a href="/n.txt">notes</a>
<a href="https://example.com/x">Ext twice</a>
<figure><img src="/img/sign.png" alt="Sign"><figcaption>Door sign</figcaption></figure>
<img src="/img/d.jpg" title="Diagram"> <img src="https://cdn.example.com/x.png"> <img src="/img/sign.png">
"""


def test_heading_anchors_skip_permalink_and_missing_ids() -> None:
    assert [(a.level, a.text, a.id) for a in content.heading_anchors(RENDER)] == [
        (1, "intro here", "intro"),
        (2, "2 step", "h-2-step"),
    ]


def test_slugify_matches_wikijs_style() -> None:
    assert content.slugify("Eye Protection!") == "eye-protection"
    assert content.slugify("3D Printers") == "h-3d-printers"


def test_extract_links() -> None:
    assert content.extract_links(RENDER, BASE) == [
        Link("internal", "other/page", "part", "Other"),
        Link("internal", "fr/autre", "", "Autre"),
        Link("external", "https://example.com/x", "", "Ext"),
    ]


def test_attachments_and_images() -> None:
    assert content.attachment_links(RENDER, BASE) == [
        (f"{BASE}/files/manual.pdf", "pdf"),
        (f"{BASE}/n.txt", "text"),
    ]
    imgs = content.images(RENDER, BASE)
    assert [(i.url, i.alt, i.caption) for i in imgs] == [
        (f"{BASE}/img/sign.png", "Sign", "Door sign"),
        (f"{BASE}/img/d.jpg", "", "Diagram"),
    ]


def test_page_markdown() -> None:
    assert content.page_markdown("markdown", "---\ntitle: x\n---\n# Hi\n", "") == "# Hi"
    html = '<h2 id="a">A</h2><p>Text</p><pre><code class="language-yaml">k: v</code></pre><script>x()</script>'
    md = content.page_markdown("html", "", html)
    assert md.startswith("## A\n\nText")
    assert "```yaml\nk: v\n```" in md
    assert "x()" not in md
    assert content.page_markdown("markdown", " ", "<p>Fallback</p>") == "Fallback"


def test_wiki_under_a_subpath() -> None:
    links = content.extract_links('<a href="/wiki/page">p</a><a href="/elsewhere">e</a>', "https://h.test/wiki")
    assert [link.target for link in links] == ["page", "elsewhere"]
