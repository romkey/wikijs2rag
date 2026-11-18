"""Turning a fetched Wiki.js page into clean markdown plus what it links to.

Markdown pages keep their source markdown. Every other content type (HTML,
the visual editor, AsciiDoc, a scraped page) is converted from Wiki.js's
rendered HTML. Heading anchors always come from the rendered HTML, because
that is where Wiki.js assigns the ids a browser jumps to.
"""

import re
from dataclasses import dataclass
from urllib.parse import unquote, urljoin, urlsplit

from bs4 import BeautifulSoup, Tag
from markdownify import markdownify

from wikijs2rag.staging import Link

ATTACHMENT_EXTENSIONS = {".pdf": "pdf", ".txt": "text", ".md": "text", ".csv": "text"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}


@dataclass
class Anchor:
    level: int
    text: str
    id: str


@dataclass
class ImageRef:
    url: str
    alt: str
    caption: str


def _soup(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html or "", "html.parser")
    # Wiki.js puts a "¶" permalink inside every heading.
    for el in soup.select("a.toc-anchor"):
        el.decompose()
    return soup


def heading_anchors(rendered_html: str) -> list[Anchor]:
    """Headings with ids, in document order."""
    anchors = []
    for el in _soup(rendered_html).find_all(re.compile(r"^h[1-6]$")):
        if isinstance(el, Tag) and el.get("id"):
            anchors.append(Anchor(int(el.name[1]), normalize_heading(el.get_text(" ")), str(el["id"])))
    return anchors


def normalize_heading(text: str) -> str:
    text = re.sub(r"[*_`~]|\[([^\]]*)\]\([^)]*\)", lambda m: m.group(1) or "", text)
    return " ".join(text.lower().split())


def slugify(text: str) -> str:
    """Best guess at a Wiki.js heading id, for headings missing from the rendered HTML."""
    slug = re.sub(r"[^\w\s-]", "", normalize_heading(text)).strip()
    slug = re.sub(r"[\s_]+", "-", slug)
    return f"h-{slug}" if slug[:1].isdigit() else slug


def html_to_markdown(html: str) -> str:
    soup = _soup(html)
    for el in soup.select("script, style, nav, header, footer, aside, [role=navigation]"):
        el.decompose()
    text: str = markdownify(str(soup), heading_style="ATX", bullets="-", code_language_callback=_code_language)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _code_language(el: Tag) -> str | None:
    """markdownify passes the <pre>; the language class is usually on its <code>."""
    code = el.find("code")
    classes = [str(c) for c in (el.get("class") or [])]
    if isinstance(code, Tag):
        classes += [str(c) for c in (code.get("class") or [])]
    for cls in classes:
        if str(cls).startswith("language-"):
            return str(cls)[len("language-") :]
    return None


def page_markdown(content_type: str, content: str, rendered_html: str) -> str:
    if content_type.lower() == "markdown" and content.strip():
        text = re.sub(r"\A---\s*\n.*?\n---\s*\n", "", content, flags=re.DOTALL)  # front matter
        return text.strip()
    return html_to_markdown(rendered_html or content)


def _internal_target(base_url: str, href: str) -> tuple[str, str] | None:
    """(path, anchor) when *href* points at a page on this wiki."""
    base = urlsplit(base_url)
    url = urlsplit(urljoin(base_url + "/", href))
    if url.scheme not in ("http", "https") or url.netloc != base.netloc:
        return None
    path = unquote(url.path)
    prefix = base.path.rstrip("/")
    if prefix and path.startswith(prefix):
        path = path[len(prefix) :]
    return path.strip("/"), url.fragment


def extract_links(rendered_html: str, base_url: str) -> list[Link]:
    """Outbound links, split into internal wiki links and external links.
    Links to files (attachments, images) and in-page anchors are left out."""
    result: list[Link] = []
    seen: set[tuple[str, str, str]] = set()
    for a in _soup(rendered_html).find_all("a", href=True):
        href = str(a["href"]).strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        label = " ".join(a.get_text(" ").split())
        internal = _internal_target(base_url, href)
        if internal is not None:
            path, anchor = internal
            if _extension(path) or not path:
                continue
            link = Link("internal", path, anchor, label)
        elif urlsplit(href).scheme in ("http", "https"):
            link = Link("external", href, "", label)
        else:
            continue
        key = (link.kind, link.target, link.anchor)
        if key not in seen:
            seen.add(key)
            result.append(link)
    return result


def _extension(path: str) -> str:
    match = re.search(r"(\.[A-Za-z0-9]+)$", path)
    return match.group(1).lower() if match else ""


def attachment_links(rendered_html: str, base_url: str) -> list[tuple[str, str]]:
    """(absolute URL, kind) for PDFs and text files on this wiki that the page links to."""
    found: dict[str, str] = {}
    for a in _soup(rendered_html).find_all("a", href=True):
        href = str(a["href"]).strip()
        internal = _internal_target(base_url, href)
        if internal is None:
            continue
        kind = ATTACHMENT_EXTENSIONS.get(_extension(internal[0]))
        if kind:
            found.setdefault(urljoin(base_url + "/", href.split("#")[0]), kind)
    return list(found.items())


def images(rendered_html: str, base_url: str) -> list[ImageRef]:
    """Images hosted on this wiki, with alt text and caption (figcaption or title)."""
    result: list[ImageRef] = []
    seen: set[str] = set()
    for img in _soup(rendered_html).find_all("img", src=True):
        src = str(img["src"]).strip()
        internal = _internal_target(base_url, src)
        if internal is None or _extension(internal[0]) not in IMAGE_EXTENSIONS:
            continue
        url = urljoin(base_url + "/", src)
        if url in seen:
            continue
        seen.add(url)
        caption = ""
        figure = img.find_parent("figure")
        if figure and (cap := figure.find("figcaption")):
            caption = " ".join(cap.get_text(" ").split())
        result.append(ImageRef(url, str(img.get("alt") or ""), caption or str(img.get("title") or "")))
    return result
