"""Files and images a wiki page links to.

PDFs and text files are parsed and become children of the page. Images are
staged with alt text and caption; with a VISION_MODEL configured, images of
signs, diagrams and whiteboards also get a machine-generated description.
Each file is fetched with the page's own crawl credentials first, so Wiki.js
decides whether that tier may read it. A file on a public page that guests
can't read but members can is kept as a member-tier child.
"""

import hashlib
import io
import logging

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from wikijs2rag import content
from wikijs2rag.llm import LLM, LLMError
from wikijs2rag.staging import StagedAttachment
from wikijs2rag.wiki_client import WikiClient

logger = logging.getLogger(__name__)

VISION_PROMPT = (
    "This image is from a hackerspace wiki page. If it is a sign, a diagram, a whiteboard, a screenshot or "
    "anything else with readable text or meaningful structure, transcribe its text and describe what it shows "
    "in a few sentences. Do not identify people. If it is only a photo with nothing to read, reply NONE."
)
_IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}


def pdf_pages(data: bytes) -> list[str]:
    try:
        reader = PdfReader(io.BytesIO(data))
        return [(page.extract_text() or "").strip() for page in reader.pages]
    except (PyPdfError, ValueError, OSError) as exc:
        logger.warning("  Could not parse PDF: %s", exc)
        return []


def _fetch(crawls: list[tuple[WikiClient, str]], url: str, max_bytes: int) -> tuple[bytes, str, str] | None:
    """(data, media type, tier) from the first crawl that can read *url*."""
    for wiki, tier in crawls:
        fetched = wiki.fetch_file(url, max_bytes)
        if fetched is not None:
            return fetched[0], fetched[1], tier
    return None


def collect(
    crawls: list[tuple[WikiClient, str]],
    rendered_html: str,
    base_url: str,
    max_bytes: int,
    llm: LLM | None,
) -> list[StagedAttachment]:
    """*crawls* are (client, tier) to try in order, the page's own crawl first."""
    found: list[StagedAttachment] = []
    page_tier = crawls[0][1]
    for url, kind in content.attachment_links(rendered_html, base_url):
        fetched = _fetch(crawls, url, max_bytes)
        if fetched is None:
            continue
        data, _media_type, tier = fetched
        pages = pdf_pages(data) if kind == "pdf" else [data.decode("utf-8", errors="replace")]
        if any(p.strip() for p in pages):
            found.append(StagedAttachment(url, kind, tier, pages=pages, content_hash=hashlib.sha256(data).hexdigest()))
    for image in content.images(rendered_html, base_url):
        item = StagedAttachment(image.url, "image", page_tier, alt=image.alt, caption=image.caption)
        if llm and llm.vision_model:
            fetched = _fetch(crawls, image.url, max_bytes)
            if fetched and fetched[1] in _IMAGE_TYPES:
                data, media_type, item.tier = fetched
                item.content_hash = hashlib.sha256(data).hexdigest()
                try:
                    described = llm.describe_image(data, media_type, VISION_PROMPT, image.alt + image.caption)
                except LLMError as exc:
                    logger.warning("  Could not describe %s: %s", image.url, exc)
                    described = ""
                if described and described.strip().upper().rstrip(".") != "NONE":
                    item.pages = [described.strip()]
                    item.machine_generated = True
        found.append(item)
    return found
