"""Ingest-time enrichment: page kind, summary, topics, per-chunk context, and
the exclusion classifier.

With an LLM configured (LLM_MODEL) each task asks the model, and the answer
is cached by content hash. Without one, every task but the exclusion
classifier has a deterministic fallback; the classifier has none, so member
pages can only be indexed with an LLM (config.py enforces that).
"""

import logging
import re
from importlib import resources
from pathlib import Path

from wikijs2rag.chunker import count_tokens
from wikijs2rag.llm import LLM, LLMError
from wikijs2rag.staging import StagedPage

logger = logging.getLogger(__name__)

PAGE_KINDS = (
    "policy",
    "procedure",
    "training",
    "equipment",
    "build-instructions",
    "project",
    "area",
    "meeting-notes",
    "reference",
    "other",
)

AUTHORITY = {"policy": 5, "training": 5}
DEFAULT_AUTHORITY = 4


def authority(page_kind: str) -> int:
    return AUTHORITY.get(page_kind, DEFAULT_AUTHORITY)


def load_topics(path: str | None) -> list[str]:
    """The shared topic taxonomy: one slug per line, # comments allowed."""
    text = (
        Path(path).read_text(encoding="utf-8")
        if path
        else resources.files("wikijs2rag").joinpath("topics.txt").read_text(encoding="utf-8")
    )
    slugs = []
    for line in text.splitlines():
        slug = line.split("#", 1)[0].strip()
        if slug:
            slugs.append(slug)
    return slugs


def _excerpt(text: str, max_tokens: int) -> str:
    words = text.split()
    out: list[str] = []
    for word in words:
        out.append(word)
        if count_tokens(" ".join(out)) >= max_tokens:
            out.append("…")
            break
    return " ".join(out)


# Heuristic page kinds, checked in order against path, title and tags.
_KIND_HINTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("meeting-notes", re.compile(r"\b(minutes|meeting|agenda)\b")),
    ("policy", re.compile(r"\b(polic(y|ies)|rules?|bylaws?|code of conduct|guidelines?)\b")),
    ("training", re.compile(r"\b(training|certification|checkout|sign-?off)\b")),
    ("build-instructions", re.compile(r"\b(build|esphome|firmware|wiring|schematic)\b")),
    ("procedure", re.compile(r"\b(how ?to|procedure|steps|checklist|opening|closing)\b")),
    ("equipment", re.compile(r"\b(equipment|tools?|machines?|laser|printer|lathe|mill|cnc|saw)\b")),
    ("area", re.compile(r"\b(areas?|rooms?|shop|section|yard|floor ?plan)\b")),
    ("project", re.compile(r"\b(projects?)\b")),
)


class Enricher:
    def __init__(self, llm: LLM | None, topics: list[str]) -> None:
        self.llm = llm
        self.topics = topics

    # -- page kind --------------------------------------------------------

    def page_kind(self, page: StagedPage) -> str:
        if self.llm:
            system = (
                "You classify pages of a hackerspace wiki. Kinds: "
                + ", ".join(PAGE_KINDS)
                + ". policy: rules members must follow. procedure: how to do a task. training: what a training or "
                "checkout covers. equipment: a machine or tool and how to use it. build-instructions: how a device "
                "the space built is made. area: a room or area of the space. "
                'Answer {"kind": "<kind>"}.'
            )
            prompt = (
                f"Path: {page.path}\nTitle: {page.title}\nTags: {', '.join(page.tags)}\n\n{_excerpt(page.text, 600)}"
            )
            try:
                kind = str(self.llm.json("page_kind", system, prompt).get("kind", "")).strip().lower()
                if kind in PAGE_KINDS:
                    return kind
                logger.debug("Model gave unknown page kind %r for %s", kind, page.path)
            except (LLMError, AttributeError) as exc:
                logger.warning("Page kind for %s falls back to heuristics: %s", page.path, exc)
        haystack = " ".join([page.path.replace("/", " ").replace("-", " "), page.title, *page.tags]).lower()
        for kind, pattern in _KIND_HINTS:
            if pattern.search(haystack):
                return kind
        return "other"

    # -- summary ----------------------------------------------------------

    def summary(self, page: StagedPage) -> str:
        if self.llm:
            system = (
                "Summarize this hackerspace wiki page in 2-4 plain sentences: what it covers and who it is for. "
                "No preamble."
            )
            prompt = f"Title: {page.title}\nDescription: {page.description}\n\n{_excerpt(page.text, 3000)}"
            try:
                return self.llm.text("summary", system, prompt, max_tokens=250)
            except LLMError as exc:
                logger.warning("Summary for %s falls back to an excerpt: %s", page.path, exc)
        lead = _excerpt(re.sub(r"[#>*`|]", " ", page.text), 80)
        return f"{page.description} {lead}".strip() if page.description else lead

    # -- topics -----------------------------------------------------------

    def page_topics(self, page: StagedPage, summary: str) -> list[str]:
        known = set(self.topics)
        if self.llm and self.topics:
            system = (
                "Map this hackerspace wiki page to topics from this list only: "
                + ", ".join(self.topics)
                + '. Pick the 1-5 that fit best, or none. Answer {"topics": ["slug", ...]}.'
            )
            prompt = f"Title: {page.title}\nTags: {', '.join(page.tags)}\nSummary: {summary}"
            try:
                found = self.llm.json("topics", system, prompt).get("topics") or []
                return sorted({str(t).strip().lower() for t in found} & known)
            except (LLMError, AttributeError) as exc:
                logger.warning("Topics for %s fall back to tag matching: %s", page.path, exc)
        words = {t.lower().replace(" ", "-") for t in page.tags} | set(page.path.lower().split("/"))
        return sorted(words & known)

    # -- per-chunk context --------------------------------------------------

    def context(self, page: StagedPage, summary: str, heading_path: str, text: str) -> str:
        if self.llm:
            system = (
                "Write 1-2 short sentences that situate this excerpt within its wiki page, so a search engine can "
                "match it. Mention the page and section topic. Do not repeat the excerpt. No preamble."
            )
            prompt = f"Page: {page.title}\nPage summary: {summary}\nSection: {heading_path}\n\nExcerpt:\n{text}"
            try:
                return self.llm.text("context", system, prompt, max_tokens=120)
            except LLMError as exc:
                logger.warning("Context for a chunk of %s falls back to a template: %s", page.path, exc)
        where = f'section "{heading_path}" of ' if heading_path and heading_path != page.title else ""
        about = f" {page.description}" if page.description else ""
        return f'From {where}the wiki page "{page.title}".{about}'.strip()

    # -- exclusion classifier ---------------------------------------------

    def exclusion(self, title: str, text: str) -> str | None:
        """The reason to keep this out of the index, or None. Raises LLMError
        if the model can't be asked: callers must fail closed."""
        if not self.llm:
            raise LLMError("The exclusion classifier needs LLM_MODEL")
        system = (
            "You screen hackerspace wiki pages before they are indexed for a members' chatbot. Members may see "
            "how to use the space (entering and leaving, the fob, the camera display, the exit dashboard), "
            "policies, procedures, where things are and how the space's own devices are built. "
            "Exclude the page if any part of it contains: credentials or secrets of any kind; admin procedures "
            "for changing, disabling, pairing, deleting or reconfiguring infrastructure; firmware or config for "
            "anything in the access path (RFID readers, door and gate controllers, locks); details of incidents "
            "(thefts, vandalism, disputes) or anything about individual people; member records, fob logs or "
            "camera footage. "
            'Answer {"exclude": true or false, "reason": "<short reason, no quotes from the page>"}.'
        )
        reasons: list[str] = []
        # Long pages are screened in pieces so nothing is cut off.
        words = text.split()
        step = 2500
        for start in range(0, max(len(words), 1), step):
            piece = " ".join(words[start : start + step])
            verdict = self.llm.json("exclusion", system, f"Title: {title}\n\n{piece}", max_tokens=150)
            if not isinstance(verdict, dict) or "exclude" not in verdict:
                raise LLMError(f"Unexpected exclusion verdict: {verdict!r}")
            if verdict.get("exclude") is True or str(verdict.get("exclude")).lower() == "true":
                reasons.append(str(verdict.get("reason") or "flagged by the exclusion classifier"))
        return "; ".join(reasons) if reasons else None
