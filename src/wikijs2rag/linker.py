"""The linker: tags chunks with the registry entities they mention.

Aliases come from the entity registry (``entity_aliases``), which humans
edit. A chunk matches an alias when the alias appears as whole words, case
ignored. An alias shared by several entities is ambiguous: the LLM picks
among them, and without an LLM ambiguous hits are dropped.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field

from wikijs2rag.llm import LLM, LLMError
from wikijs2rag.staging import Staging

logger = logging.getLogger(__name__)

# Registry entity type -> payload field.
ENTITY_FIELDS = {
    "asset": "asset_ids",
    "location": "location_ids",
    "work": "work_ids",
    "event": "event_ids",
    "series": "event_ids",
    "training": "training_ids",
}
MIN_ALIAS_LENGTH = 3

# Page kind -> (entity type, role recorded in entity_documents).
PAGE_ROLES = {
    "equipment": ("asset", "how-to-use"),
    "area": ("location", "area"),
    "training": ("training", "training"),
    "build-instructions": ("asset", "build-instructions"),
}


@dataclass
class Links:
    ids: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))  # payload field -> ids

    def add(self, entity_type: str, entity_id: str) -> None:
        if entity_type in ENTITY_FIELDS:
            self.ids[ENTITY_FIELDS[entity_type]].add(entity_id)

    def merged(self, other: Links) -> Links:
        result = Links()
        for source in (self, other):
            for key, values in source.ids.items():
                result.ids[key] |= values
        return result

    def payload(self) -> dict[str, list[str]]:
        return {name: sorted(self.ids.get(name, set())) for name in sorted(set(ENTITY_FIELDS.values()))}


class Linker:
    def __init__(self, staging: Staging, llm: LLM | None) -> None:
        self.llm = llm
        self._by_alias: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for entity_type, entity_id, alias in staging.aliases():
            alias = " ".join(alias.lower().split())
            if len(alias) >= MIN_ALIAS_LENGTH:
                self._by_alias[alias].add((entity_type, entity_id))
        self.attributes = staging.entity_attributes()
        self._pattern: re.Pattern[str] | None = None
        if self._by_alias:
            alternatives = sorted(self._by_alias, key=len, reverse=True)
            self._pattern = re.compile(
                r"(?<!\w)(" + "|".join(re.escape(a).replace(r"\ ", r"\s+") for a in alternatives) + r")(?!\w)",
                re.IGNORECASE,
            )
        logger.debug("Linker loaded %d aliases", len(self._by_alias))

    def _hits(self, text: str) -> dict[str, set[tuple[str, str]]]:
        if not self._pattern:
            return {}
        return {
            alias: self._by_alias[alias]
            for alias in {" ".join(m.group(1).lower().split()) for m in self._pattern.finditer(text)}
        }

    def _resolve(self, alias: str, candidates: set[tuple[str, str]], text: str) -> set[tuple[str, str]]:
        if len(candidates) == 1:
            return candidates
        if not self.llm:
            return set()
        options = sorted(f"{t}:{i}" for t, i in candidates)
        system = (
            f'The text mentions "{alias}", which could mean any of these registry entities: '
            + ", ".join(options)
            + '. Which does it mean? Answer {"entity": "<type:id>"} or {"entity": null} if none or unsure.'
        )
        try:
            choice = self.llm.json("link", system, text[:4000]).get("entity")
        except (LLMError, AttributeError) as exc:
            logger.warning("Could not resolve ambiguous alias %r: %s", alias, exc)
            return set()
        if isinstance(choice, str) and choice in options:
            entity_type, entity_id = choice.split(":", 1)
            return {(entity_type, entity_id)}
        return set()

    def link(self, text: str) -> Links:
        links = Links()
        for alias, candidates in self._hits(text).items():
            for entity_type, entity_id in self._resolve(alias, candidates, text):
                links.add(entity_type, entity_id)
        return links

    def link_page(
        self, page_kind: str, title: str, path: str, tags: list[str]
    ) -> tuple[Links, list[tuple[str, str, str]]]:
        """Page-level links from title, path and tags, and the (entity_type,
        entity_id, role) rows this page documents for its kind."""
        haystack = " ".join([title, path.replace("/", " ").replace("-", " "), *tags])
        links = Links()
        roles: list[tuple[str, str, str]] = []
        wanted = PAGE_ROLES.get(page_kind)
        for alias, candidates in self._hits(haystack).items():
            for entity_type, entity_id in self._resolve(alias, candidates, haystack):
                links.add(entity_type, entity_id)
                if wanted and entity_type == wanted[0]:
                    roles.append((entity_type, entity_id, wanted[1]))
        return links, roles

    def section(self, links: Links) -> str | None:
        """The space section (1-4 or yard) of the locations linked, if they agree."""
        sections = {
            str(self.attributes.get(("location", lid), {}).get("section") or "")
            for lid in links.ids.get("location_ids", set())
        } - {""}
        return sections.pop() if len(sections) == 1 else None
