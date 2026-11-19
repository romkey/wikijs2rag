"""Settings, read once at startup from environment variables.

Every variable is documented in docs/configuration.md and listed, with a
safe example value, in .env.example.
"""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass

from wikijs2rag.embedder import OpenAIEmbedder, default_prefixes


class ConfigError(ValueError):
    """An environment variable is missing or set to something unusable."""


def _str(env: Mapping[str, str], name: str, default: str) -> str:
    """The stripped value of *name*, or *default* when unset or blank."""
    return env.get(name, "").strip() or default


def _optional(env: Mapping[str, str], name: str) -> str | None:
    return env.get(name, "").strip() or None


def _url(env: Mapping[str, str], name: str, default: str | None = None) -> str:
    value = _optional(env, name) or default
    if value is None:
        raise ConfigError(f"{name} is required")
    if not value.startswith(("http://", "https://")):
        raise ConfigError(f"{name} must start with http:// or https://, not {value!r}")
    return value.rstrip("/")


_POSTGRES_SCHEMES = ("postgres://", "postgresql://", "postgresql+psycopg://")


def database_url(env: Mapping[str, str], default: str = "sqlite:///wikijs2rag.db") -> str:
    """DATABASE_URL as a SQLAlchemy URL for the drivers installed (SQLite, psycopg).

    postgres:// and postgresql:// (the libpq forms most tools print) use psycopg.
    """
    value = _str(env, "DATABASE_URL", default)
    for scheme in _POSTGRES_SCHEMES:
        if value.startswith(scheme):
            return "postgresql+psycopg://" + value.removeprefix(scheme)
    if value.startswith("sqlite:"):
        return value
    shown = value.split("://", 1)[0] if "://" in value else value
    raise ConfigError(f"DATABASE_URL must start with postgresql:// (or postgres://) or sqlite:, not {shown!r}")


def _int(env: Mapping[str, str], name: str, default: int, minimum: int = 0) -> int:
    text = _str(env, name, str(default))
    try:
        value = int(text)
    except ValueError:
        raise ConfigError(f"{name} must be a whole number, not {text!r}") from None
    if value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}, not {value}")
    return value


def _float(env: Mapping[str, str], name: str, default: float, minimum: float = 0.0) -> float:
    text = _str(env, name, str(default))
    try:
        value = float(text)
    except ValueError:
        raise ConfigError(f"{name} must be a number, not {text!r}") from None
    if value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}, not {value}")
    return value


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    text = env.get(name, "").strip().lower()
    if not text:
        return default
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ConfigError(f"{name} must be true or false, not {text!r}")


def _prefix(env: Mapping[str, str], name: str, default: str) -> str:
    """A text prefix: kept as written (not stripped), with a literal \\n read
    as a newline and a space added if it doesn't end in whitespace. Blank
    means *default*; ``none`` means no prefix."""
    raw = env.get(name, "")
    if not raw.strip():
        return default
    if raw.strip().lower() == "none":
        return ""
    value = raw.lstrip().replace("\\n", "\n")
    return value if value[-1].isspace() else value + " "


def _log_level(env: Mapping[str, str]) -> str:
    level = _str(env, "LOG_LEVEL", "INFO").upper()
    if level not in logging.getLevelNamesMapping():
        raise ConfigError(f"LOG_LEVEL {level!r} is not a logging level")
    return level


def _list(env: Mapping[str, str], name: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    """A comma-separated list; blank items dropped."""
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _tier(env: Mapping[str, str]) -> str:
    tier = _str(env, "QUERY_TIER", "public").lower()
    if tier not in ("public", "member"):
        raise ConfigError(f"QUERY_TIER must be public or member, not {tier!r}")
    return tier


@dataclass(frozen=True)
class QdrantSettings:
    host: str = "qdrant"
    port: int = 6333
    public_collection: str = "hs_public"
    member_collection: str = "hs_member"

    @classmethod
    def from_env(cls, env: Mapping[str, str], default_host: str = "qdrant") -> QdrantSettings:
        port = _int(env, "QDRANT_PORT", cls.port, minimum=1)
        if port > 65535:
            raise ConfigError(f"QDRANT_PORT must be 1-65535, not {port}")
        if _optional(env, "QDRANT_COLLECTION"):
            raise ConfigError(
                "QDRANT_COLLECTION is no longer used; set QDRANT_PUBLIC_COLLECTION and QDRANT_MEMBER_COLLECTION"
            )
        public = _str(env, "QDRANT_PUBLIC_COLLECTION", cls.public_collection)
        member = _str(env, "QDRANT_MEMBER_COLLECTION", cls.member_collection)
        if public == member:
            raise ConfigError("QDRANT_PUBLIC_COLLECTION and QDRANT_MEMBER_COLLECTION must differ")
        return cls(
            host=_str(env, "QDRANT_HOST", default_host),
            port=port,
            public_collection=public,
            member_collection=member,
        )

    def collection(self, tier: str) -> str:
        return self.public_collection if tier == "public" else self.member_collection


@dataclass(frozen=True)
class EmbeddingSettings:
    base_url: str = "http://localhost:8080/v1"
    model: str = "nomic-ai/modernbert-embed-base"
    context_length: int = 8192
    api_key: str | None = None
    dimensions: int | None = None
    query_prefix: str = "search_query: "
    document_prefix: str = "search_document: "

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> EmbeddingSettings:
        dimensions = _int(env, "EMBEDDING_DIMENSIONS", 0, minimum=0) or None
        model = _str(env, "EMBEDDING_MODEL", cls.model)
        query_default, document_default = default_prefixes(model)
        return cls(
            base_url=_url(env, "OPENAI_BASE_URL", cls.base_url),
            model=model,
            context_length=_int(env, "EMBEDDING_CONTEXT_LENGTH", cls.context_length, minimum=1),
            api_key=_optional(env, "OPENAI_API_KEY"),
            dimensions=dimensions,
            query_prefix=_prefix(env, "EMBEDDING_QUERY_PREFIX", query_default),
            document_prefix=_prefix(env, "EMBEDDING_DOCUMENT_PREFIX", document_default),
        )

    def embedder(self) -> OpenAIEmbedder:
        """Connect to the embeddings API; probes it for the vector size."""
        return OpenAIEmbedder(
            self.model,
            base_url=self.base_url,
            context_length=self.context_length,
            api_key=self.api_key,
            dimensions=self.dimensions,
            query_prefix=self.query_prefix,
            document_prefix=self.document_prefix,
        )


@dataclass(frozen=True)
class LLMSettings:
    """The generation model, for ingest-time summaries, classification and
    linking. Reached through the same kind of OpenAI-compatible gateway."""

    base_url: str = "http://localhost:8080/v1"
    api_key: str | None = None
    model: str | None = None
    vision_model: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> LLMSettings:
        return cls(
            base_url=_url(env, "LLM_BASE_URL", _optional(env, "OPENAI_BASE_URL") or cls.base_url),
            api_key=_optional(env, "LLM_API_KEY") or _optional(env, "OPENAI_API_KEY"),
            model=_optional(env, "LLM_MODEL"),
            vision_model=_optional(env, "VISION_MODEL"),
        )


@dataclass(frozen=True)
class Settings:
    """Settings for the ingestion service (``wikijs2rag``)."""

    wiki_url: str
    wiki_member_api_key: str | None = None
    wiki_default_locale: str = "en"
    wiki_excluded_paths: tuple[str, ...] = ("admin",)
    database_url: str = "sqlite:///wikijs2rag.db"
    qdrant: QdrantSettings = QdrantSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    llm: LLMSettings = LLMSettings()
    topics_file: str | None = None
    chunk_target_tokens: int = 400
    chunk_max_tokens: int = 800
    chunk_overlap_tokens: int = 50
    batch_size: int = 32
    page_delay: float = 0.1
    attachment_max_bytes: int = 20 * 1024 * 1024
    force_reingest: bool = False
    poll_interval: int = 3600
    reconcile_interval: int = 86400
    canary_admin: tuple[str, ...] = ()
    canary_member: tuple[str, ...] = ()
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Settings:
        if _optional(env, "WIKI_API_KEY"):
            raise ConfigError(
                "WIKI_API_KEY is no longer used. Public pages are crawled with no credentials; set "
                "WIKI_MEMBER_API_KEY to a key scoped to the members group (never a full-access key)."
            )
        for gone in ("CHUNK_SIZE", "CHUNK_OVERLAP", "PARENT_CHUNK_RATIO"):
            if _optional(env, gone):
                raise ConfigError(f"{gone} is no longer used; see CHUNK_TARGET_TOKENS and CHUNK_MAX_TOKENS")
        target = _int(env, "CHUNK_TARGET_TOKENS", cls.chunk_target_tokens, minimum=50)
        maximum = _int(env, "CHUNK_MAX_TOKENS", cls.chunk_max_tokens, minimum=50)
        overlap = _int(env, "CHUNK_OVERLAP_TOKENS", cls.chunk_overlap_tokens)
        if target > maximum:
            raise ConfigError(f"CHUNK_TARGET_TOKENS ({target}) must not exceed CHUNK_MAX_TOKENS ({maximum})")
        if overlap >= target:
            raise ConfigError(f"CHUNK_OVERLAP_TOKENS ({overlap}) must be less than CHUNK_TARGET_TOKENS ({target})")
        llm = LLMSettings.from_env(env)
        member_key = _optional(env, "WIKI_MEMBER_API_KEY")
        if member_key and not llm.model:
            raise ConfigError("WIKI_MEMBER_API_KEY needs LLM_MODEL: member pages go through the exclusion classifier")
        return cls(
            wiki_url=_url(env, "WIKI_URL"),
            wiki_member_api_key=member_key,
            wiki_default_locale=_str(env, "WIKI_DEFAULT_LOCALE", cls.wiki_default_locale),
            wiki_excluded_paths=tuple(
                p.strip("/") for p in _list(env, "WIKI_EXCLUDED_PATHS", cls.wiki_excluded_paths) if p.strip("/")
            ),
            database_url=database_url(env, cls.database_url),
            qdrant=QdrantSettings.from_env(env),
            embedding=EmbeddingSettings.from_env(env),
            llm=llm,
            topics_file=_optional(env, "TOPICS_FILE"),
            chunk_target_tokens=target,
            chunk_max_tokens=maximum,
            chunk_overlap_tokens=overlap,
            batch_size=_int(env, "EMBEDDING_BATCH_SIZE", cls.batch_size, minimum=1),
            page_delay=_float(env, "PAGE_DELAY_SECONDS", cls.page_delay),
            attachment_max_bytes=_int(env, "ATTACHMENT_MAX_MB", 20, minimum=0) * 1024 * 1024,
            force_reingest=_bool(env, "FORCE_REINGEST", cls.force_reingest),
            poll_interval=_int(env, "POLL_INTERVAL_SECONDS", cls.poll_interval),
            reconcile_interval=_int(env, "RECONCILE_INTERVAL_SECONDS", cls.reconcile_interval, minimum=60),
            canary_admin=_list(env, "CANARY_ADMIN"),
            canary_member=_list(env, "CANARY_MEMBER"),
            log_level=_log_level(env),
        )


@dataclass(frozen=True)
class QuerySettings:
    """Settings for the query tool (``wikijs2rag-query``); flags override them."""

    qdrant: QdrantSettings = QdrantSettings(host="localhost")
    embedding: EmbeddingSettings = EmbeddingSettings()
    tier: str = "public"
    limit: int = 5
    show_text: bool = False
    min_score: float = 0.0
    width: int = 100
    color: bool = True

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> QuerySettings:
        min_score = _float(env, "QUERY_MIN_SCORE", cls.min_score)
        if min_score > 1:
            raise ConfigError(f"QUERY_MIN_SCORE must be 0-1, not {min_score}")
        return cls(
            qdrant=QdrantSettings.from_env(env, default_host="localhost"),
            embedding=EmbeddingSettings.from_env(env),
            tier=_tier(env),
            limit=_int(env, "QUERY_LIMIT", cls.limit, minimum=1),
            show_text=_bool(env, "QUERY_SHOW_TEXT", cls.show_text),
            min_score=min_score,
            width=_int(env, "QUERY_WIDTH", cls.width, minimum=20),
            # https://no-color.org/: any non-empty value disables color.
            color=not env.get("NO_COLOR"),
        )
