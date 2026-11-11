"""Settings, read once at startup from environment variables.

Every variable is documented in docs/configuration.md and listed, with a
safe example value, in .env.example.
"""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass


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


def _log_level(env: Mapping[str, str]) -> str:
    level = _str(env, "LOG_LEVEL", "INFO").upper()
    if level not in logging.getLevelNamesMapping():
        raise ConfigError(f"LOG_LEVEL {level!r} is not a logging level")
    return level


@dataclass(frozen=True)
class QdrantSettings:
    host: str = "qdrant"
    port: int = 6333
    collection: str = "wiki"

    @classmethod
    def from_env(cls, env: Mapping[str, str], default_host: str = "qdrant") -> QdrantSettings:
        port = _int(env, "QDRANT_PORT", cls.port, minimum=1)
        if port > 65535:
            raise ConfigError(f"QDRANT_PORT must be 1-65535, not {port}")
        return cls(
            host=_str(env, "QDRANT_HOST", default_host),
            port=port,
            collection=_str(env, "QDRANT_COLLECTION", cls.collection),
        )


@dataclass(frozen=True)
class EmbeddingSettings:
    ollama_url: str = "http://localhost:11434"
    model: str = "nomic-embed-text"
    context_length: int = 8192
    api_key: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> EmbeddingSettings:
        return cls(
            ollama_url=_url(env, "OLLAMA_URL", cls.ollama_url),
            model=_str(env, "EMBEDDING_MODEL", cls.model),
            context_length=_int(env, "EMBEDDING_CONTEXT_LENGTH", cls.context_length, minimum=1),
            api_key=_optional(env, "OLLAMA_API_KEY"),
        )


@dataclass(frozen=True)
class Settings:
    """Settings for the ingestion service (``wikijs2rag``)."""

    wiki_url: str
    wiki_api_key: str | None = None
    qdrant: QdrantSettings = QdrantSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    chunk_size: int = 256
    chunk_overlap: int = 50
    batch_size: int = 32
    page_delay: float = 0.1
    force_reingest: bool = False
    parent_ratio: int = 4
    poll_interval: int = 3600
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Settings:
        chunk_size = _int(env, "CHUNK_SIZE", cls.chunk_size, minimum=1)
        chunk_overlap = _int(env, "CHUNK_OVERLAP", cls.chunk_overlap)
        if chunk_overlap >= chunk_size:
            raise ConfigError(f"CHUNK_OVERLAP ({chunk_overlap}) must be less than CHUNK_SIZE ({chunk_size})")
        return cls(
            wiki_url=_url(env, "WIKI_URL"),
            wiki_api_key=_optional(env, "WIKI_API_KEY"),
            qdrant=QdrantSettings.from_env(env),
            embedding=EmbeddingSettings.from_env(env),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=_int(env, "EMBEDDING_BATCH_SIZE", cls.batch_size, minimum=1),
            page_delay=_float(env, "PAGE_DELAY_SECONDS", cls.page_delay),
            force_reingest=_bool(env, "FORCE_REINGEST", cls.force_reingest),
            parent_ratio=_int(env, "PARENT_CHUNK_RATIO", cls.parent_ratio, minimum=1),
            poll_interval=_int(env, "POLL_INTERVAL_SECONDS", cls.poll_interval),
            log_level=_log_level(env),
        )


@dataclass(frozen=True)
class QuerySettings:
    """Settings for the query tool (``wikijs2rag-query``); flags override them."""

    qdrant: QdrantSettings = QdrantSettings(host="localhost")
    embedding: EmbeddingSettings = EmbeddingSettings()
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
            limit=_int(env, "QUERY_LIMIT", cls.limit, minimum=1),
            show_text=_bool(env, "QUERY_SHOW_TEXT", cls.show_text),
            min_score=min_score,
            width=_int(env, "QUERY_WIDTH", cls.width, minimum=20),
            # https://no-color.org/: any non-empty value disables color.
            color=not env.get("NO_COLOR"),
        )
