import pytest

from wikijs2rag.config import (
    ConfigError,
    EmbeddingSettings,
    QdrantSettings,
    QuerySettings,
    Settings,
)

WIKI = {"WIKI_URL": "https://wiki.example.com/"}


def test_defaults() -> None:
    s = Settings.from_env(WIKI)
    assert s == Settings(wiki_url="https://wiki.example.com")
    assert s.qdrant == QdrantSettings(host="qdrant", port=6333, collection="wiki")
    assert s.embedding == EmbeddingSettings()


def test_reads_environment() -> None:
    env = {
        **WIKI,
        "WIKI_API_KEY": "dummy-wiki-key",
        "QDRANT_HOST": "db",
        "QDRANT_PORT": "7000",
        "QDRANT_COLLECTION": "c",
        "OPENAI_BASE_URL": "http://o:1/v1",
        "OPENAI_API_KEY": "dummy-openai-key",
        "EMBEDDING_DIMENSIONS": "256",
        "EMBEDDING_MODEL": "m",
        "EMBEDDING_CONTEXT_LENGTH": "512",
        "CHUNK_SIZE": "100",
        "CHUNK_OVERLAP": "10",
        "EMBEDDING_BATCH_SIZE": "8",
        "PAGE_DELAY_SECONDS": "0.5",
        "FORCE_REINGEST": "yes",
        "PARENT_CHUNK_RATIO": "3",
        "POLL_INTERVAL_SECONDS": "0",
        "LOG_LEVEL": "debug",
    }
    assert Settings.from_env(env) == Settings(
        wiki_url="https://wiki.example.com",
        wiki_api_key="dummy-wiki-key",
        qdrant=QdrantSettings(host="db", port=7000, collection="c"),
        embedding=EmbeddingSettings(
            base_url="http://o:1/v1",
            model="m",
            context_length=512,
            api_key="dummy-openai-key",
            dimensions=256,
            query_prefix="",
            document_prefix="",
        ),
        chunk_size=100,
        chunk_overlap=10,
        batch_size=8,
        page_delay=0.5,
        force_reingest=True,
        parent_ratio=3,
        poll_interval=0,
        log_level="DEBUG",
    )


def test_default_embedding_is_modernbert_with_its_prefixes() -> None:
    e = Settings.from_env(WIKI).embedding
    assert e.model == "nomic-ai/modernbert-embed-base"
    assert e.base_url == "http://localhost:8080/v1"
    assert e.context_length == 8192
    assert (e.query_prefix, e.document_prefix) == ("search_query: ", "search_document: ")


@pytest.mark.parametrize(
    ("model", "query", "document"),
    [
        ("intfloat/multilingual-e5-large", "query: ", "passage: "),
        ("mixedbread-ai/mxbai-embed-large-v1", "Represent this sentence for searching relevant passages: ", ""),
        ("text-embedding-3-small", "", ""),
    ],
)
def test_prefixes_default_to_the_models(model: str, query: str, document: str) -> None:
    e = EmbeddingSettings.from_env({"EMBEDDING_MODEL": model})
    assert (e.query_prefix, e.document_prefix) == (query, document)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", "search_query: "),  # blank: the model's prefix
        ("   ", "search_query: "),
        ("none", ""),
        (" NONE ", ""),
        ("q: ", "q: "),  # kept as written
        ("q:", "q: "),  # a space added
        ("Instruct: find\\nQuery:", "Instruct: find\nQuery: "),  # a literal backslash-n becomes a newline
    ],
)
def test_query_prefix_values(value: str, expected: str) -> None:
    e = EmbeddingSettings.from_env({"EMBEDDING_QUERY_PREFIX": value})
    assert e.query_prefix == expected
    assert e.document_prefix == "search_document: "


def test_document_prefix_override_and_none() -> None:
    assert EmbeddingSettings.from_env({"EMBEDDING_DOCUMENT_PREFIX": "doc: "}).document_prefix == "doc: "
    assert EmbeddingSettings.from_env({"EMBEDDING_DOCUMENT_PREFIX": "none"}).document_prefix == ""


def test_embedder_gets_prefixes(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}
    monkeypatch.setattr("wikijs2rag.config.OpenAIEmbedder", lambda *_a, **k: seen.update(k))
    EmbeddingSettings(query_prefix="q: ", document_prefix="d: ").embedder()
    assert seen["query_prefix"] == "q: "
    assert seen["document_prefix"] == "d: "


def test_blank_values_use_defaults() -> None:
    s = Settings.from_env(
        {
            **WIKI,
            "WIKI_API_KEY": "",
            "OPENAI_API_KEY": " ",
            "EMBEDDING_DIMENSIONS": "",
            "CHUNK_SIZE": "",
            "FORCE_REINGEST": "",
        }
    )
    assert s.wiki_api_key is None
    assert s.embedding.api_key is None
    assert s.embedding.dimensions is None
    assert s.chunk_size == 256
    assert s.force_reingest is False


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({}, "WIKI_URL is required"),
        ({"WIKI_URL": "wiki.example.com"}, "WIKI_URL must start with http"),
        ({**WIKI, "OPENAI_BASE_URL": "ollama:11434/v1"}, "OPENAI_BASE_URL"),
        ({**WIKI, "EMBEDDING_DIMENSIONS": "-1"}, "EMBEDDING_DIMENSIONS must be at least 0"),
        ({**WIKI, "QDRANT_PORT": "0"}, "QDRANT_PORT must be at least 1"),
        ({**WIKI, "QDRANT_PORT": "70000"}, "QDRANT_PORT must be 1-65535"),
        ({**WIKI, "CHUNK_SIZE": "big"}, "CHUNK_SIZE must be a whole number"),
        ({**WIKI, "CHUNK_OVERLAP": "-1"}, "CHUNK_OVERLAP must be at least 0"),
        ({**WIKI, "CHUNK_SIZE": "50", "CHUNK_OVERLAP": "50"}, "must be less than CHUNK_SIZE"),
        ({**WIKI, "PAGE_DELAY_SECONDS": "soon"}, "PAGE_DELAY_SECONDS must be a number"),
        ({**WIKI, "PAGE_DELAY_SECONDS": "-1"}, "PAGE_DELAY_SECONDS must be at least"),
        ({**WIKI, "FORCE_REINGEST": "maybe"}, "FORCE_REINGEST must be true or false"),
        ({**WIKI, "LOG_LEVEL": "loud"}, "LOG_LEVEL"),
    ],
)
def test_rejects_bad_values(env: dict[str, str], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        Settings.from_env(env)


@pytest.mark.parametrize(("text", "value"), [("off", False), ("0", False), ("On", True), ("1", True)])
def test_bool_spellings(text: str, value: bool) -> None:
    assert Settings.from_env({**WIKI, "FORCE_REINGEST": text}).force_reingest is value


def test_query_defaults() -> None:
    q = QuerySettings.from_env({})
    assert q == QuerySettings()
    assert q.qdrant.host == "localhost"
    assert q.color is True


def test_query_reads_environment() -> None:
    env = {
        "QUERY_LIMIT": "9",
        "QUERY_SHOW_TEXT": "true",
        "QUERY_MIN_SCORE": "0.4",
        "QUERY_WIDTH": "80",
        "NO_COLOR": "1",
        "QDRANT_HOST": "q",
    }
    q = QuerySettings.from_env(env)
    assert (q.limit, q.show_text, q.min_score, q.width, q.color, q.qdrant.host) == (9, True, 0.4, 80, False, "q")


@pytest.mark.parametrize(
    ("env", "message"),
    [({"QUERY_MIN_SCORE": "2"}, "QUERY_MIN_SCORE must be 0-1"), ({"QUERY_LIMIT": "0"}, "QUERY_LIMIT")],
)
def test_query_rejects_bad_values(env: dict[str, str], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        QuerySettings.from_env(env)
