import pytest

from wikijs2rag.config import (
    ConfigError,
    EmbeddingSettings,
    LLMSettings,
    QdrantSettings,
    QuerySettings,
    Settings,
    database_url,
)

WIKI = {"WIKI_URL": "https://wiki.example.com/"}


def test_defaults() -> None:
    s = Settings.from_env(WIKI)
    assert s == Settings(wiki_url="https://wiki.example.com")
    assert s.qdrant == QdrantSettings(host="qdrant", port=6333)
    assert (s.qdrant.collection("public"), s.qdrant.collection("member")) == ("hs_public", "hs_member")
    assert s.llm == LLMSettings()
    assert s.wiki_member_api_key is None
    assert s.embedding == EmbeddingSettings()


def test_reads_environment() -> None:
    env = {
        **WIKI,
        "WIKI_MEMBER_API_KEY": "dummy-wiki-key",
        "WIKI_DEFAULT_LOCALE": "fr",
        "WIKI_EXCLUDED_PATHS": "/admin/, board ,",
        "DATABASE_URL": "postgresql+psycopg://db/x",
        "QDRANT_HOST": "db",
        "QDRANT_PORT": "7000",
        "QDRANT_PUBLIC_COLLECTION": "pub",
        "QDRANT_MEMBER_COLLECTION": "mem",
        "LLM_BASE_URL": "http://gw/v1",
        "LLM_MODEL": "gen",
        "VISION_MODEL": "vis",
        "TOPICS_FILE": "/t.txt",
        "CANARY_ADMIN": "a1,a2",
        "CANARY_MEMBER": "m1",
        "RECONCILE_INTERVAL_SECONDS": "600",
        "ATTACHMENT_MAX_MB": "1",
        "OPENAI_BASE_URL": "http://o:1/v1",
        "OPENAI_API_KEY": "dummy-openai-key",
        "EMBEDDING_DIMENSIONS": "256",
        "EMBEDDING_MODEL": "m",
        "EMBEDDING_CONTEXT_LENGTH": "512",
        "CHUNK_TARGET_TOKENS": "300",
        "CHUNK_MAX_TOKENS": "600",
        "CHUNK_OVERLAP_TOKENS": "20",
        "EMBEDDING_BATCH_SIZE": "8",
        "PAGE_DELAY_SECONDS": "0.5",
        "FORCE_REINGEST": "yes",
        "POLL_INTERVAL_SECONDS": "0",
        "LOG_LEVEL": "debug",
    }
    assert Settings.from_env(env) == Settings(
        wiki_url="https://wiki.example.com",
        wiki_member_api_key="dummy-wiki-key",
        wiki_default_locale="fr",
        wiki_excluded_paths=("admin", "board"),
        database_url="postgresql+psycopg://db/x",
        qdrant=QdrantSettings(host="db", port=7000, public_collection="pub", member_collection="mem"),
        embedding=EmbeddingSettings(
            base_url="http://o:1/v1",
            model="m",
            context_length=512,
            api_key="dummy-openai-key",
            dimensions=256,
            query_prefix="",
            document_prefix="",
        ),
        llm=LLMSettings(base_url="http://gw/v1", api_key="dummy-openai-key", model="gen", vision_model="vis"),
        topics_file="/t.txt",
        chunk_target_tokens=300,
        chunk_max_tokens=600,
        chunk_overlap_tokens=20,
        batch_size=8,
        page_delay=0.5,
        attachment_max_bytes=1024 * 1024,
        force_reingest=True,
        poll_interval=0,
        reconcile_interval=600,
        canary_admin=("a1", "a2"),
        canary_member=("m1",),
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
            "WIKI_MEMBER_API_KEY": "",
            "OPENAI_API_KEY": " ",
            "EMBEDDING_DIMENSIONS": "",
            "CHUNK_TARGET_TOKENS": "",
            "CANARY_ADMIN": " ",
            "FORCE_REINGEST": "",
        }
    )
    assert s.wiki_member_api_key is None
    assert s.embedding.api_key is None
    assert s.embedding.dimensions is None
    assert s.chunk_target_tokens == 400
    assert s.canary_admin == ()
    assert s.force_reingest is False


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({}, "WIKI_URL is required"),
        ({"WIKI_URL": "wiki.example.com"}, "WIKI_URL must start with http"),
        ({**WIKI, "OPENAI_BASE_URL": "ollama:11434/v1"}, "LLM_BASE_URL"),
        ({**WIKI, "EMBEDDING_DIMENSIONS": "-1"}, "EMBEDDING_DIMENSIONS must be at least 0"),
        ({**WIKI, "QDRANT_PORT": "0"}, "QDRANT_PORT must be at least 1"),
        ({**WIKI, "QDRANT_PORT": "70000"}, "QDRANT_PORT must be 1-65535"),
        ({**WIKI, "CHUNK_TARGET_TOKENS": "big"}, "CHUNK_TARGET_TOKENS must be a whole number"),
        ({**WIKI, "CHUNK_OVERLAP_TOKENS": "-1"}, "CHUNK_OVERLAP_TOKENS must be at least 0"),
        ({**WIKI, "CHUNK_TARGET_TOKENS": "900"}, "must not exceed CHUNK_MAX_TOKENS"),
        ({**WIKI, "CHUNK_TARGET_TOKENS": "100", "CHUNK_OVERLAP_TOKENS": "100"}, "must be less than CHUNK_TARGET"),
        ({**WIKI, "WIKI_API_KEY": "k"}, "WIKI_API_KEY is no longer used"),
        ({**WIKI, "CHUNK_SIZE": "100"}, "CHUNK_SIZE is no longer used"),
        ({**WIKI, "QDRANT_COLLECTION": "wiki"}, "QDRANT_COLLECTION is no longer used"),
        ({**WIKI, "QDRANT_MEMBER_COLLECTION": "hs_public"}, "must differ"),
        ({**WIKI, "WIKI_MEMBER_API_KEY": "k"}, "needs LLM_MODEL"),
        ({**WIKI, "RECONCILE_INTERVAL_SECONDS": "5"}, "RECONCILE_INTERVAL_SECONDS must be at least 60"),
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


def test_llm_defaults_to_embeddings_gateway() -> None:
    llm = LLMSettings.from_env({"OPENAI_BASE_URL": "http://gw/v1", "OPENAI_API_KEY": "dummy-key"})
    assert (llm.base_url, llm.api_key, llm.model) == ("http://gw/v1", "dummy-key", None)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", "sqlite:///wikijs2rag.db"),
        ("sqlite:////data/s.db", "sqlite:////data/s.db"),
        ("postgres://u:pw@db/x", "postgresql+psycopg://u:pw@db/x"),
        ("postgresql://u:pw@db:5432/x", "postgresql+psycopg://u:pw@db:5432/x"),
        ("postgresql+psycopg://db/x", "postgresql+psycopg://db/x"),
    ],
)
def test_database_url(value: str, expected: str) -> None:
    assert database_url({"DATABASE_URL": value}) == expected


@pytest.mark.parametrize("value", ["mysql://u:secret-pw@db/x", "postgresql+psycopg2://u:secret-pw@db/x", "db"])
def test_database_url_refuses_other_drivers(value: str) -> None:
    with pytest.raises(ConfigError, match="DATABASE_URL") as exc:
        Settings.from_env({**WIKI, "DATABASE_URL": value})
    assert "secret-pw" not in str(exc.value)
