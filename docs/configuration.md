# Configuration

Both commands read environment variables at startup. Put them in `.env`
(copy `.env.example`); compose passes it to the containers. An invalid
value stops the command with a message naming the variable (exit code 2).
Blank values count as unset.

## Ingestion (`wikijs2rag`)

| Variable | Default | Meaning |
|----------|---------|---------|
| `WIKI_URL` | required | Base URL of the Wiki.js instance (`http://` or `https://`) |
| `WIKI_MEMBER_API_KEY` | none | API key for the member crawl, scoped to a group with exactly member-level read rights. Never a full-access key: a run whose key can list API keys stops. Requires `LLM_MODEL`. Unset: public pages only |
| `WIKI_DEFAULT_LOCALE` | `en` | Locale served at `/<path>`; other locales' URLs are `/<locale>/<path>` |
| `WIKI_EXCLUDED_PATHS` | `admin` | Comma-separated path prefixes never ingested, whatever a crawl can see |
| `CANARY_ADMIN` | none | Comma-separated strings planted in admin-only places. Seen by any crawl, the run stops before staging the page |
| `CANARY_MEMBER` | none | Comma-separated strings planted in member-only pages. Seen by the anonymous crawl, the run stops |
| `DATABASE_URL` | `sqlite:///wikijs2rag.db` (compose: the `postgres` service) | Staging database: `postgresql://user:pass@host/db` (`postgres://` and `postgresql+psycopg://` work too; all use psycopg) or `sqlite:///path` |
| `QDRANT_HOST` | `qdrant` | Qdrant host (the compose files set `qdrant`) |
| `QDRANT_PORT` | `6333` | Qdrant REST port, 1-65535 |
| `QDRANT_PUBLIC_COLLECTION` | `hs_public` | Collection for public-tier points |
| `QDRANT_MEMBER_COLLECTION` | `hs_member` | Collection for member-tier points (and public ones) |
| `OPENAI_BASE_URL` | `http://localhost:8080/v1` (compose: `http://tei/v1`) | Base URL of an OpenAI-compatible embeddings API (the part before `/embeddings`): Hugging Face TEI, OpenAI (`https://api.openai.com/v1`), vLLM, … |
| `OPENAI_API_KEY` | none | Bearer token. Required for OpenAI; leave unset for TEI and other servers without auth |
| `EMBEDDING_MODEL` | `nomic-ai/modernbert-embed-base` | Embedding model: a Hugging Face model id for TEI (compose starts TEI with it) or an OpenAI model name. Recorded in each collection's metadata; a run with a different model stops |
| `EMBEDDING_QUERY_PREFIX` | the model's (see below) | Text put before each search query before embedding. `none` for no prefix |
| `EMBEDDING_DOCUMENT_PREFIX` | the model's (see below) | Text put before each chunk before embedding. `none` for no prefix. Changing it needs `wikijs2rag --reindex` |
| `EMBEDDING_DIMENSIONS` | model's size | Shorter vectors, for models that accept the `dimensions` parameter (OpenAI `text-embedding-3-*`); `0` or unset for the full size |
| `EMBEDDING_CONTEXT_LENGTH` | `8192` | Model context in tokens, at least 1; texts (prefix included) are cut to about 0.75 words per token |
| `EMBEDDING_BATCH_SIZE` | `32` | Chunks per embeddings request, at least 1 |
| `LLM_BASE_URL` | `OPENAI_BASE_URL` | OpenAI-compatible chat API (the part before `/chat/completions`) for ingest-time LLM work |
| `LLM_API_KEY` | `OPENAI_API_KEY` | Bearer token for `LLM_BASE_URL` |
| `LLM_MODEL` | none | Generation model for summaries, chunk context lines, page kinds, topics, ambiguous links and the exclusion classifier. Unset: deterministic fallbacks, public pages only |
| `VISION_MODEL` | none | Vision model that describes images of signs, diagrams and whiteboards. Unset: images are staged with alt text and caption only |
| `TOPICS_FILE` | built-in list | Shared topic taxonomy, one slug per line, `#` comments |
| `CHUNK_TARGET_TOKENS` | `400` | Size chunks are packed to, at least 50 |
| `CHUNK_MAX_TOKENS` | `800` | Hard cap; only code blocks may exceed it |
| `CHUNK_OVERLAP_TOKENS` | `50` | Overlap when prose is split inside a section; less than the target |
| `ATTACHMENT_MAX_MB` | `20` | Largest file or image fetched |
| `POLL_INTERVAL_SECONDS` | `3600` | Seconds between incremental runs; `0` runs once and exits |
| `RECONCILE_INTERVAL_SECONDS` | `86400` | Seconds between full reconciles, at least 60 |
| `FORCE_REINGEST` | `false` | `true`/`false` (also `1`/`0`, `yes`/`no`, `on`/`off`): every run is full and re-indexes every page |
| `REINGEST_ON_START` | `false` | Same values: the first run after the service starts is full and re-indexes every page; later runs are incremental. For a one-off rebuild: set it, restart, set it back |
| `PAGE_DELAY_SECONDS` | `0.1` | Pause between fetched pages, at least 0 |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL` |

Removed in 3.0, and refused if set: `WIKI_API_KEY` (use
`WIKI_MEMBER_API_KEY`), `QDRANT_COLLECTION` (use the two tier collections),
`CHUNK_SIZE`, `CHUNK_OVERLAP`, `PARENT_CHUNK_RATIO`.

Flags: `--full` makes the first run a full reconcile; `--reindex` rebuilds
both collections from staging without contacting the wiki, then exits.

### Embedding prefixes

Many open embedding models are trained with a task prefix on each input,
one for search queries and another for the documents being searched, and
retrieve noticeably worse without them. The OpenAI embeddings API has no
field for this, so the prefix is put in front of the text before it's sent.
Ingestion uses the document prefix; `wikijs2rag-query` uses the query prefix.
Any other application that searches the collection must embed its queries
with the same model and query prefix.

When a prefix variable is unset, it defaults to the prefix of the model in
`EMBEDDING_MODEL`, matched by name:

| Model (name contains) | Query prefix | Document prefix |
|---|---|---|
| `modernbert-embed` (e.g. `nomic-ai/modernbert-embed-base`) | `search_query: ` | `search_document: ` |
| `nomic-embed-text` | `search_query: ` | `search_document: ` |
| `mxbai-embed-large` | `Represent this sentence for searching relevant passages: ` | none |
| `qwen3-embedding` | `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:` | none |
| `e5-` (e.g. `intfloat/e5-large-v2`) | `query: ` | `passage: ` |
| anything else, e.g. OpenAI `text-embedding-3-*` | none | none |

A prefix you set is used as written, except that `\n` becomes a newline and
a space is added if it doesn't end in whitespace. Quote it in `.env` to keep
its trailing space: `EMBEDDING_QUERY_PREFIX="search_query: "`.

Exit codes: `0` success (or stopped by SIGTERM), `1` Qdrant or the wiki
unreachable, a credential with admin rights, a canary where it must not be,
or a different embedding model in a collection; `2` bad configuration, or a
one-shot run in which some pages failed.

## Query tool (`wikijs2rag-query`)

Uses `QDRANT_*`, `OPENAI_BASE_URL`, `OPENAI_API_KEY` and the `EMBEDDING_*`
settings as above, except that `QDRANT_HOST` defaults to
`localhost`. Command-line flags override these:

| Variable | Flag | Default | Meaning |
|----------|------|---------|---------|
| `QUERY_LIMIT` | `--limit` | `5` | Number of results, at least 1 |
| `QUERY_TIER` | `--tier` | `public` | Search `QDRANT_PUBLIC_COLLECTION` or `QDRANT_MEMBER_COLLECTION` |
| | `--collection` | the tier's | Search this collection instead |
| `QUERY_SHOW_TEXT` | `--show-text` | `false` | Print each matched chunk's text |
| `QUERY_MIN_SCORE` | `--min-score` | `0.0` | Hide results below this fused (reciprocal rank) score, 0-1 |
| `QUERY_WIDTH` | `--width` | `100` | Text wrapping width, at least 20 |
| `NO_COLOR` | | unset | Any value turns off colored output ([no-color.org](https://no-color.org/)) |

## Reports (`wikijs2rag-report`)

Reads `DATABASE_URL`; `canaries` reads every ingestion setting (it checks
Qdrant too). Exit code `1` from `canaries` means a canary was found.

## Docker compose (not read by the app)

| Variable | Default | Meaning |
|----------|---------|---------|
| `APP_VERSION` | `latest` | Release of `ghcr.io/romkey/wikijs2rag` that `docker-compose.prod.yaml` runs |
| `COMPOSE_PROFILES` | unset (`.env.example`: `tei`) | `tei` runs the Hugging Face Text Embeddings Inference service; leave it out to use another embeddings server |
| `TEI_IMAGE` | `ghcr.io/huggingface/text-embeddings-inference:cpu-1.9.4` | TEI image; on ARM use `…:cpu-arm64-1.9.4`, on NVIDIA GPUs one of TEI's GPU images |
| `TEI_MAX_BATCH_TOKENS` | `2048` | Longest input TEI embeds, in tokens; longer inputs are truncated. Memory grows with its square: about 1.5 GB at 2048, 3 GB at 4096, over 8 GB at 8192. Keep `EMBEDDING_CONTEXT_LENGTH` no larger |
| `TEI_PORT` | `8080` | Host port for TEI, for the query tool run natively and for RAG apps embedding their queries |
| `POSTGRES_PASSWORD` | required | Password of the `postgres` service; letters and digits, since it goes into `DATABASE_URL` unescaped |
| `QDRANT_REST_PORT` | `6333` | Host port for Qdrant's REST API and dashboard |
| `QDRANT_GRPC_PORT` | `6334` | Host port for Qdrant's gRPC API |
