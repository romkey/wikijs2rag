# Configuration

Both commands read environment variables at startup. Put them in `.env`
(copy `.env.example`); compose passes it to the containers. An invalid
value stops the command with a message naming the variable (exit code 2).
Blank values count as unset.

## Ingestion (`wikijs2rag`)

| Variable | Default | Meaning |
|----------|---------|---------|
| `WIKI_URL` | required | Base URL of the Wiki.js instance (`http://` or `https://`) |
| `WIKI_API_KEY` | none | Wiki.js API key, sent as a bearer token. Only needed if guests can't use the GraphQL API |
| `QDRANT_HOST` | `qdrant` | Qdrant host (the compose files set `qdrant`) |
| `QDRANT_PORT` | `6333` | Qdrant REST port, 1-65535 |
| `QDRANT_COLLECTION` | `wiki` | Collection to write; one per wiki |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Base URL of an OpenAI-compatible embeddings API (the part before `/embeddings`). For Ollama on the Docker host: `http://host.docker.internal:11434/v1` |
| `OPENAI_API_KEY` | none | Bearer token. Required for OpenAI; leave unset for local servers without auth |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model. Changing it changes the vector size: use a new collection and `FORCE_REINGEST=true` |
| `EMBEDDING_DIMENSIONS` | model's size | Shorter vectors, for `text-embedding-3-*` models only; `0` or unset for the full size |
| `EMBEDDING_CONTEXT_LENGTH` | `8191` | Model context in tokens, at least 1; texts are cut to about 0.75 words per token |
| `EMBEDDING_BATCH_SIZE` | `32` | Chunks per embeddings request, at least 1 |
| `CHUNK_SIZE` | `256` | Maximum words per chunk, at least 1 |
| `CHUNK_OVERLAP` | `50` | Words shared by consecutive chunks; less than `CHUNK_SIZE` |
| `PARENT_CHUNK_RATIO` | `4` | Chunks per `parent_chunk_index` group, at least 1 |
| `POLL_INTERVAL_SECONDS` | `3600` | Seconds between runs; `0` runs once and exits |
| `FORCE_REINGEST` | `false` | `true`/`false` (also `1`/`0`, `yes`/`no`, `on`/`off`): re-ingest pages even if `updatedAt` is unchanged |
| `PAGE_DELAY_SECONDS` | `0.1` | Pause between pages, at least 0 |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL` |

Exit codes: `0` success (or stopped by SIGTERM), `1` Qdrant or the wiki
unreachable, `2` bad configuration, or a one-shot run in which some pages
failed.

## Query tool (`wikijs2rag-query`)

Uses `QDRANT_*`, `OPENAI_BASE_URL`, `OPENAI_API_KEY` and the `EMBEDDING_*`
settings as above, except that `QDRANT_HOST` defaults to
`localhost`. Command-line flags override these:

| Variable | Flag | Default | Meaning |
|----------|------|---------|---------|
| `QUERY_LIMIT` | `--limit` | `5` | Number of results, at least 1 |
| `QDRANT_COLLECTION` | `--collection` | `wiki` | Collection to search |
| `QUERY_SHOW_TEXT` | `--show-text` | `false` | Print each matched chunk's text |
| `QUERY_MIN_SCORE` | `--min-score` | `0.0` | Hide results below this cosine similarity, 0-1 |
| `QUERY_WIDTH` | `--width` | `100` | Text wrapping width, at least 20 |
| `NO_COLOR` | | unset | Any value turns off colored output ([no-color.org](https://no-color.org/)) |

## Docker compose (not read by the app)

| Variable | Default | Meaning |
|----------|---------|---------|
| `APP_VERSION` | `latest` | Release of `ghcr.io/romkey/wikijs2rag` that `docker-compose.prod.yaml` runs |
| `QDRANT_REST_PORT` | `6333` | Host port for Qdrant's REST API and dashboard |
| `QDRANT_GRPC_PORT` | `6334` | Host port for Qdrant's gRPC API |
