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
| `OLLAMA_URL` | `http://localhost:11434` | Ollama base URL. In Docker, `http://host.docker.internal:11434` reaches the host |
| `OLLAMA_API_KEY` | none | Bearer token for hosted or proxied Ollama APIs |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model. Changing it changes the vector size: use a new collection and `FORCE_REINGEST=true` |
| `EMBEDDING_CONTEXT_LENGTH` | `8192` | Model context in tokens, at least 1; texts are cut to about 0.75 words per token |
| `EMBEDDING_BATCH_SIZE` | `32` | Chunks per Ollama request, at least 1 |
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

Uses `QDRANT_*`, `OLLAMA_URL`, `OLLAMA_API_KEY`, `EMBEDDING_MODEL` and
`EMBEDDING_CONTEXT_LENGTH` as above, except that `QDRANT_HOST` defaults to
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
