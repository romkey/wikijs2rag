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
| `OPENAI_BASE_URL` | `http://localhost:8080/v1` (compose: `http://tei/v1`) | Base URL of an OpenAI-compatible embeddings API (the part before `/embeddings`): Hugging Face TEI, OpenAI (`https://api.openai.com/v1`), vLLM, … |
| `OPENAI_API_KEY` | none | Bearer token. Required for OpenAI; leave unset for TEI and other servers without auth |
| `EMBEDDING_MODEL` | `nomic-ai/modernbert-embed-base` | Embedding model: a Hugging Face model id for TEI (compose starts TEI with it) or an OpenAI model name. Changing it changes the vectors: use a new collection and `FORCE_REINGEST=true` |
| `EMBEDDING_QUERY_PREFIX` | the model's (see below) | Text put before each search query before embedding. `none` for no prefix |
| `EMBEDDING_DOCUMENT_PREFIX` | the model's (see below) | Text put before each chunk before embedding. `none` for no prefix. Changing it needs `FORCE_REINGEST=true` |
| `EMBEDDING_DIMENSIONS` | model's size | Shorter vectors, for models that accept the `dimensions` parameter (OpenAI `text-embedding-3-*`); `0` or unset for the full size |
| `EMBEDDING_CONTEXT_LENGTH` | `8192` | Model context in tokens, at least 1; texts (prefix included) are cut to about 0.75 words per token |
| `EMBEDDING_BATCH_SIZE` | `32` | Chunks per embeddings request, at least 1 |
| `CHUNK_SIZE` | `256` | Maximum words per chunk, at least 1 |
| `CHUNK_OVERLAP` | `50` | Words shared by consecutive chunks; less than `CHUNK_SIZE` |
| `PARENT_CHUNK_RATIO` | `4` | Chunks per `parent_chunk_index` group, at least 1 |
| `POLL_INTERVAL_SECONDS` | `3600` | Seconds between runs; `0` runs once and exits |
| `FORCE_REINGEST` | `false` | `true`/`false` (also `1`/`0`, `yes`/`no`, `on`/`off`): re-ingest pages even if `updatedAt` is unchanged |
| `PAGE_DELAY_SECONDS` | `0.1` | Pause between pages, at least 0 |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL` |

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
| `COMPOSE_PROFILES` | unset (`.env.example`: `tei`) | `tei` runs the Hugging Face Text Embeddings Inference service; leave it out to use another embeddings server |
| `TEI_IMAGE` | `ghcr.io/huggingface/text-embeddings-inference:cpu-1.9.4` | TEI image; on ARM use `…:cpu-arm64-1.9.4`, on NVIDIA GPUs one of TEI's GPU images |
| `TEI_MAX_BATCH_TOKENS` | `2048` | Longest input TEI embeds, in tokens; longer inputs are truncated. Memory grows with its square: about 1.5 GB at 2048, 3 GB at 4096, over 8 GB at 8192. Keep `EMBEDDING_CONTEXT_LENGTH` no larger |
| `TEI_PORT` | `8080` | Host port for TEI, for the query tool run natively and for RAG apps embedding their queries |
| `QDRANT_REST_PORT` | `6333` | Host port for Qdrant's REST API and dashboard |
| `QDRANT_GRPC_PORT` | `6334` | Host port for Qdrant's gRPC API |
