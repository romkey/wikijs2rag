# wikijs2rag

[![CI](https://github.com/romkey/wikijs2rag/actions/workflows/ci.yaml/badge.svg)](https://github.com/romkey/wikijs2rag/actions/workflows/ci.yaml)
[![Lint](https://github.com/romkey/wikijs2rag/actions/workflows/lint.yaml/badge.svg)](https://github.com/romkey/wikijs2rag/actions/workflows/lint.yaml)
[![Python 3.15](https://img.shields.io/badge/python-3.15-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Ingests all **public** pages from a [Wiki.js](https://js.wiki/) instance into a
[Qdrant](https://qdrant.tech/) vector store, ready for any RAG
(Retrieval-Augmented Generation) pipeline. Embeddings come from
[Hugging Face Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference)
running [ModernBERT Embed](https://huggingface.co/nomic-ai/modernbert-embed-base)
by default, or from any OpenAI-compatible embeddings API, including OpenAI's.
It runs
continuously, re-embedding only pages that changed, and stores each chunk with
enough context (title, section path, neighbours) for a chatbot to ground its
answers, plus wiki-level statistics such as page counts and contributors.

> **AI disclosure:** This project was written by AI under human oversight,
> following [these rules](https://github.com/romkey/python-ai-rules/blob/main/RULES.md).

## What's here

| Path | What it is |
|------|------------|
| `src/wikijs2rag/__main__.py` | The `wikijs2rag` command: ingests on start, then every `POLL_INTERVAL_SECONDS`; stops cleanly on SIGTERM |
| `src/wikijs2rag/ingest.py` | One ingestion run: fetch, chunk, embed, store; builds chunk payloads and wiki metadata chunks |
| `src/wikijs2rag/wiki_client.py` | Wiki.js GraphQL client: lists published, non-private pages; retries; falls back to scraping HTML |
| `src/wikijs2rag/chunker.py` | Structure-aware splitting: headers → tables/lists/code kept whole → overlapping word windows |
| `src/wikijs2rag/embedder.py` | OpenAI-compatible `/v1/embeddings` client with batching and truncation to the model's context |
| `src/wikijs2rag/store.py` | Qdrant wrapper: collection, payload indexes, replace-per-page upserts |
| `src/wikijs2rag/query.py` | The `wikijs2rag-query` command-line search tool |
| `src/wikijs2rag/config.py` | Settings from environment variables, validated at startup |
| `tests/` | pytest tests; no real Wiki.js, embeddings API or Qdrant needed |
| `docs/configuration.md` | Every environment variable |
| `docs/overview.md` | A short map of the project |

## Technical details

```
Wiki.js GraphQL API
      │
      ▼
  wiki_client.py   ← lists published, non-private pages (+ HTML scraping fallback)
      │
      ▼
   chunker.py      ← structure-aware splitting: headers → tables/lists → word window
      │
      ▼
   embedder.py     ← encodes chunks via an OpenAI-compatible embeddings API
      │
      ▼
    store.py       ← upserts into Qdrant with payload indexes
```

- **Incremental:** a page is re-fetched and re-embedded only when its
  `updatedAt` differs from what's stored. `FORCE_REINGEST=true` redoes all.
- **Per-page replacement:** a page's old points are deleted before its new
  ones are written, so edits never leave stale chunks behind.
- **HTML fallback:** when Wiki.js's GraphQL `pages.single` refuses a page
  that guests can see in the browser (error 6013), the rendered page is
  scraped instead.
- **Wiki metadata:** after the pages, it stores chunks describing the wiki
  itself (page count, contributors, tags, recently updated, newest pages and a
  full page listing), so a chatbot can answer "how many pages are there?" or
  "who are the top contributors?".

Qdrant keeps the vectors in a Docker volume. Anything that speaks Qdrant's
REST or gRPC API can query them; nothing depends on this code.

### Querying from the command line

```bash
docker compose -f docker-compose.prod.yaml run --rm query "how do I reset my password?"
docker compose -f docker-compose.prod.yaml run --rm query --limit 10 --show-text "event manager setup"
docker compose -f docker-compose.prod.yaml run --rm query --min-score 0.6 --collection wiki-internal "door code"
```

| Flag | Default | Description |
|---|---|---|
| `--limit N` | `5` | Number of results |
| `--show-text` | off | Print matched chunk text |
| `--min-score F` | `0.0` | Hide results below this similarity (0–1) |
| `--collection NAME` | `wiki` | Qdrant collection to search |
| `--width N` | `100` | Terminal width for text wrapping |
| `--version` | | Print the version |

Each flag has an environment variable default; see
[docs/configuration.md](docs/configuration.md).

### Querying from other applications

```python
import httpx
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

# Embed the query with the same model and server used for ingestion (here
# TEI on its host port) and the model's query prefix: ModernBERT Embed
# expects "search_query: ". Add an Authorization header if the server needs a key.
resp = httpx.post(
    "http://localhost:8080/v1/embeddings",
    json={
        "model": "nomic-ai/modernbert-embed-base",
        "input": ["search_query: How do I reset my password?"],
    },
)
query_vector = resp.json()["data"][0]["embedding"]

results = client.query_points(
    collection_name="wiki",
    query=query_vector,
    limit=5,
    with_payload=True,
).points

for hit in results:
    p = hit.payload
    print(f"{hit.score:.3f}  {p['page_title']}  {p['page_url']}")
    # Use p["context"] for LLM grounding – includes title, section, description
    print(p["context"][:300])
```

### Payload fields on every point

| Field                | Type         | Description                                           |
|----------------------|--------------|-------------------------------------------------------|
| `page_id`            | int          | Wiki.js page ID (`0` for metadata chunks)             |
| `page_title`         | str          | Page title                                            |
| `page_path`          | str          | Wiki-relative path, e.g. `guides/setup`               |
| `page_url`           | str          | Full URL                                              |
| `description`        | str          | Page description / excerpt                            |
| `tags`               | list[str]    | Wiki.js tags                                          |
| `updated_at`         | str          | ISO-8601 last-modified timestamp                      |
| `section`            | str          | Nearest header above this chunk                       |
| `section_breadcrumb` | str          | Full header path, e.g. `"Setup > Docker > Compose"`   |
| `chunk_index`        | int          | Position within the page                              |
| `chunk_id`           | str          | Unique UUID for this chunk                            |
| `text`               | str          | Raw chunk text (for embedding / display)              |
| `context`            | str          | Enriched text with title+section+description for LLMs |
| `content_hash`       | str          | SHA-256 prefix for deduplication                      |
| `prev_chunk_id`      | str \| null  | UUID of the previous chunk (null for first)           |
| `next_chunk_id`      | str \| null  | UUID of the next chunk (null for last)                |
| `parent_chunk_index` | int          | Index of the parent chunk group                       |
| `total_chunks`       | int          | Total chunks in the page                              |
| `is_meta`            | str \| null  | `"true"` for wiki metadata chunks, absent for regular |
| `meta_type`          | str \| null  | Metadata category: `overview`, `contributors`, `tags`, `recent`, `newest`, `page_listing` |

### Chatbot integration tips

- **Use `context` for LLM grounding**, not `text`. It adds the page title,
  description and section breadcrumb, so the LLM knows where the chunk came
  from.
- **Filter by tags** for scoped search:
  ```python
  from qdrant_client.models import Filter, FieldCondition, MatchValue

  results = client.query_points(
      collection_name="wiki",
      query=query_vector,
      query_filter=Filter(must=[FieldCondition(key="tags", match=MatchValue(value="safety"))]),
  )
  ```
- **Small-to-big retrieval**: search returns small, focused chunks. Use
  `parent_chunk_index` to fetch sibling chunks for broader context, or follow
  `prev_chunk_id` / `next_chunk_id` to widen the window.
- **Deduplicate results**: if pages share boilerplate, several top results
  may be near-identical. Group by `content_hash` and keep the best hit of each.
- **Wiki metadata questions** ("how many pages?", "what changed recently?")
  are answered by the metadata chunks (`is_meta` = `"true"`), which are
  embedded alongside regular content, so no special filter is needed.

### Embedding models

Hugging Face models, served by TEI (set `EMBEDDING_MODEL` to the model id;
compose starts TEI with it). Their query and document prefixes are applied
automatically; see [docs/configuration.md](docs/configuration.md#embedding-prefixes).

| Model | Dimensions | Context | Notes |
|---|---|---|---|
| `nomic-ai/modernbert-embed-base` | 768 | 8192 | ModernBERT-based, long context (default) |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | 8192 | Same prefixes as ModernBERT Embed |
| `intfloat/e5-large-v2` | 1024 | 512 | Set `EMBEDDING_CONTEXT_LENGTH=512` |
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | 512 | Query prefix only; set `EMBEDDING_CONTEXT_LENGTH=512` |
| `Qwen/Qwen3-Embedding-0.6B` | 1024 | 32768 | Instruction-style query prefix |

OpenAI (`OPENAI_BASE_URL=https://api.openai.com/v1`, `OPENAI_API_KEY` set,
`COMPOSE_PROFILES` without `tei`): `text-embedding-3-small` (1536) or
`text-embedding-3-large` (3072); no prefixes.

Changing models changes the vector size, which needs a new collection: set a
new `QDRANT_COLLECTION` and run with `FORCE_REINGEST=true`.

## Running it

Requires Docker. The compose files run
[Hugging Face Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference)
(TEI) serving `EMBEDDING_MODEL` next to the app, when `COMPOSE_PROFILES=tei`
(as in `.env.example`). The first start downloads the model. TEI needs about
1.5 GB of memory with the default `TEI_MAX_BATCH_TOKENS=2048`. On ARM machines
(Apple Silicon, Graviton) set
`TEI_IMAGE=ghcr.io/huggingface/text-embeddings-inference:cpu-arm64-1.9.4`.
To use OpenAI or another server instead, remove `COMPOSE_PROFILES` and set
`OPENAI_BASE_URL` (and `OPENAI_API_KEY` if it needs one).

Configuration comes from environment variables, kept in `.env` (see
[docs/configuration.md](docs/configuration.md)); set at least `WIKI_URL`:

```bash
cp .env.example .env   # then edit it
docker compose -f docker-compose.prod.yaml up -d
```

That runs the latest release from `ghcr.io/romkey/wikijs2rag` next to Qdrant;
set `APP_VERSION=X.Y.Z` to pin one. It ingests every public page, then checks
for changes every `POLL_INTERVAL_SECONDS` (default an hour). Qdrant's REST
API and dashboard are at http://localhost:6333/dashboard.

For a one-shot run (ingest once and exit), or a full re-ingest:

```bash
POLL_INTERVAL_SECONDS=0 docker compose -f docker-compose.prod.yaml run --rm app
FORCE_REINGEST=true POLL_INTERVAL_SECONDS=0 docker compose -f docker-compose.prod.yaml run --rm app
```

### Wiki.js API key

An API key is only needed if your wiki restricts guest access to the GraphQL
endpoint. To make one:

1. Log in as an administrator and go to **Administration → API Access**.
2. Create a key with **Read** permission on **Pages**.
3. Put it in `.env` as `WIKI_API_KEY`.

### With pip

Needs Python 3.15 or newer, plus a Qdrant you can reach. Install into a
virtual environment from a release's wheel:

```bash
python3 -m venv ~/.venvs/wikijs2rag
~/.venvs/wikijs2rag/bin/pip install https://github.com/romkey/wikijs2rag/releases/download/vX.Y.Z/wikijs2rag-X.Y.Z-py3-none-any.whl
~/.venvs/wikijs2rag/bin/wikijs2rag --version
```

Set its environment variables yourself (see
[docs/configuration.md](docs/configuration.md)); it doesn't read `.env`.

## Development

With Docker, nothing else to install:

```bash
docker compose up --build                                  # ingest from the working tree, with Qdrant
docker compose run --rm query "some question"              # search it
docker compose -f docker-compose.test.yaml run --rm test   # tests with coverage
docker compose -f docker-compose.lint.yaml run --rm lint   # every lint check, as CI runs them
```

With [uv](https://docs.astral.sh/uv/), which installs the exact versions CI
tests (`uv.lock`):

```bash
uv sync
uv run pytest
uv run mypy
uv run wikijs2rag
```

With pip (25.1 or newer, for `--group`). pip doesn't read `uv.lock`, so it
may pick newer dependency versions than CI tested:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e . --group dev
pytest
mypy
wikijs2rag
```

Either way, `uvx pre-commit install` (or `pip install pre-commit &&
pre-commit install`) runs the lint checks on every commit. Tests fake Wiki.js
and the embeddings API over HTTP and use Qdrant's in-memory mode, so no services are
needed.

Releases: run the **Release** workflow from the Actions tab, then merge the
PR it opens. See [CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE)
