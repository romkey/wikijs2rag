# wiki2rag

Ingests all **public** pages from a [Wiki.js](https://js.wiki/) instance into a
[Qdrant](https://qdrant.tech/) vector store, ready for use in any RAG
(Retrieval-Augmented Generation) pipeline.

## How it works

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
   embedder.py     ← encodes chunks (local fastembed/ONNX or OpenAI)
      │
      ▼
    store.py       ← upserts into Qdrant with payload indexes
```

Qdrant persists the vectors in a named Docker volume.  Any application that
can speak to Qdrant's REST or gRPC API can query the same data – no coupling
to this ingestion code required.

---

## Quick start

### 1 – Copy and edit the config

```bash
cp .env.example .env
$EDITOR .env          # set WIKI_URL at minimum
```

**Required:**

| Variable   | Description                      |
|------------|----------------------------------|
| `WIKI_URL` | Base URL of your Wiki.js instance |

**Optional but common:**

| Variable             | Default                  | Description                                |
|----------------------|--------------------------|--------------------------------------------|
| `WIKI_API_KEY`       | *(empty)*                | API key if the wiki requires auth          |
| `QDRANT_COLLECTION`  | `wiki`                   | Collection name (one per wiki is sensible) |
| `EMBEDDING_MODEL`    | `BAAI/bge-small-en-v1.5` | Any fastembed-supported model name        |
| `CHUNK_SIZE`         | `256`                    | Max words per chunk                        |
| `FORCE_REINGEST`     | `false`                  | Set `true` to re-ingest unchanged pages    |

See `.env.example` for the full list.

### 2 – Start Qdrant

```bash
docker compose up -d qdrant
```

Qdrant's REST API is now available at `http://localhost:6333`.
Browse the built-in dashboard at `http://localhost:6333/dashboard`.

### 3 – Build and run the ingestion

```bash
docker compose build wiki2rag    # only needed once (or after code changes)
docker compose run --rm wiki2rag
```

The first build downloads the embedding model and bakes it into the image.
Subsequent runs start in seconds.

Ingestion is **incremental** – only pages whose `updatedAt` timestamp has
changed are re-fetched and re-embedded.  Set `FORCE_REINGEST=true` to
override this and re-ingest everything.

---

## Querying

### Command-line query tool

`query.py` lets you search the collection directly from the terminal.

**Via Docker (recommended):**

```bash
# Basic search
docker compose run --rm query "how do I reset my password?"

# More results
docker compose run --rm query --limit 10 "event manager setup"

# Show matching text under each result
docker compose run --rm query --show-text "access control"

# Filter out weak matches
docker compose run --rm query --min-score 0.6 "door code"

# Different collection
docker compose run --rm query --collection wiki-internal "onboarding"

# All options
docker compose run --rm query --help
```

**Locally (outside Docker):**

```bash
# Point at localhost since we're outside the compose network
QDRANT_HOST=localhost python src/query.py "how do I reset my password?"
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--limit N` | `5` | Number of results |
| `--show-text` | off | Print matched chunk text |
| `--min-score F` | `0.0` | Hide results below this similarity (0–1) |
| `--collection NAME` | `wiki` | Qdrant collection to search |
| `--width N` | `100` | Terminal width for text wrapping |

---

## Querying from other applications

Any language/framework with a Qdrant client can query the store directly.

**Python example:**

```python
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
model  = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

query_vector = list(model.embed(["How do I reset my password?"]))[0].tolist()

results = client.query_points(
    collection_name="wiki",
    query=query_vector,
    limit=5,
    with_payload=True,
).points

for hit in results:
    p = hit.payload
    print(f"{hit.score:.3f}  {p['page_title']}  {p['page_url']}")
    # Use p["context"] for LLM grounding – it includes title, section, description
    print(p["context"][:300])
    print()
```

### Payload fields on every point

| Field                | Type         | Description                                           |
|----------------------|--------------|-------------------------------------------------------|
| `page_id`            | int          | Wiki.js page ID                                       |
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

### Chatbot integration tips

- **Use `context` for LLM grounding**, not `text`.  The `context` field includes
  the page title, description, and section breadcrumb, giving the LLM enough
  surrounding information to produce accurate answers.

- **Filter by tags** for scoped search.  If your wiki tags pages by topic
  (e.g. "safety", "events", "membership"), pass a Qdrant filter to constrain
  results:
  ```python
  from qdrant_client.models import Filter, FieldCondition, MatchValue
  results = client.query_points(
      collection_name="wiki",
      query=query_vector,
      query_filter=Filter(must=[
          FieldCondition(key="tags", match=MatchValue(value="safety"))
      ]),
      ...
  )
  ```

- **Small-to-big retrieval**: search returns small, focused chunks.  Use
  `parent_chunk_index` to fetch all sibling chunks for broader context, or
  follow `prev_chunk_id` / `next_chunk_id` to expand the window.

- **Deduplicate results**: if multiple pages share boilerplate text, several
  top-k results may be near-identical.  Group results by `content_hash` and
  keep only the best-scoring hit per hash before sending context to the LLM.

---

## Keeping the data fresh

Ingestion is **incremental** by default – only pages whose `updatedAt`
timestamp has changed since the last run are re-fetched and re-embedded.
This makes it practical to run every few hours instead of nightly.

### cron example (every 4 hours)

```cron
0 */4 * * * cd /opt/wiki2rag && docker compose run --rm wiki2rag >> /var/log/wiki2rag.log 2>&1
```

### GitHub Actions example

```yaml
on:
  schedule:
    - cron: "0 */4 * * *"

jobs:
  ingest:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v4
      - run: docker compose run --rm wiki2rag
        env:
          WIKI_URL: ${{ secrets.WIKI_URL }}
          WIKI_API_KEY: ${{ secrets.WIKI_API_KEY }}
```

### Force full re-ingestion

```bash
FORCE_REINGEST=true docker compose run --rm wiki2rag
```

---

## Using Ollama for embeddings

If you're already running [Ollama](https://ollama.com/) locally, you can use
it for embeddings instead of the built-in fastembed model.  No extra Python
packages are needed — the Ollama backend talks to Ollama's HTTP API directly.

```bash
# Pull an embedding model in Ollama first
ollama pull nomic-embed-text
```

```dotenv
EMBEDDING_BACKEND=ollama
EMBEDDING_MODEL=nomic-embed-text
OLLAMA_URL=http://host.docker.internal:11434
```

> `host.docker.internal` resolves to the host machine from inside Docker
> containers on macOS and Windows.  On Linux, use `--add-host=host.docker.internal:host-gateway`
> or set `OLLAMA_URL` to the actual LAN IP.

Popular Ollama embedding models:

| Model | Dimensions | Notes |
|---|---|---|
| `nomic-embed-text` | 768 | Good balance of speed and quality (default) |
| `all-minilm` | 384 | Fast, small |
| `mxbai-embed-large` | 1024 | Highest quality |
| `snowflake-arctic-embed` | 1024 | Strong retrieval performance |

No rebuild needed — just change `.env` and re-run ingestion with
`FORCE_REINGEST=true` (different model = different vectors).

> **Note:** Change the collection name when switching embedding models –
> vector dimensions differ and the old collection cannot be reused.

## Switching to OpenAI embeddings

```dotenv
EMBEDDING_BACKEND=openai
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-...
```

No rebuild needed — just change `.env` and re-run ingestion.

> **Note:** Change the collection name when switching embedding models –
> vector dimensions differ and the old collection cannot be reused.

---

## File layout

```
wiki2rag/
├── src/
│   ├── main.py          # ingestion entry point / orchestration
│   ├── query.py         # command-line query tool
│   ├── wiki_client.py   # Wiki.js GraphQL client (+ HTML scraping fallback)
│   ├── chunker.py       # structure-aware text chunking
│   ├── embedder.py      # embedding backends (fastembed, Ollama, OpenAI)
│   ├── store.py         # Qdrant wrapper with payload indexes
│   └── version.py       # version from VERSION file
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── VERSION
└── .env.example
```

## Wiki.js API key

1. Log in as an administrator and go to **Administration → API Access**.
2. Create a new key with **Read** permission on **Pages**.
3. Copy the token into `.env` as `WIKI_API_KEY`.

An API key is only required if your wiki restricts guest access to the GraphQL
endpoint.  Many public wikis work fine without one.
