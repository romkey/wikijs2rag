# wiki2rag

Ingests all **public** pages from a [Wiki.js](https://js.wiki/) instance into a
[Qdrant](https://qdrant.tech/) vector store, ready for use in any RAG
(Retrieval-Augmented Generation) pipeline.

## How it works

```
Wiki.js GraphQL API
      │
      ▼
  wiki_client.py   ← lists published, non-private pages
      │
      ▼
   chunker.py      ← splits markdown/HTML by headers then by word window
      │
      ▼
   embedder.py     ← encodes chunks (local sentence-transformers or OpenAI)
      │
      ▼
    store.py       ← upserts into Qdrant (deletes stale chunks first)
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

| Variable        | Default              | Description                                |
|-----------------|----------------------|--------------------------------------------|
| `WIKI_API_KEY`  | *(empty)*            | API key if the wiki requires auth           |
| `QDRANT_COLLECTION` | `wiki`           | Collection name (one per wiki is sensible)  |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Any sentence-transformers model name       |

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

The first build downloads the embedding model and bakes it into the image
(~500 MB with `all-MiniLM-L6-v2`).  Subsequent runs start in seconds.

The script is idempotent – re-running it refreshes changed pages and removes
stale chunks.

---

## Querying from other applications

Any language/framework that has a Qdrant client can query the store directly.

**Python example:**

```python
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient(host="localhost", port=6333)
model  = SentenceTransformer("all-MiniLM-L6-v2")

query_vector = model.encode("How do I reset my password?", normalize_embeddings=True).tolist()

results = client.search(
    collection_name="wiki",
    query_vector=query_vector,
    limit=5,
    with_payload=True,
)

for hit in results:
    print(hit.score, hit.payload["page_title"], hit.payload["page_url"])
    print(hit.payload["text"][:300])
    print()
```

**Payload fields on every point:**

| Field         | Type        | Description                              |
|---------------|-------------|------------------------------------------|
| `page_id`     | int         | Wiki.js page ID                          |
| `page_title`  | str         | Page title                               |
| `page_path`   | str         | Wiki-relative path, e.g. `/guides/setup` |
| `page_url`    | str         | Full URL                                 |
| `description` | str         | Page description / excerpt               |
| `tags`        | list[str]   | Wiki.js tags                             |
| `updated_at`  | str         | ISO-8601 last-modified timestamp         |
| `section`     | str         | Nearest header above this chunk          |
| `chunk_index` | int         | Position within the page                 |
| `text`        | str         | Raw chunk text (for display / re-ranking)|

---

## Keeping the data fresh

The ingestion job is intentionally **one-shot** – run it on a schedule to keep
Qdrant up to date.

### cron example (every night at 02:00)

```cron
0 2 * * * cd /opt/wiki2rag && docker compose run --rm wiki2rag >> /var/log/wiki2rag.log 2>&1
```

### GitHub Actions example

```yaml
on:
  schedule:
    - cron: "0 2 * * *"

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

---

## Switching to OpenAI embeddings

```dotenv
EMBEDDING_BACKEND=openai
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-...
```

Then rebuild: `docker compose build wiki2rag`.

> **Note:** Change the collection name if you switch models – vector dimensions
> differ and the old collection cannot be reused.

---

## File layout

```
wiki2rag/
├── src/
│   ├── main.py          # entry point / orchestration
│   ├── wiki_client.py   # Wiki.js GraphQL client
│   ├── chunker.py       # markdown/HTML → overlapping text chunks
│   ├── embedder.py      # local or OpenAI embedding backends
│   └── store.py         # Qdrant wrapper
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Wiki.js API key

1. Log in as an administrator and go to **Administration → API Access**.
2. Create a new key with **Read** permission on **Pages**.
3. Copy the token into `.env` as `WIKI_API_KEY`.

An API key is only required if your wiki restricts guest access to the GraphQL
endpoint.  Many public wikis work fine without one.
