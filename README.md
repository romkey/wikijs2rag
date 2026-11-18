# wikijs2rag

[![CI](https://github.com/romkey/wikijs2rag/actions/workflows/ci.yaml/badge.svg)](https://github.com/romkey/wikijs2rag/actions/workflows/ci.yaml)
[![Lint](https://github.com/romkey/wikijs2rag/actions/workflows/lint.yaml/badge.svg)](https://github.com/romkey/wikijs2rag/actions/workflows/lint.yaml)
[![Python 3.15](https://img.shields.io/badge/python-3.15-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Syncs a [Wiki.js](https://js.wiki/) instance into tiered
[Qdrant](https://qdrant.tech/) collections for retrieval: `hs_public` for
pages anyone can read and `hs_member` for everything a member can read. It is
the wiki source of a larger retrieval pipeline; the collections, the payload
fields and the staging database are shared with the other sources' ingesters.
Embeddings come from
[Hugging Face Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference)
running [ModernBERT Embed](https://huggingface.co/nomic-ai/modernbert-embed-base)
by default, or any OpenAI-compatible embeddings API; summaries, context
lines and classification come from any OpenAI-compatible chat model. All of
it can run locally.

> **AI disclosure:** This project was written by AI under human oversight,
> following [these rules](https://github.com/romkey/python-ai-rules/blob/main/RULES.md).

## What's here

| Path | What it is |
|------|------------|
| `src/wikijs2rag/__main__.py` | The `wikijs2rag` command: a run on start, then every `POLL_INTERVAL_SECONDS`; `--full`, `--reindex` |
| `src/wikijs2rag/ingest.py` | One sync run: two crawls, staging, indexing, reconcile |
| `src/wikijs2rag/wiki_client.py` | Wiki.js GraphQL client: one per crawl; admin-rights probe, guest check, file fetch, HTML fallback |
| `src/wikijs2rag/staging.py` | Staging database (Postgres): pages, links, attachments, LLM cache, review queue, quarantine, sync status, entity registry |
| `src/wikijs2rag/content.py` | Page to clean markdown; heading anchors, links, attachments, images |
| `src/wikijs2rag/sanitize.py` | Secret scanner and network-identifier placeholders |
| `src/wikijs2rag/enrich.py` | Page kind, summary, topics, chunk context, exclusion classifier |
| `src/wikijs2rag/linker.py` | Tags chunks with registry entities (assets, locations, trainings, …) |
| `src/wikijs2rag/chunker.py` | Structure-first chunking by heading, then size |
| `src/wikijs2rag/attachments.py` | PDFs and text files a page links to; image descriptions |
| `src/wikijs2rag/llm.py` | OpenAI-compatible chat client with a result cache |
| `src/wikijs2rag/embedder.py` | OpenAI-compatible `/v1/embeddings` client |
| `src/wikijs2rag/sparse.py` | Keyword (sparse) vectors |
| `src/wikijs2rag/store.py` | Qdrant: tier collections, named dense + sparse vectors, payload indexes |
| `src/wikijs2rag/query.py` | The `wikijs2rag-query` hybrid search tool |
| `src/wikijs2rag/report.py` | The `wikijs2rag-report` status, reports and canary check |
| `src/wikijs2rag/topics.txt` | Default shared topic taxonomy |
| `tests/` | pytest tests; no real Wiki.js, model servers, Postgres or Qdrant needed |
| `docs/configuration.md` | Every environment variable |
| `docs/overview.md` | A short map of the project |

## Technical details

```
anonymous crawl (no credentials) ──┐
                                   ├─> staging (Postgres) ─> sanitize ─> classify ─> chunk ─> link ─> embed
member crawl (members-group key) ──┘                                                                  │
                                              hs_public  <── public tier only ─────────────────────────┤
                                              hs_member  <── public and member tiers ──────────────────┘
```

**Access tiers are structural.** The anonymous crawl uses no credentials:
every page it can see is `tier: public`. The member crawl uses an API key
scoped to a group with exactly member-level read rights: pages only it can
see are `tier: member` and go to `hs_member` only. No admin credential is
ever configured, so admin pages are unreachable rather than filtered; a run
whose key can list API keys (an admin-only query) stops. Tier is recomputed
every run, so a page made member-only leaves `hs_public` on the next run.
Files linked from a page are fetched with the page's crawl credentials, so
Wiki.js decides who may read them too. If guests can't use the GraphQL
listing, each member-crawl page is fetched as a guest instead to decide
whether it is public.

**What is ingested.** Every published, non-private page both crawls can see,
in every locale: path, locale, title, description, tags, timestamps, source
and rendered content, table of contents, outbound links (internal and
external), and attachments: linked PDFs and text files (indexed as children
of the page, PDFs per page) and images (staged with alt text and caption;
with `VISION_MODEL`, signs, diagrams and whiteboards get a machine-generated
description). Never: page history, comments, user lists, author names, or
anything under `WIKI_EXCLUDED_PATHS` (default `admin`).

**Processing.**
- Markdown pages keep their markdown; other content types are converted
  from the rendered HTML. Code blocks, tables and lists stay intact.
- Sanitization: a secret scanner redacts credentials (and logs them to the
  quarantine report, without the value); IP and MAC addresses, internal
  hostnames and Home Assistant entity IDs inside code blocks become
  placeholders. Member pages (and member-only files) go through an LLM
  exclusion classifier; flagged ones go to the review queue, are kept out
  of the index, and are indexed once a human sets their review row to
  `approved`. If the classifier can't be reached, the page waits for the
  next run: it fails closed.
- Each page is classified (`page_kind`: `policy`, `procedure`, `training`,
  `equipment`, `build-instructions`, `project`, `area`, `meeting-notes`,
  `reference`, `other`), summarized, and mapped to the shared topic
  taxonomy. Authority is 5 for policy and training, 4 otherwise.
- Chunks follow headings first, then size (about 400 tokens, at most 800;
  code blocks are never split; long tables repeat their caption and header
  row). `heading_path` is the page tree breadcrumb plus the heading chain;
  `url` opens the page at the heading's anchor. Each chunk gets a 1-2
  sentence `context` line, embedded with its text. Every page also gets a
  summary point.
- The linker matches registry aliases (`entity_aliases`, edited by humans)
  in each chunk; an LLM resolves aliases shared by several entities.
  Equipment, area, training and build-instruction pages are recorded in
  `entity_documents` as the documentation of the entity they name (for
  example an asset's "how to use" page).
- Every ingest-time LLM result is cached by content hash.

**Sync.** Each run takes a per-source lock and records a status row.
Incremental runs (every `POLL_INTERVAL_SECONDS`) refetch pages whose
`updatedAt` or tier changed and skip unchanged content by hash. A full
reconcile (every `RECONCILE_INTERVAL_SECONDS`, or `--full`) refetches
everything, stamps every document seen with the run ID and deletes this
source's points with an older one. If a full run sees fewer than half the
pages the previous full run saw, it deletes nothing and fails instead.
`wikijs2rag --reindex` rebuilds both collections from staging (after a
chunking change, say) without contacting the wiki.

**Vectors.** Each point has a dense vector `dense` and a sparse keyword
vector `sparse` (Qdrant's IDF modifier; scheme in `sparse.py`). Query both
and fuse with reciprocal rank fusion, then rerank. Collection metadata
records the embedding model, its dimensions and the sparse scheme, and a run
with a different model stops: changing the model means re-embedding every
source into new collections.

### Querying from the command line

```bash
docker compose -f docker-compose.prod.yaml run --rm query "how do I reset my password?"
docker compose -f docker-compose.prod.yaml run --rm query --tier member --limit 10 --show-text "door procedure"
```

| Flag | Default | Description |
|---|---|---|
| `--tier public\|member` | `public` | Collection to search |
| `--limit N` | `5` | Number of results |
| `--show-text` | off | Print matched chunk text |
| `--min-score F` | `0.0` | Hide results below this fused score |
| `--collection NAME` | the tier's | Search another collection |
| `--width N` | `100` | Terminal width for text wrapping |
| `--version` | | Print the version |

Each flag has an environment variable default; see
[docs/configuration.md](docs/configuration.md).

### Reports

```bash
docker compose -f docker-compose.prod.yaml run --rm report status    # last run, counts, recent runs
docker compose -f docker-compose.prod.yaml run --rm report stale     # not updated in 18 months (--months N)
docker compose -f docker-compose.prod.yaml run --rm report flagged   # review queue, redacted secrets
docker compose -f docker-compose.prod.yaml run --rm report links     # broken internal links
docker compose -f docker-compose.prod.yaml run --rm report canaries  # exit 1 if a canary leaked
```

### Querying from other applications

```python
import zlib, re
import httpx
from qdrant_client import QdrantClient, models

client = QdrantClient(host="localhost", port=6333)
question = "How do I use the laser cutter?"

# Embed with the same model and query prefix used for ingestion
# (ModernBERT Embed expects "search_query: ").
resp = httpx.post(
    "http://localhost:8080/v1/embeddings",
    json={"model": "nomic-ai/modernbert-embed-base", "input": [f"search_query: {question}"]},
)
dense = resp.json()["data"][0]["embedding"]

# Sparse query vector, scheme wikijs2rag-bm25-crc32-v1 (or: from wikijs2rag.sparse import query_vector).
tokens = {t for t in re.findall(r"[^\W_]+", question.lower()) if len(t) > 1}
indices = sorted(zlib.crc32(t.encode()) for t in tokens)
keywords = models.SparseVector(indices=indices, values=[1.0] * len(indices))

hits = client.query_points(
    collection_name="hs_public",  # the caller's tier decides the collection, never the request
    prefetch=[
        models.Prefetch(query=dense, using="dense", limit=40),
        models.Prefetch(query=keywords, using="sparse", limit=40),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=10,
).points
for hit in hits:
    p = hit.payload
    print(f"{p['title']}  {p['heading_path']}  {p['url']}")
```

### Payload fields on every point

These are consumed by other applications: renaming or removing one is a
breaking change.

| Field | Type | Description |
|---|---|---|
| `id` | str | Point ID, uuid5 of `source:doc_id:chunk_index`; stable across runs |
| `source` | str | `wiki` |
| `doc_id` | str | Wiki.js page ID (`_meta` for wiki statistics) |
| `chunk_index` | int | Position within the document; `0` is the summary point |
| `chunk_kind` | str | `summary`, `chunk`, `attachment`, `image` or `meta` |
| `title` | str | Page title (attachments: page title and file name) |
| `heading_path` | str | Page tree breadcrumb and heading chain, `"equipment > Laser Cutter > Safety"` |
| `text` | str | Chunk text as shown to the model |
| `context` | str | 1-2 sentences placing the chunk in its page (embedded with the text) |
| `url` | str | Link a human can open: page with `#heading` anchor, or file (`#page=N` for PDFs) |
| `tier` | str | `public` or `member` |
| `authority` | int | 5 for policy and training pages, 4 for other wiki pages |
| `created_at`, `updated_at` | str \| null | ISO-8601 page timestamps |
| `indexed_at` | str | When this point was written |
| `stale_after`, `superseded_by` | null | Reserved; always null for the wiki |
| `content_hash` | str | SHA-256 of `text` |
| `sync_run_id` | str | ID of the last run that saw the document |
| `asset_ids`, `location_ids`, `work_ids`, `event_ids`, `training_ids` | list[str] | Registry entities the chunk or page mentions |
| `topics` | list[str] | Shared taxonomy slugs for the page |
| `section` | str \| null | Space section (`1`-`4`, `yard`) when the linked locations agree |
| `page` | int \| null | PDF page number |
| `lang` | str | Page locale |
| `page_kind` | str | Page classification (above) |
| `path`, `locale`, `tags`, `description` | | Wiki.js page path, locale, tags and description |
| `machine_generated` | bool | `true` for vision-model image descriptions |
| `meta_type` | str | For `chunk_kind: meta`: `overview`, `tags`, `recent`, `newest`, `page_listing` |

Payload indexes: `source`, `tier`, `asset_ids`, `location_ids`, `topics`,
`updated_at`, `section`, `doc_id`, `sync_run_id`, `page_kind`, `chunk_kind`.

### Staging database

Tables (created if missing; shared with the other sources' ingesters):
`documents`, `links`, `attachments`, `llm_cache`, `review_queue` (set
`status` to `approved` or `rejected`), `quarantine`, `sync_status` (one row
per source: last run, counts, errors; what a status page reads),
`sync_runs`, and the entity registry `entities`, `entity_aliases` and
`entity_documents`.

### Evaluation

Golden-set recall and red-team prompts belong to the retrieval service. This
ingester supplies the wiki half of the canary tests: set `CANARY_ADMIN` and
`CANARY_MEMBER` to the strings planted in an admin page and a member page.
A run stops if either shows up where it must not, and
`wikijs2rag-report canaries` checks staging and both collections after every
re-index.

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

Changing models needs new collections for every source (the collections
record their model): set new `QDRANT_PUBLIC_COLLECTION` and
`QDRANT_MEMBER_COLLECTION` and run `wikijs2rag --reindex`, which re-embeds
from staging without refetching.

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
[docs/configuration.md](docs/configuration.md)); set at least `WIKI_URL` and
`POSTGRES_PASSWORD`, and for member pages `WIKI_MEMBER_API_KEY` and
`LLM_MODEL`:

```bash
cp .env.example .env   # then edit it
docker compose -f docker-compose.prod.yaml up -d
```

That runs the latest release from `ghcr.io/romkey/wikijs2rag` next to
Postgres and Qdrant; set `APP_VERSION=X.Y.Z` to pin one. The first run is a
full reconcile; after that it runs incrementally every
`POLL_INTERVAL_SECONDS` (default an hour) and fully every
`RECONCILE_INTERVAL_SECONDS` (default a day). Qdrant's REST API and
dashboard are at http://localhost:6333/dashboard.

For a one-shot run, a full reconcile, or a rebuild from staging:

```bash
POLL_INTERVAL_SECONDS=0 docker compose -f docker-compose.prod.yaml run --rm app
POLL_INTERVAL_SECONDS=0 docker compose -f docker-compose.prod.yaml run --rm app wikijs2rag --full
docker compose -f docker-compose.prod.yaml run --rm app wikijs2rag --reindex
```

### The member API key

Public pages need no key: the anonymous crawl sends no credentials. For
member pages:

1. In Wiki.js, make sure there is a group whose page rules give exactly what
   a member may read (the "Members" group), and that admin pages are not
   readable by it.
2. As an administrator, go to **Administration → API Access** and create a
   key whose permission scope is **that group**. Never choose **Full
   Access**: the pipeline checks, and stops if the key can list API keys.
3. Put it in `.env` as `WIKI_MEMBER_API_KEY`, and set `LLM_MODEL`.
4. Plant canaries: a unique string in an admin-only page (`CANARY_ADMIN`)
   and one in a member-only page (`CANARY_MEMBER`). Then run
   `wikijs2rag-report canaries` after the first run.

Upgrading from 2.x: the old `wiki` collection is not reused. Delete it when
the new collections are in use; nothing needs migrating.

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
Without `DATABASE_URL` it stages into `wikijs2rag.db` (SQLite) in the
current directory.

## Development

With Docker, nothing else to install:

```bash
docker compose up --build                                  # ingest from the working tree, with Postgres and Qdrant
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
and the model servers over HTTP, use Qdrant's in-memory mode and SQLite for
staging, so no services are needed.

Releases: run the **Release** workflow from the Actions tab, then merge the
PR it opens. See [CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE)
