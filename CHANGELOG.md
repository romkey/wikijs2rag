# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [3.0.1] - 2026-11-19

### Added

- `REINGEST_ON_START=true` re-fetches and re-indexes every page on the first
  run after the service starts, then goes back to incremental runs: a
  one-off rebuild, where `FORCE_REINGEST` rebuilds on every run.

### Fixed

- A `DATABASE_URL` starting with `postgres://` or `postgresql://` failed every
  run with `Can't load plugin: sqlalchemy.dialects:postgres`. Both now use the
  bundled psycopg driver, and any other scheme is refused at startup.
- Pages whose table of contents Wiki.js stores as a list failed to fetch
  (`String cannot represent value` on `pages.single.toc`). The ingester no
  longer asks for the table of contents; section anchors come from the
  rendered HTML as before.

### Documentation

- Creating the member API key: how to get past Wiki.js 2.5's Full Access
  checkbox, which can't be unchecked by clicking it, and a GraphQL fallback.

## [3.0.0] - 2026-11-18

The wiki ingester becomes the wiki source of the shared, tiered RAG
pipeline. Existing indexed data is not migrated: the new collections are
built from scratch.

### Added

- Two crawls: anonymous (no credentials) for `tier: public`, and
  `WIKI_MEMBER_API_KEY` (a members-group key) for `tier: member`. Public
  points go to `hs_public` and `hs_member`; member-only points to
  `hs_member` only (`QDRANT_PUBLIC_COLLECTION`, `QDRANT_MEMBER_COLLECTION`).
  Tier is recomputed every run. A run stops if a key has admin rights.
  If guests can't use the GraphQL listing, pages are checked as a guest.
- Postgres staging store (`DATABASE_URL`; a `postgres` service in compose):
  normalized pages, links, attachments, LLM cache, review queue,
  quarantine, sync status and the entity registry. `wikijs2rag --reindex`
  rebuilds the collections from it without refetching.
- Every locale; outbound links (internal and external); linked PDFs and text
  files indexed as children of their page; images staged with alt text and
  caption, described by `VISION_MODEL` when set.
- Sanitization: a secret scanner that redacts and reports to the quarantine
  table, network identifiers in code blocks replaced with placeholders, and
  an LLM exclusion classifier for member content with a review queue.
- LLM enrichment (`LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`): page kind,
  page summary point, per-chunk context line, topics from the shared
  taxonomy (`TOPICS_FILE`), all cached by content hash. Deterministic
  fallbacks without an LLM.
- Linker: registry aliases tag chunks with `asset_ids`, `location_ids`,
  `training_ids`, …; equipment, area, training and build pages are recorded
  in `entity_documents`.
- Hybrid vectors: named `dense` and `sparse` (keyword, IDF) vectors per
  point; `wikijs2rag-query` fuses them with reciprocal rank fusion and takes
  `--tier`.
- Full reconcile every `RECONCILE_INTERVAL_SECONDS` (or `--full`) with
  run-ID stamping, deletion of what's gone, and a safety valve; a per-source
  lock and status row.
- Canaries (`CANARY_ADMIN`, `CANARY_MEMBER`) stop a run that sees one where
  it must not.
- `wikijs2rag-report`: `status`, `stale`, `flagged`, `links`, `canaries`.

- `EMBEDDING_QUERY_PREFIX` and `EMBEDDING_DOCUMENT_PREFIX`: task prefixes put
  before queries and documents before embedding. Unset, they default to the
  model's own prefixes for known models (ModernBERT Embed, nomic-embed-text,
  e5, mxbai-embed-large, Qwen3-Embedding).
- The compose files run Hugging Face Text Embeddings Inference as the `tei`
  service (profile `tei`, on in `.env.example`).

### Changed

- **Breaking:** new payload fields (see README): `source`, `doc_id`,
  `title`, `heading_path`, `url`, `tier`, `authority`, `chunk_kind`,
  `page_kind`, entity ID lists, `topics`, … replace `page_id`,
  `page_title`, `page_path`, `page_url`, `section_breadcrumb`,
  `prev_chunk_id`, `next_chunk_id`, `parent_chunk_index`, `total_chunks`
  and `is_meta`. `section` now means the space section, not a heading.
  Point IDs are stable (uuid5 of source, doc and chunk index).
- **Breaking:** chunking follows headings first, then size
  (`CHUNK_TARGET_TOKENS`, `CHUNK_MAX_TOKENS`, `CHUNK_OVERLAP_TOKENS`); URLs
  open at the heading's anchor.
- **Breaking:** wiki metadata no longer lists contributors or author names.
- Collections record their embedding model; a run with a different one
  stops instead of mixing vectors.

### Removed

- **Breaking:** `WIKI_API_KEY`, `QDRANT_COLLECTION`, `CHUNK_SIZE`,
  `CHUNK_OVERLAP` and `PARENT_CHUNK_RATIO`; setting one is a configuration
  error that names its replacement.

### Changed (embeddings)

- **Breaking:** the default embedding model is now
  `nomic-ai/modernbert-embed-base` served by TEI (`OPENAI_BASE_URL` defaults
  to `http://tei/v1` under compose, `http://localhost:8080/v1` natively), and
  `EMBEDDING_CONTEXT_LENGTH` defaults to 8192. To keep using OpenAI, set
  `OPENAI_BASE_URL=https://api.openai.com/v1` and
  `EMBEDDING_MODEL=text-embedding-3-small`. A new model needs a new
  `QDRANT_COLLECTION` and `FORCE_REINGEST=true`.

## [2.0.2] - 2026-11-12

### Fixed

- The release workflow failed to push the image when it ran for a merged
  release PR, because GitHub gives those runs a read-only build cache.
  Writing the cache is now best effort.

## [2.0.1] - 2026-11-12

### Changed

- Follows [python-ai-rules](https://github.com/romkey/python-ai-rules): the
  code is now the `wikijs2rag` package, installable with pip or uv, with
  `wikijs2rag` (ingestion) and `wikijs2rag-query` (search) commands. Both
  support `--version`.
- **Breaking:** the image runs `wikijs2rag` and `wikijs2rag-query` instead of
  `python main.py` and `python query.py`. Override commands accordingly.
- **Breaking:** `docker-compose.yml` is now `docker-compose.prod.yaml` for
  running the released image (`docker compose -f docker-compose.prod.yaml up -d`,
  service `app` instead of `wiki2rag`); `docker-compose.yaml` is for
  development. Qdrant's `qdrant_data` volume is unchanged, so run from the
  same directory to keep existing vectors.
- **Breaking:** images are published only by releases (`X.Y.Z`, `X.Y`, `X`,
  `latest`), not on every push to `main`; `:main` and `:sha-*` tags are gone.
- Python 3.15, Qdrant 1.20, and current versions of every dependency.
- Configuration is validated at startup; a bad value exits with code 2 and
  names the variable. `FORCE_REINGEST` also accepts `on`/`off`.
- SIGTERM (`docker stop`) finishes the current page and exits cleanly
  instead of being killed.
- The query tool reads `QDRANT_HOST`, `QUERY_*` and `NO_COLOR` as before;
  errors from the embeddings API now exit with code 1 instead of a traceback.

### Fixed

- A table or ordered list at the very end of a section lost its last row,
  which became a separate prose chunk.
- The Qdrant health check in compose used `curl`, which the Qdrant image
  doesn't have, so the ingester never started under `docker compose up`.
- The retry log message reported the base delay, not the actual one.

### Added

- Test suite (99% coverage), strict mypy, ruff, pre-commit hooks, secret
  scanning, CI, lint and release workflows, Renovate, MIT license.

## [2.0.0] - 2026-11-11

Published as the `2.0.0` image from `main`; never tagged.

### Changed

- **Breaking:** embeddings come from an OpenAI-compatible `/v1/embeddings`
  API instead of Ollama's `/api/embed`. `OPENAI_BASE_URL` (default OpenAI)
  and `OPENAI_API_KEY` replace `OLLAMA_URL` and `OLLAMA_API_KEY`; the default
  model is now `text-embedding-3-small` and the default context length 8191.
  Ollama still works through its `/v1` endpoint. New optional
  `EMBEDDING_DIMENSIONS` for `text-embedding-3-*` models. The new default
  model has a different vector size, so use a new `QDRANT_COLLECTION` (or
  `FORCE_REINGEST=true` on a new collection) when switching.
