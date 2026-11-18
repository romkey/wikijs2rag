# wikijs2rag overview

**What:** the wiki source of a tiered RAG pipeline: syncs Wiki.js pages into
`hs_public` (what guests can read) and `hs_member` (what members can read),
via a Postgres staging store, with sanitization, LLM enrichment and entity
linking.
**Runs as:** worker (`wikijs2rag`), incremental every `POLL_INTERVAL_SECONDS`,
full reconcile every `RECONCILE_INTERVAL_SECONDS`; plus `wikijs2rag-query`
(hybrid search) and `wikijs2rag-report` (status, reports, canary check).
**Talks to:** Wiki.js (GraphQL as a guest and with a members-group key; HTML
for guest checks and files), an OpenAI-compatible embeddings API and chat
API, Postgres, Qdrant (REST).

## Files

- `src/wikijs2rag/__main__.py`: `wikijs2rag` command: logging, SIGTERM, poll loop, `--full`, `--reindex`.
- `src/wikijs2rag/config.py`: `Settings` / `QuerySettings` from environment variables.
- `src/wikijs2rag/ingest.py`: one run: crawls, tiering, staging, indexing, metadata, reconcile.
- `src/wikijs2rag/wiki_client.py`: Wiki.js client, one per crawl.
- `src/wikijs2rag/staging.py`: SQLAlchemy tables and access: staging, cache, review, quarantine, status, registry.
- `src/wikijs2rag/content.py`: markdown, anchors, links, attachments, images from a page.
- `src/wikijs2rag/sanitize.py`: secret scanner, network-identifier placeholders.
- `src/wikijs2rag/enrich.py`: page kind, summary, topics, context, exclusion classifier (LLM or fallback).
- `src/wikijs2rag/linker.py`: registry alias matching.
- `src/wikijs2rag/chunker.py`: heading-first chunking.
- `src/wikijs2rag/attachments.py`: PDFs, text files, image descriptions.
- `src/wikijs2rag/llm.py`, `embedder.py`: model clients.
- `src/wikijs2rag/sparse.py`, `store.py`: sparse vectors; Qdrant collections.
- `src/wikijs2rag/query.py`, `report.py`: the two tools.
- `tests/`: pytest. HTTP faked with respx, Qdrant in-memory, staging in SQLite. `test_ingest.py` is end to end.
- `Dockerfile`: `dev` and `production` stages.
- `docker-compose.yaml` (dev, with Postgres and Qdrant), `.test`, `.lint`, `.prod`.

## Data flow

list pages as guest and as member -> tier per page -> skip unchanged
(`updatedAt`, tier) -> fetch with the tier's credentials -> canary check ->
markdown, links, attachments -> sanitize -> stage -> (member: exclusion
classifier, review queue) -> kind, summary, topics -> chunk -> context, link
-> embed dense + sparse -> replace the doc's points in hs_member (and
hs_public if public) -> metadata points -> full runs: stamp, delete stale.

## Conventions

- All configuration from environment variables; see docs/configuration.md.
- Qdrant payload fields are a public interface (README lists them); changing
  them is a breaking change. So are the staging tables other sources share.
- Tier is enforced by credentials and collections, never by filtering
  alone. Never add a code path that uses an admin credential.
- Rules: https://github.com/romkey/python-ai-rules/blob/main/RULES.md
