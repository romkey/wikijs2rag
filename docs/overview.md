# wikijs2rag overview

**What:** ingests every public Wiki.js page into a Qdrant collection, with
Ollama embeddings, for any RAG application to search.
**Runs as:** worker (`wikijs2rag`), polling every `POLL_INTERVAL_SECONDS`;
plus a CLI search tool (`wikijs2rag-query`).
**Talks to:** Wiki.js (GraphQL, HTML fallback), Ollama (`/api/embed`),
Qdrant (REST).

## Files

- `src/wikijs2rag/__main__.py`: `wikijs2rag` command: logging, SIGTERM, poll loop.
- `src/wikijs2rag/config.py`: `Settings` / `QuerySettings` from environment variables.
- `src/wikijs2rag/ingest.py`: one ingestion run; payload and wiki-metadata builders.
- `src/wikijs2rag/wiki_client.py`: Wiki.js GraphQL client with retries and HTML scraping fallback.
- `src/wikijs2rag/chunker.py`: header-aware chunking; tables, lists, code kept whole.
- `src/wikijs2rag/embedder.py`: Ollama embedding client with truncation.
- `src/wikijs2rag/store.py`: Qdrant collection, payload indexes, replace-per-page upserts.
- `src/wikijs2rag/query.py`: `wikijs2rag-query` command.
- `tests/`: pytest; one file per module. HTTP faked with respx, Qdrant in-memory.
- `Dockerfile`: `dev` and `production` stages.
- `docker-compose.yaml` (dev, with Qdrant), `.test`, `.lint`, `.prod`.

## Data flow

list public pages -> skip unchanged (`updatedAt` vs stored) -> fetch page ->
chunk -> embed -> delete the page's old points, upsert new ones -> after all
pages, rebuild metadata chunks (`is_meta="true"`, `page_id=0`).

## Conventions

- All configuration from environment variables; see docs/configuration.md.
- Qdrant payload fields are a public interface (README lists them); changing
  them is a breaking change.
- Rules: https://github.com/romkey/python-ai-rules/blob/main/RULES.md
