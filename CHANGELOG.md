# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
