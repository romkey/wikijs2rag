# Instructions for AI assistants

Follow https://github.com/romkey/python-ai-rules/blob/main/RULES.md.
Follows python-ai-rules v0.0.0 (main at 790642a, 2026-10-12; no release was
tagged yet). When re-scanning, apply the entries in its UPGRADING.md newer
than that, then update this line.
With Claude Code, the `python-rules` plugin loads them for you.

Read docs/overview.md first. `main` is protected: work on a branch and
open a pull request; CI and Lint must pass before it can merge.

<!-- Project-specific notes go below. -->

- The Qdrant payload fields listed in README.md are consumed by other
  applications; renaming or removing one is a breaking change.
- Tests never touch a real Wiki.js, embeddings API or Qdrant: HTTP is faked with
  respx and Qdrant runs in-memory (`tests/conftest.py`).
