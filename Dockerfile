# syntax=docker/dockerfile:1
# The directive above enables BuildKit cache mounts (--mount=type=cache),
# which persist pip's package cache between builds.

# ── Stage 1: dependency installation ──────────────────────────────────────────
FROM python:3.12-slim AS deps

WORKDIR /install

COPY requirements.txt .

# --mount=type=cache keeps the pip HTTP cache on the build host between
# builds.  If requirements.txt is unchanged the packages are read from
# disk instead of being re-downloaded.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Pre-download the embedding model directly into /app/hf-cache so it is
# baked into the image layer and available without internet access at runtime.
# Override at build time:
#   docker build --build-arg DEFAULT_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5 …
ARG DEFAULT_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RUN HF_MODEL="${DEFAULT_EMBEDDING_MODEL}" python -c \
    "import os; \
     from fastembed import TextEmbedding; \
     m = os.environ['HF_MODEL']; \
     print('Downloading model:', m, flush=True); \
     list(TextEmbedding(model_name=m, cache_dir='/app/hf-cache').embed(['warmup'])); \
     print('Model ready.', flush=True)"


# ── Stage 2: final image ───────────────────────────────────────────────────────
FROM python:3.12-slim

RUN useradd -m -u 1000 appuser

# Copy installed packages from stage 1
COPY --from=deps /usr/local/lib/python3.12/site-packages \
                  /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

WORKDIR /app
COPY src/ .
COPY VERSION .

# Copy the pre-downloaded model and hand all of /app to appuser
COPY --from=deps /app/hf-cache /app/hf-cache
RUN chown -R appuser:appuser /app

# HF_HOME tells fastembed where to find models at runtime.
# The named volume hf_cache in docker-compose mounts here so that models
# persist across --rm runs without being re-downloaded.
ENV HF_HOME=/app/hf-cache

# Stamp the image with the version from the VERSION file.
ARG VERSION=unknown
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.source="https://github.com/romkey/wikijs2rag"

USER appuser

CMD ["python", "main.py"]
