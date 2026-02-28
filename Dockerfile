# syntax=docker/dockerfile:1
# The directive above enables BuildKit cache mounts (--mount=type=cache),
# which persist pip's package cache and the model download across builds
# so that only changed layers are re-fetched.

# ── Stage 1: dependency installation ──────────────────────────────────────────
FROM python:3.12-slim AS deps

WORKDIR /install

COPY requirements.txt .

# --mount=type=cache keeps the pip HTTP cache on the build host between
# builds.  If requirements.txt is unchanged the packages are read from
# disk instead of being re-downloaded.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Pre-download the embedding model into /app/hf-cache.
# --mount=type=cache keeps the raw downloaded files across builds so that
# the model is only fetched from the internet once, even when the layer is
# invalidated (e.g. after a requirements.txt change).
# Override at build time:  docker build --build-arg DEFAULT_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5 …
ARG DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2
RUN --mount=type=cache,target=/tmp/model-dl-cache \
    HF_MODEL="${DEFAULT_EMBEDDING_MODEL}" python -c \
    "import os, shutil; \
     from fastembed import TextEmbedding; \
     m = os.environ['HF_MODEL']; \
     print('Downloading model:', m, flush=True); \
     model = TextEmbedding(model_name=m, cache_dir='/tmp/model-dl-cache'); \
     list(model.embed(['warmup'])); \
     shutil.copytree('/tmp/model-dl-cache', '/app/hf-cache', dirs_exist_ok=True); \
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

# Copy the pre-downloaded model and hand all of /app to appuser
COPY --from=deps /app/hf-cache /app/hf-cache
RUN chown -R appuser:appuser /app

# HF_HOME tells fastembed (and HuggingFace libs) where to find models.
# The named volume hf_cache in docker-compose mounts here so that models
# persist across --rm runs without being re-downloaded.
ENV HF_HOME=/app/hf-cache

USER appuser

CMD ["python", "main.py"]
