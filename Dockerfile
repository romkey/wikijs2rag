# syntax=docker/dockerfile:1

# ── Stage 1: install deps + download model ────────────────────────────────────
FROM python:3.12-slim AS deps

WORKDIR /build

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model (needs all deps intact including Pillow).
ARG DEFAULT_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RUN HF_MODEL="${DEFAULT_EMBEDDING_MODEL}" python -c \
    "import os; \
     from fastembed import TextEmbedding; \
     m = os.environ['HF_MODEL']; \
     print('Downloading model:', m, flush=True); \
     list(TextEmbedding(model_name=m, cache_dir='/app/hf-cache').embed(['warmup'])); \
     print('Model ready.', flush=True)"

# ── Strip packages not needed at runtime ──────────────────────────────────────
# This runs AFTER the model download so fastembed can still use Pillow etc.
# during the build step above.
RUN rm -rf \
    # sympy + mpmath: 79 MB — onnxruntime only needs these for symbolic
    # optimization of custom operators, never triggered for text embeddings.
    /usr/local/lib/python3.12/site-packages/sympy \
    /usr/local/lib/python3.12/site-packages/sympy-*.dist-info \
    /usr/local/lib/python3.12/site-packages/mpmath \
    /usr/local/lib/python3.12/site-packages/mpmath-*.dist-info \
    # Pillow: fastembed imports it unconditionally at module load — must keep.
    # Pygments: 9 MB — optional rich/httpx dep, not used.
    /usr/local/lib/python3.12/site-packages/pygments \
    /usr/local/lib/python3.12/site-packages/Pygments* \
    # Rich + typer: 4 MB — fastembed CLI, not used.
    /usr/local/lib/python3.12/site-packages/rich \
    /usr/local/lib/python3.12/site-packages/rich-*.dist-info \
    /usr/local/lib/python3.12/site-packages/typer \
    /usr/local/lib/python3.12/site-packages/typer-*.dist-info \
    # grpc: qdrant-client imports it at module load — must keep.
    # pip/setuptools: 12 MB — not needed at runtime.
    /usr/local/lib/python3.12/site-packages/pip \
    /usr/local/lib/python3.12/site-packages/pip-*.dist-info \
    /usr/local/lib/python3.12/site-packages/setuptools \
    /usr/local/lib/python3.12/site-packages/setuptools-*.dist-info \
    /usr/local/lib/python3.12/site-packages/pkg_resources \
    && find /usr/local/lib/python3.12/site-packages \
       -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true


# ── Stage 2: final image ───────────────────────────────────────────────────────
FROM python:3.12-slim

RUN useradd -m -u 1000 appuser && \
    rm -rf /usr/local/lib/python3.12/site-packages/pip \
           /usr/local/lib/python3.12/site-packages/setuptools \
    && find /usr/local/lib/python3.12 -type d -name __pycache__ \
       -exec rm -rf {} + 2>/dev/null; true

COPY --from=deps /usr/local/lib/python3.12/site-packages \
                  /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

WORKDIR /app
COPY src/ .
COPY VERSION .

COPY --from=deps /app/hf-cache /app/hf-cache
RUN chown -R appuser:appuser /app

ENV HF_HOME=/app/hf-cache

ARG VERSION=unknown
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.source="https://github.com/romkey/wikijs2rag"

USER appuser

CMD ["python", "main.py"]
