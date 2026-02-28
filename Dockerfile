# ── Stage 1: dependency installation ──────────────────────────────────────────
FROM python:3.12-slim AS deps

WORKDIR /install

# torch CPU-only wheel — unpinned so the latest compatible version is used.
# --index-url (not --extra-index-url) ensures pip fetches only from the
# PyTorch CPU index for this step, which is where +cpu wheels live.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the default sentence-transformers model into /app/hf-cache.
# HF_HOME is set here and carried into the final image so the runtime
# process always finds the model at this fixed path regardless of which
# user runs the container.
# Override at build time:  docker build --build-arg DEFAULT_EMBEDDING_MODEL=all-mpnet-base-v2 …
ARG DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2
RUN HF_HOME=/app/hf-cache HF_MODEL="${DEFAULT_EMBEDDING_MODEL}" python -c \
    "import os; \
     from sentence_transformers import SentenceTransformer; \
     m = os.environ['HF_MODEL']; \
     print('Downloading model:', m, flush=True); \
     SentenceTransformer(m); \
     print('Model ready.', flush=True)"


# ── Stage 2: final image ───────────────────────────────────────────────────────
FROM python:3.12-slim

# Copy installed packages from stage 1
COPY --from=deps /usr/local/lib/python3.12/site-packages \
                  /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Create app user before chowning files
RUN useradd -m -u 1000 appuser

WORKDIR /app
COPY src/ .

# Copy the pre-downloaded model cache and hand ownership to appuser.
# HF_HOME tells HuggingFace/sentence-transformers to look here at runtime.
# Docker initialises a named volume from image content on first use, so
# mounting hf_cache:/ app/hf-cache persists the model across --rm runs
# without re-downloading.
COPY --from=deps /app/hf-cache /app/hf-cache
RUN chown -R appuser:appuser /app

ENV HF_HOME=/app/hf-cache

USER appuser

CMD ["python", "main.py"]
