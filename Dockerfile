# ── Stage 1: dependency installation ──────────────────────────────────────────
FROM python:3.12-slim AS deps

WORKDIR /install

# torch CPU-only wheel is much smaller than the default GPU build
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the default sentence-transformers model so the container
# works without internet access at runtime.
# Override by setting DEFAULT_EMBEDDING_MODEL at build time:
#   docker build --build-arg DEFAULT_EMBEDDING_MODEL=all-mpnet-base-v2 …
ARG DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2
# Pass the model name via env var so Python can read it unambiguously from
# os.environ; avoids shell quoting issues with the -c one-liner.
RUN HF_MODEL="${DEFAULT_EMBEDDING_MODEL}" python -c \
    "import os, sys; \
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

# Copy pre-downloaded HuggingFace / sentence-transformers model cache
COPY --from=deps /root/.cache /root/.cache

WORKDIR /app
COPY src/ .

# Run as a non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser /app
# The model cache lives in /root/.cache – make it readable
RUN chmod -R a+rX /root/.cache
USER appuser

CMD ["python", "main.py"]
