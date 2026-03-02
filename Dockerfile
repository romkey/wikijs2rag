# syntax=docker/dockerfile:1

FROM python:3.12-slim AS deps

WORKDIR /build
COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /usr/local/lib/python3.12/site-packages/pip \
           /usr/local/lib/python3.12/site-packages/pip-*.dist-info \
           /usr/local/lib/python3.12/site-packages/setuptools \
           /usr/local/lib/python3.12/site-packages/setuptools-*.dist-info \
           /usr/local/lib/python3.12/site-packages/pkg_resources \
    && find /usr/local/lib/python3.12/site-packages \
       -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true


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
RUN chown -R appuser:appuser /app

ARG VERSION=unknown
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.source="https://github.com/romkey/wikijs2rag"

USER appuser

CMD ["python", "main.py"]
