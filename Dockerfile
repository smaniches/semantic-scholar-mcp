# Multi-stage Dockerfile for semantic-scholar-mcp
# Produces a minimal production image for running the MCP server

# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN pip install --no-cache-dir build

# Copy only what's needed for building
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

# Build wheel
RUN python -m build --wheel --outdir /build/dist

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="Santiago Maniches <santiago@topologica.ai>"
LABEL org.opencontainers.image.source="https://github.com/smaniches/semantic-scholar-mcp"
LABEL org.opencontainers.image.description="MCP server for Semantic Scholar - 200M+ academic papers"
LABEL org.opencontainers.image.licenses="MIT"

# Create non-root user
RUN groupadd --gid 1000 mcp && \
    useradd --uid 1000 --gid mcp --shell /bin/bash --create-home mcp

WORKDIR /app

# Install the built wheel
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && \
    rm -rf /tmp/*.whl

# Switch to non-root user
USER mcp

# The MCP server communicates via stdio
# API key should be passed as environment variable
ENV SEMANTIC_SCHOLAR_API_KEY=""

ENTRYPOINT ["semantic-scholar-mcp"]
