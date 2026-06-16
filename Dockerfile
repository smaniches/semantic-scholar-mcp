# Multi-stage Dockerfile for semantic-scholar-mcp
# Produces a minimal production image for running the MCP server

# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /build

# Install build dependencies
RUN pip install --no-cache-dir build

# Copy only what's needed for building
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

# Build wheel
RUN python -m build --wheel --outdir /build/dist

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

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

# Unbuffered output so structured JSON logs reach stderr promptly under the
# stdio transport; skip .pyc writes the non-root user can't persist anyway.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Switch to non-root user
USER mcp

# The MCP server speaks stdio by default; pass `--transport http` (or set
# MCP_TRANSPORT=http) to serve MCP Streamable HTTP instead. MCP_HOST defaults
# to 0.0.0.0 here — unlike the CLI's loopback default — because a container's
# port is only reachable if explicitly published (-p) or mapped by a platform.
#
# SEMANTIC_SCHOLAR_API_KEY is intentionally NOT declared as a build-time ENV:
# baking a secret-named variable into an image layer trips secret scanners, and
# an empty default is behaviorally identical to leaving it unset (the client
# resolves the key with `os.environ.get(..., "")` and only sends an x-api-key
# header when the value is truthy). Pass it at runtime instead, e.g.
# `docker run -e SEMANTIC_SCHOLAR_API_KEY=... ...`; without it the server runs
# in rate-limited keyless mode.
ENV MCP_HOST="0.0.0.0"

# Informational: the HTTP transport's default port (override with --port/PORT)
EXPOSE 8000

ENTRYPOINT ["semantic-scholar-mcp"]
