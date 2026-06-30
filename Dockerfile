# Single-stage image for the Semantic Scholar MCP server.
#
# This Dockerfile is consumed directly by Glama's builder, so it must stay
# SINGLE-STAGE on python:3.12-slim. Glama does not pick up a multi-stage build
# and falls back to an auto-generated debian:bookworm-slim base that fails to
# resolve from Docker Hub ("context deadline exceeded"). The package is
# pure-Python, so a single stage needs no separate build stage. This matches the
# build pattern proven on the sibling MCP servers (alphafold-sovereign-mcp,
# uniprot-mcp).
FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/smaniches/semantic-scholar-mcp"
LABEL org.opencontainers.image.description="MCP server for Semantic Scholar - 200M+ academic papers"
LABEL org.opencontainers.image.licenses="MIT"

# Unbuffered output so structured logs reach stderr promptly under the stdio
# transport; skip .pyc writes the non-root user can't persist anyway.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create the non-root user before copying source, so this static system-level
# layer stays cached when only the application code changes.
RUN groupadd --gid 1000 mcp && \
    useradd --uid 1000 --gid mcp --shell /bin/bash --create-home mcp

WORKDIR /app

# Install the package and its runtime deps from source (pure-Python wheel build,
# no compilers needed). The .dockerignore keeps the build context lean so the
# remote-context upload to Glama's builder stays small.
COPY . .
RUN pip install --no-cache-dir .

USER mcp

# The MCP server speaks stdio by default (what Glama's build test exercises);
# pass `--transport http` (or set MCP_TRANSPORT=http) to serve Streamable HTTP
# instead. MCP_HOST defaults to 0.0.0.0 here so a published container port is
# reachable. SEMANTIC_SCHOLAR_API_KEY is intentionally NOT baked in as a
# build-time ENV (it would trip secret scanners and an empty default is
# behaviorally identical to unset) — pass it at runtime, e.g.
# `docker run -e SEMANTIC_SCHOLAR_API_KEY=... ...`.
ENV MCP_HOST="0.0.0.0"

# Informational: the HTTP transport's default port (override with --port/PORT).
EXPOSE 8000

ENTRYPOINT ["semantic-scholar-mcp"]
