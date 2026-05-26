# Reviewer Guide

Quick-start for auditing or evaluating this repository.

## Install

```bash
pip install -e ".[dev]"
```

Or run without cloning:

```bash
uvx s2-mcp-server
```

## Run the test suite

```bash
pytest --cov=src/semantic_scholar_mcp --cov-report=term-missing --tb=short -q
```

CI enforces `--cov-fail-under=80`. Tests that require a live
`SEMANTIC_SCHOLAR_API_KEY` are skipped by default (14 tests in
`test_api_compatibility.py`).

## Linting

```bash
ruff check src/ tests/
ruff format --check src/ tests/
```

## Type checking

```bash
mypy src/
```

Configured with `strict = true` in `pyproject.toml`.

## Inspect the tool surface

```bash
python -c "
from semantic_scholar_mcp.server import mcp
tools = mcp._tool_manager._tools
for name in sorted(tools):
    print(name)
"
```

Expected: 14 tools prefixed with `semantic_scholar_`.

## Version consistency

```bash
pytest tests/test_version_consistency.py -v
```

This test fails fast if version strings diverge across:

| File | Field |
| --- | --- |
| `pyproject.toml` | `project.version` |
| `server.json` | `$.version`, `$.packages[0].version` |
| `CITATION.cff` | `version` |
| `.zenodo.json` | `version` |
| Runtime | `importlib.metadata.version("s2-mcp-server")` |

## Documentation-to-code sync

```bash
pytest tests/test_docs_tool_surface.py -v
```

Fails if a registered `@mcp.tool` is missing from the README, or if the
README advertises a tool the server does not register.

## Security-relevant files

| File | What to look at |
| --- | --- |
| `SECURITY.md` | Vulnerability reporting, API key handling, security features |
| `LIMITATIONS.md` | Known gaps (URL encoding, per-request key, Actions pinning) |
| `src/semantic_scholar_mcp/validators.py` | Paper-ID input validation, injection guards |
| `src/semantic_scholar_mcp/client.py` | HTTP client, rate limiter, retry loop, key handling |
| `src/semantic_scholar_mcp/errors.py` | Typed exception hierarchy |

## CI workflows

| Workflow | Trigger | What it checks |
| --- | --- | --- |
| `ci.yml` | push/PR to main | lint, format, mypy, tests (3.10-3.13, Linux/macOS/Windows), CodeQL, dependency review |
| `test-api-compat.yml` | push/PR/weekly | live API compatibility (requires `SEMANTIC_SCHOLAR_API_KEY` secret) |
| `publish.yml` | release published | build + trusted PyPI publish |
| `docker.yml` | release/push to main | build + push to ghcr.io |
| `release-please.yml` | push to main | automated release PR |
