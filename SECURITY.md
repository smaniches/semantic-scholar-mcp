# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.3.x   | :white_check_mark: |
| < 1.3   | :x:                |

## Reporting a Vulnerability

**DO NOT** open public issues for security vulnerabilities.

Email: santiago@topologica.ai

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

Expected response time: 48 hours.

## Security Best Practices for Users

- **Never** commit API keys to version control.
- Prefer the `SEMANTIC_SCHOLAR_API_KEY` environment variable over the
  per-request `api_key` tool parameter (see API Key Handling below).
- Keep this package updated to the latest version.
- Review [CHANGELOG.md](CHANGELOG.md) for security-related patches.

## API Key Handling

The server accepts the Semantic Scholar API key in two places:

1. **`SEMANTIC_SCHOLAR_API_KEY` environment variable** (recommended). The key
   stays in the local process environment and is never serialized into tool
   arguments or transcripts.
2. **Per-request `api_key` tool parameter** (deprecated since v1.3.x). Because
   tool-call arguments are typically captured by MCP clients and may be
   surfaced in LLM tool-call history, **using `api_key` per-request can
   expose the key in client logs and transcripts**. A runtime
   `DeprecationWarning` is now emitted when this parameter is used.
   Removal is planned for v2.0.0.

In both cases the key is sent only to `api.semanticscholar.org` over HTTPS,
as the `x-api-key` header, and is never written to disk by this server.

## Security Features

- API keys are never persisted to disk by the server.
- Input validation on all tool parameters; paper-ID format whitelist rejects
  null bytes, `?`, `#`, and `../`.
- Semantic Scholar API rate limits respected automatically; exponential
  backoff with jitter on `429` and `503`.
- HTTPS-only API communication.
- Minimal direct dependency footprint (`mcp`, `httpx`, `pydantic`).

## Known Limitations

See [LIMITATIONS.md](LIMITATIONS.md) for the full list of known gaps tracked
for follow-up releases.
