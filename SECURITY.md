# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 1.5.x   | Yes       |
| < 1.5   | No        |

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

The server accepts the Semantic Scholar API key in three places:

1. **`SEMANTIC_SCHOLAR_API_KEY` environment variable** (recommended). The key
   stays in the local process environment and is never serialized into tool
   arguments or transcripts.
2. **Per-request `api_key` tool parameter** (deprecated since v1.3.x). Because
   tool-call arguments are typically captured by MCP clients and may be
   surfaced in LLM tool-call history, **using `api_key` per-request can
   expose the key in client logs and transcripts**. A runtime
   `DeprecationWarning` is now emitted when this parameter is used.
   Removal is planned for v2.0.0.
3. **Per-request transport credentials** (Streamable HTTP only, since
   v1.5.0). When served with `--transport http`, each request may carry a key
   in the `x-api-key` header (preferred) or a
   `SEMANTIC_SCHOLAR_API_KEY`/`api_key` query parameter. The key is bound to
   a request-scoped context variable, is never written to the server's own
   logs, and cannot leak across concurrent requests. Prefer the header: a key
   in a query string (unlike the header) commonly ends up in proxy and access
   logs.

In all cases this server forwards the key only to `api.semanticscholar.org`
over HTTPS, as the `x-api-key` header, and never writes it to disk. Be aware
that when you connect to a **remotely hosted** instance over Streamable HTTP,
your key necessarily transits that endpoint's operator before it is forwarded
to Semantic Scholar — send keys only to remote endpoints you trust, and only
over HTTPS.

## Security Features

- API keys are never persisted to disk by the server.
- Input validation on all tool parameters; paper-ID format whitelist rejects
  null bytes, `?`, `#`, and `../`.
- Semantic Scholar API rate limits respected automatically; exponential
  backoff with jitter on `429`, `502`, and `503`.
- HTTPS-only API communication.
- Minimal direct dependency footprint (`mcp`, `httpx`, `pydantic`).
- Supply-chain hardening: GitHub Actions pinned to commit SHAs; Sigstore-backed
  SLSA build-provenance attestations on the wheel, sdist, and container image;
  PEP 740 attestations on PyPI uploads; and a CycloneDX SBOM published with each
  release.

## Known Limitations

See [LIMITATIONS.md](LIMITATIONS.md) for the full list of known gaps tracked
for follow-up releases.
