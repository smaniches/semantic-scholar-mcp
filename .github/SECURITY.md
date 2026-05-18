# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.2.x   | :white_check_mark: |
| < 1.2   | :x:                |

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
- Review [CHANGELOG.md](../CHANGELOG.md) for security-related patches.

## API Key Handling

The server accepts the Semantic Scholar API key in two places:

1. **`SEMANTIC_SCHOLAR_API_KEY` environment variable** (recommended). The key
   stays in the local process environment and is never serialized into tool
   arguments or transcripts.
2. **Per-request `api_key` tool parameter** (advertised, see README). Because
   tool-call arguments are typically captured by MCP clients and may be
   surfaced in LLM tool-call history, **using `api_key` per-request can
   expose the key in client logs and transcripts**. Use the environment
   variable in any non-trivial deployment. Removal of the per-request
   parameter is planned for a follow-up release (see [Known Limitations](#known-limitations)).

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

The following are known gaps tracked for a follow-up release. They are not
defects in the current code — they are explicit boundaries on what this
project's security posture provides today.

- **Paper IDs are interpolated into request URL paths without
  `urllib.parse.quote`.** The validator rejects the most common
  injection-relevant characters (`NUL`, `?`, `#`, `../`), so the current
  surface is constrained; full URL-encoding is the next iteration.
- **The per-request `api_key` parameter carries the transcript-exposure
  risk** described above. Environment-variable use is the recommended path
  today; parameter removal is planned.
- **GitHub Actions are tag-pinned, not SHA-pinned.** Major versions are
  guarded by Dependabot grouping rules and the `update-types: [minor,
  patch]` restriction so unreviewed major bumps cannot land. SHA-pinning,
  Sigstore signing, build-provenance attestations, and CycloneDX SBOM
  generation are the next iteration of supply-chain hardening.
