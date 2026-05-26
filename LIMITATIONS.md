# Known Limitations

Explicit boundaries on what this project provides today. These are not
defects — they are tracked gaps for follow-up releases.

## Paper-ID URL encoding

Paper IDs are interpolated into request URL paths without
`urllib.parse.quote`. The validator rejects the most common
injection-relevant characters (`NUL`, `?`, `#`, `../`), so the current
attack surface is constrained. Full URL-encoding is planned for a future
release.

## Per-request `api_key` transcript exposure

The per-request `api_key` tool parameter is **deprecated** (runtime
`DeprecationWarning` emitted since v1.3.x; removal planned for v2.0.0).
Tool-call arguments are typically captured by MCP clients and may be surfaced
in LLM tool-call history. Use the `SEMANTIC_SCHOLAR_API_KEY` environment
variable instead.

## GitHub Actions pin strategy

GitHub Actions are tag-pinned, not SHA-pinned. Major versions are guarded by
Dependabot grouping rules and `update-types: [minor, patch]` restrictions so
unreviewed major bumps cannot land automatically. SHA-pinning, Sigstore
signing, build-provenance attestations, and CycloneDX SBOM generation are
planned for a future supply-chain hardening iteration.

## Development status

The package classifies itself as `Development Status :: 4 - Beta` in
`pyproject.toml`. This reflects that documentation and security hardening are
ongoing. The API surface is stable across the v1.x series.
