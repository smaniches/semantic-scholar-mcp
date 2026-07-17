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

## Structured tool outputs (deferred decision)

Since v1.7.0 the server deliberately disables the SDK's auto-generated
output schemas (`structured_output=False` on every tool): each tool returns
a single text content block, with `response_format="json"` as the per-call
machine-readable option. Declaring real, typed `outputSchema`s is deferred
to v2.0.0 at the earliest, and only if one of these triggers fires:

1. A downstream consumer demonstrably parses tool results programmatically
   (issue reports, or an application consuming the JSON output in code).
2. The MCP spec adopts per-call response-format negotiation
   (modelcontextprotocol/modelcontextprotocol#1710) or relaxes the
   duplicate-text-block backwards-compatibility guidance (see SEP-1624).
3. Major MCP hosts start feeding `structuredContent` to models *instead of*
   the text block, flipping the token economics in favor of schemas.

## Nested `params` input wrapper

Every tool takes a single Pydantic model argument, so generated input
schemas nest fields under a `params` object rather than exposing top-level
parameters. Flattening is a v2.0.0 candidate: it is a schema-shape
improvement only, and would churn the full test suite and README tool
reference, so it is not worth a mid-1.x break.

## Development status

The package classifies itself as `Development Status :: 4 - Beta` in
`pyproject.toml`. This reflects that documentation and security hardening are
ongoing. The API surface is stable across the v1.x series.
