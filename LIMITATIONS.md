# Known Limitations

This document distinguishes limitations of this server from observed
limitations of its upstream services. Documenting an upstream metadata
inconsistency does not mean the server created it or can independently
resolve it.

## BibTeX export: provenance and verification scope

Bibliographic metadata and generated BibTeX originate from Semantic Scholar.
This MCP server provides access to those records; it does not independently
validate publication identity, version relationships, or citation accuracy.
Inconsistent upstream records are not silently corrected.

`semantic_scholar_export_citation` returns the upstream
`citationStyles.bibtex` unchanged when `format="bibtex"` (the default).
The opt-in `format="bibtex+doi"` mode can insert the DOI from that same
record's `externalIds.DOI` when missing. It rejects missing, malformed, or
contradictory DOI evidence rather than inventing or overwriting an identifier.
Neither mode queries an independent bibliographic source or verifies that
the DOI, date, authors, venue, and other fields refer to the same publication
version. An enriched DOI is **not** a publisher-validated citation.

### Observed upstream example: AlexNet publication versions

**Observation date:** November 12, 2026
**Semantic Scholar paper ID:** `abd1c342495432171beb7ca8fd9551ef13cbd0ff`
**Queried identifier:** `DOI:10.1145/3065386`

The Semantic Scholar Academic Graph response contained:

- `year: 2012` and `publicationDate: 2012-12-03`;
- `externalIds.DBLP: conf/nips/KrizhevskySH12`;
- `venue: Communications of the ACM`, journal volume `60`, pages `84 - 90`;
- `externalIds.DOI: 10.1145/3065386`;
- generated `citationStyles.bibtex` with `year = {2012}`, the ACM journal
  fields, and no `doi` field.

The [ACM publication](https://doi.org/10.1145/3065386) identifies that DOI
with *Communications of the ACM*, **60(6):84–90 (2017)** (published
May 24, 2017). The [original 2012 NeurIPS proceedings
record](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c-Abstract.html)
identifies the earlier conference publication.

**Interpretation:** This particular Semantic Scholar response combines a
2012 publication year with the journal, volume, pages, and DOI of the 2017
ACM publication. Those values are already present in the upstream response;
the MCP did not generate the mixed metadata. Opt-in DOI insertion preserves
the year and other fields and does not reconcile the two publication versions.

This is one reproducible upstream-record observation, **not** a claim about
the frequency of such inconsistencies or their internal cause. The source
record may change. Verify the intended version against the publisher or
proceedings before using the citation in scholarly work.

## Shared anonymous API access and rate limits

Most public Semantic Scholar endpoints can be queried without a key, and a
real no-key MCP stdio smoke succeeded on October 9, 2026. On subsequent
anonymous executions the same API returned HTTP 429 repeatedly, exhausting
the client's bounded retries. This is an external shared-pool limit, not a
guarantee of 1 successful request per second per caller. The server continues
to surface this as a typed rate-limit failure rather than synthesizing data.

A manual/scheduled live smoke exercises real anonymous retrieval and graph
pagination; its failure must not be silently interpreted as a product code
regression. Hermetic CI separately tests the stdio and Streamable HTTP MCP
handshakes, tool discovery, and invalid-input rejection without contacting
the third-party API.

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
