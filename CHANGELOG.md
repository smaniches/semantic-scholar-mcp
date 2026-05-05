# Changelog

All notable changes documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/). Versioning: [Semantic Versioning](https://semver.org/).

## [1.2.2] - 2026-05-05

Trust-repair patch. Documentation, metadata, and test-contract corrections
only — no behavior change to the API surface, no path-encoding changes, no
key-handling redesign, no supply-chain hardening. Those substantive items
remain tracked for v1.3.0.

### Fixed
- `tests/test_api_compatibility.py::test_x_api_key_rejected` was a verbatim
  duplicate of `test_api_key_accepted` but asserted `403`. Replaced with
  `test_bearer_auth_not_accepted`, which sends `Authorization: Bearer <key>`
  (without `x-api-key`) and asserts the response is not silently authenticated.
- `tests/test_property_based.py::test_any_doi_prefix_is_valid` was failing on
  main: the Hypothesis strategy generated `DOI:?` and `_validate_paper_id`
  intentionally rejects `?` as a URL-injection guard. Narrowed the strategy
  to `DOI:[^\s?#]+` (also excluding `../`) to match the validator's contract.
  The validator was *not* weakened.
- `server.json` `version` and `packages[0].version` were stuck at `1.0.0`
  while the package shipped `1.2.1` to PyPI. Synchronized to `1.2.2`.
- `.github/SECURITY.md` "Supported Versions" table was stuck at `1.0.x`.
  Updated to `1.2.x`.
- README "Tools Reference" documented only 7 of the 14 registered tools.
  Added entries for `bulk_search`, `export_citation`, `match_paper`,
  `paper_authors`, `author_batch`, `multi_recommend`, `snippet_search`.
- README claim "Your API key never leaves your machine" was false: the key
  is sent to `api.semanticscholar.org` as the `x-api-key` header. Replaced
  with an accurate description of where the key actually goes.
- README and SECURITY.md disagreed about the per-request `api_key` parameter
  (README instructed its use; SECURITY denied it existed). Reconciled both
  to acknowledge the parameter, document the transcript-exposure risk, and
  recommend the `SEMANTIC_SCHOLAR_API_KEY` environment variable. Parameter
  removal is deferred to v1.3.0.
- README example response for `semantic_scholar_status` showed
  `"version": "1.0.0"`. Updated to `"1.2.2"`.

### Changed
- README opening tagline narrowed from "The most comprehensive MCP server
  for academic research" (no comparison evidence; three live PyPI rivals
  exist) to a measurable claim: "A comprehensive 14-tool MCP server for
  Semantic Scholar academic research workflows."
- CHANGELOG entry for 1.2.0 narrowed "100% S2 API coverage" to
  "14 tools across the Semantic Scholar Graph, Recommendations, and Snippet
  APIs". Historical accuracy is preserved by this change-log entry recording
  the rewrite.
- `pyproject.toml` `Development Status` classifier downgraded from
  `5 - Production/Stable` to `4 - Beta` as a deliberate honesty correction
  while documentation and security hardening are brought into alignment.
  Re-promotion is contingent on the v1.3.0 hardening landing.

### Added
- `tests/test_version_consistency.py`: fails fast in CI if `pyproject.toml`,
  `server.json`, and runtime `__version__` diverge, or if the README install
  command points at the wrong PyPI distribution name.
- `tests/test_docs_tool_surface.py`: fails fast in CI if a registered
  `@mcp.tool` is missing from the README, or if the README advertises a tool
  the server does not register.
- `RELEASE_AUDIT_v1.2.2.md`: scope, validation results, deferred items, and
  attribution note for this release.
- SECURITY.md: explicit "Known Limitations (v1.2.x)" section enumerating
  the items deferred to v1.3.0 (path encoding, per-request `api_key`
  parameter, GitHub Actions SHA-pinning, Sigstore/SBOM/attestation).

## [1.2.1] - 2026-04-06

### Fixed
- `bulk_search`: HTTP 400 caused by requesting unsupported fields (tldr, influentialCitationCount, openAccessPdf); added PAPER_BULK_SEARCH_FIELDS
- `snippet_search`: API key not forwarded; reverted auth header from Authorization: Bearer back to x-api-key (the Bearer migration in v1.0.2 broke the snippet endpoint)
- mypy no-any-return error in export_citation
- Pre-existing ruff lint (E501, F401, F841) and format issues
- Tool registration tests updated from 7 to 14 tools

### Changed
- Version bump to 1.2.1

## [1.2.0] - 2026-04-05

### Added
- New tool: `semantic_scholar_match_paper` — find paper by exact title match with match score
- New tool: `semantic_scholar_paper_authors` — full author profiles for a paper's authors
- New tool: `semantic_scholar_author_batch` — batch author retrieval (up to 1000)
- New tool: `semantic_scholar_multi_recommend` — multi-paper recommendations with positive and negative examples
- New tool: `semantic_scholar_snippet_search` — search within paper full text (requires API key)
- `from_pool` parameter on recommendations: choose "recent" (default) or "all-cs" paper pool
- Smart status output: rate tier info, API key setup tip for unauthenticated users
- Input sanitization: reject null bytes, path traversal, and query injection in paper IDs
- Friendly 429 error messages: API key signup URL for unauthenticated users

### Changed
- Version bump to 1.2.0 (14 tools across the Semantic Scholar Graph, Recommendations, and Snippet APIs)

## [1.1.0] - 2026-04-05

### Added
- New tool: `semantic_scholar_bulk_search` — sorted search with cursor-based pagination for large result sets
- New tool: `semantic_scholar_export_citation` — BibTeX citation export for any paper
- Parallel fetching of citations and references in `get_paper_details` via `asyncio.gather`

### Changed
- Version bump to 1.1.0

## [1.0.3] - 2026-04-04

### Fixed
- Filter tldr from citations endpoint (missed in v1.0.2); use PAPER_SEARCH_FIELDS_LITE for all sub-endpoints
- Fix incorrect test asserting citations accepts tldr

### Changed
- Version bump to 1.0.3

## [1.0.2] - 2026-04-01

### Fixed
- Auth: migrated from deprecated x-api-key to Authorization: Bearer
- Removed unsupported aliases field from author queries
- Filtered tldr from recommendations, author/papers, and references endpoints

### Added
- PAPER_SEARCH_FIELDS_LITE constant for sub-endpoints with restricted field support
- Automated S2 API compatibility test suite (14 tests)
- Weekly CI workflow to catch S2 API changes early

## [1.0.0] - 2026-02-23

### Changed
- Distribution moved to GitHub releases only (no PyPI)
- Replaced broken PyPI badge with GitHub Release badge

### Added
- SECURITY.md with vulnerability reporting process
- CODEOWNERS file
- py.typed marker for type checker compatibility
- GitHub Actions CI (test matrix: Python 3.10–3.12, lint, type check)
- GitHub issue templates (bug report, feature request)
- Pull request template
- CHANGELOG.md
- Comprehensive pyproject.toml metadata (classifiers, keywords, URLs, ruff config)

### Security
- Environment-variable-only API key management enforced
- Input validation documented
- HTTPS-only API communication

## [0.2.0] - 2026-01-18

### Added
- Full Semantic Scholar API coverage: paper search, author lookup, recommendations, bulk retrieval
- Support for all paper ID formats: DOI, ArXiv, PubMed, ACL, CorpusId
- API key authentication with rate limit handling
- Claude Desktop integration via stdio MCP protocol

## [0.1.0] - 2025-11-25

### Added
- Initial release
