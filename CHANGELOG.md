# Changelog

All notable changes documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/). Versioning: [Semantic Versioning](https://semver.org/).

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
- Version bump to 1.2.0 (14 tools, 100% S2 API coverage)

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
