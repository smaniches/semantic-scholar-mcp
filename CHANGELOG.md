# Changelog

All notable changes documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/). Versioning: [Semantic Versioning](https://semver.org/).

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
