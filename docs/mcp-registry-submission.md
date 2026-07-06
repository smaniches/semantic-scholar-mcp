# Add smaniches/semantic-scholar-mcp — Semantic Scholar API server

## Server Info
- **Name:** semantic-scholar-mcp
- **Author:** Santiago Maniches, TOPOLOGICA LLC (ORCID: [0009-0005-6480-1987](https://orcid.org/0009-0005-6480-1987))
- **Repository:** https://github.com/smaniches/semantic-scholar-mcp
- **Description:** MCP server providing access to 200M+ academic papers via the Semantic Scholar API. Supports paper search, author lookup, citation graphs, recommendations, bulk retrieval, full-text snippet search, and BibTeX export.

## Tools
- `semantic_scholar_search_papers` — Search for academic papers with advanced filters
- `semantic_scholar_get_paper` — Get detailed information about a specific paper
- `semantic_scholar_search_authors` — Search for academic authors by name
- `semantic_scholar_get_author` — Get author profile with publications
- `semantic_scholar_recommendations` — Get AI-powered paper recommendations based on a seed paper
- `semantic_scholar_bulk_papers` — Retrieve multiple papers in a single request (max 500)
- `semantic_scholar_bulk_search` — Search with sorting and cursor-based pagination for large result sets
- `semantic_scholar_export_citation` — Export a citation for a paper in BibTeX format
- `semantic_scholar_match_paper` — Find the single best paper matching a title string
- `semantic_scholar_paper_authors` — Get full author profiles for a paper's authors
- `semantic_scholar_author_batch` — Retrieve multiple authors in a single request (max 1000)
- `semantic_scholar_multi_recommend` — Recommendations from multiple positive/negative example papers
- `semantic_scholar_snippet_search` — Search within paper full text, returning snippets with context
- `semantic_scholar_status` — Check server health and API connectivity status

## Installation

```bash
# Run directly from PyPI (recommended)
uvx s2-mcp-server

# Or from source
git clone https://github.com/smaniches/semantic-scholar-mcp.git
cd semantic-scholar-mcp
pip install -e .
```

## Claude Desktop Configuration

```json
{
  "mcpServers": {
    "semantic-scholar": {
      "command": "uvx",
      "args": ["s2-mcp-server"],
      "env": {
        "SEMANTIC_SCHOLAR_API_KEY": "your-api-key-here"
      }
    }
  }
}
```
