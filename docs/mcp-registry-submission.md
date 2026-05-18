# Add smaniches/semantic-scholar-mcp — Semantic Scholar API server

## Server Info
- **Name:** semantic-scholar-mcp
- **Author:** Santiago Maniches, TOPOLOGICA LLC (ORCID: [0009-0005-6480-1987](https://orcid.org/0009-0005-6480-1987))
- **Repository:** https://github.com/smaniches/semantic-scholar-mcp
- **Description:** MCP server providing access to 200M+ academic papers via the Semantic Scholar API. Supports paper search, author lookup, citation graphs, recommendations, and bulk retrieval.

## Tools
- `semantic_scholar_search_papers` — Search for academic papers with advanced filters
- `semantic_scholar_get_paper` — Get detailed information about a specific paper
- `semantic_scholar_search_authors` — Search for academic authors by name
- `semantic_scholar_get_author` — Get author profile with publications
- `semantic_scholar_recommendations` — Get AI-powered paper recommendations based on a seed paper
- `semantic_scholar_bulk_papers` — Retrieve multiple papers in a single request (max 500)
- `semantic_scholar_status` — Check server health and API connectivity status

## Installation

```bash
git clone https://github.com/smaniches/semantic-scholar-mcp.git
cd semantic-scholar-mcp
pip install -e .
```

## Claude Desktop Configuration

```json
{
  "mcpServers": {
    "semantic-scholar": {
      "command": "python",
      "args": ["-m", "semantic_scholar_mcp"],
      "env": {
        "SEMANTIC_SCHOLAR_API_KEY": "your-api-key-here"
      }
    }
  }
}
```
