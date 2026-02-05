# Semantic Scholar MCP Server

[![PyPI version](https://badge.fury.io/py/semantic-scholar-mcp.svg)](https://pypi.org/project/semantic-scholar-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP](https://img.shields.io/badge/MCP-Compatible-blue)](https://modelcontextprotocol.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**The most comprehensive MCP server for academic research.** Direct access to 200M+ papers from [Semantic Scholar](https://www.semanticscholar.org/) within Claude Desktop.

---

## Installation

### Option 1: pip (Recommended)
```bash
pip install semantic-scholar-mcp
```

### Option 2: From Source
```bash
git clone https://github.com/smaniches/semantic-scholar-mcp.git
cd semantic-scholar-mcp
pip install -e .
```

---

## Configuration

### API Key Options

You can provide your API key in two ways:

1. **Environment Variable** (recommended for persistent use):
   ```bash
   export SEMANTIC_SCHOLAR_API_KEY="your-api-key-here"
   ```

2. **Per-Request Parameter** (overrides env var):
   ```json
   {
     "api_key": "your-api-key-here"
   }
   ```

Get a free API key at: https://www.semanticscholar.org/product/api

### Claude Desktop Setup

Add to your Claude Desktop config file:

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Linux:** `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "semantic_scholar": {
      "command": "python",
      "args": ["-m", "semantic_scholar_mcp"],
      "env": {
        "SEMANTIC_SCHOLAR_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

Then **restart Claude Desktop**.

---

## Supported ID Formats

The server accepts the following paper identifier formats:

| Format | Pattern | Example |
|--------|---------|---------|
| Semantic Scholar ID | 40-character hex | `649def34f8be52c8b66281af98ae884c09aef38b` |
| DOI | `DOI:xxx` | `DOI:10.1038/s41586-021-03819-2` |
| ArXiv | `ARXIV:xxx` | `ARXIV:2106.15928` or `ARXIV:2106.15928v2` |
| PubMed | `PMID:xxx` | `PMID:32908142` |
| Corpus ID | `CorpusId:xxx` | `CorpusId:215416146` |
| ACL | `ACL:xxx` | `ACL:P19-1285` |
| URL | `URL:xxx` | `URL:https://arxiv.org/abs/2106.15928` |

---

## Tools Reference

### 1. `semantic_scholar_search_papers`

Search for academic papers with advanced filters.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Search query (supports AND, OR, NOT operators and "phrase search") |
| `year` | string | No | Year filter: `"2024"`, `"2020-2024"`, or `"2020-"` |
| `fields_of_study` | string[] | No | Filter by fields: `["Computer Science", "Biology"]` |
| `publication_types` | string[] | No | Filter by type: `["Review", "JournalArticle"]` |
| `open_access_only` | boolean | No | Only return open access papers (default: false) |
| `min_citation_count` | integer | No | Minimum citation count |
| `limit` | integer | No | Max results 1-100 (default: 10) |
| `offset` | integer | No | Pagination offset (default: 0) |
| `response_format` | string | No | `"markdown"` or `"json"` (default: markdown) |
| `api_key` | string | No | Override environment API key |

**Example:**
```
Search for "transformer attention mechanism" papers from 2023 with at least 100 citations
```

**JSON Example:**
```json
{
  "query": "transformer attention mechanism",
  "year": "2023",
  "min_citation_count": 100,
  "fields_of_study": ["Computer Science"],
  "limit": 20
}
```

---

### 2. `semantic_scholar_get_paper`

Get detailed information about a specific paper.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `paper_id` | string | Yes | Paper ID in any supported format |
| `include_citations` | boolean | No | Include citing papers (default: false) |
| `include_references` | boolean | No | Include referenced papers (default: false) |
| `citations_limit` | integer | No | Max citations to return 1-100 (default: 10) |
| `references_limit` | integer | No | Max references to return 1-100 (default: 10) |
| `response_format` | string | No | `"markdown"` or `"json"` (default: markdown) |
| `api_key` | string | No | Override environment API key |

**Example:**
```
Get details for DOI:10.1038/s41586-021-03819-2 including its top 20 citations
```

**JSON Example:**
```json
{
  "paper_id": "DOI:10.1038/s41586-021-03819-2",
  "include_citations": true,
  "citations_limit": 20
}
```

---

### 3. `semantic_scholar_search_authors`

Search for academic authors by name.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Author name to search |
| `limit` | integer | No | Max results 1-100 (default: 10) |
| `offset` | integer | No | Pagination offset (default: 0) |
| `response_format` | string | No | `"markdown"` or `"json"` (default: markdown) |
| `api_key` | string | No | Override environment API key |

**Example:**
```
Find author "Yoshua Bengio"
```

**JSON Example:**
```json
{
  "query": "Yoshua Bengio",
  "limit": 5
}
```

---

### 4. `semantic_scholar_get_author`

Get author profile with publications.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `author_id` | string | Yes | Semantic Scholar author ID |
| `include_papers` | boolean | No | Include publications (default: true) |
| `papers_limit` | integer | No | Max papers to return 1-100 (default: 20) |
| `response_format` | string | No | `"markdown"` or `"json"` (default: markdown) |
| `api_key` | string | No | Override environment API key |

**Example:**
```
Get author profile for author ID 1741101 with their top 50 publications
```

**JSON Example:**
```json
{
  "author_id": "1741101",
  "include_papers": true,
  "papers_limit": 50
}
```

---

### 5. `semantic_scholar_recommendations`

Get AI-powered paper recommendations based on a seed paper.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `paper_id` | string | Yes | Seed paper ID in any supported format |
| `limit` | integer | No | Max recommendations 1-100 (default: 10) |
| `response_format` | string | No | `"markdown"` or `"json"` (default: markdown) |
| `api_key` | string | No | Override environment API key |

**Example:**
```
Get recommendations based on paper 649def34f8be52c8b66281af98ae884c09aef38b
```

**JSON Example:**
```json
{
  "paper_id": "ARXIV:1706.03762",
  "limit": 15
}
```

---

### 6. `semantic_scholar_bulk_papers`

Retrieve multiple papers in a single request (max 500).

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `paper_ids` | string[] | Yes | List of paper IDs (max 500) |
| `response_format` | string | No | `"markdown"` or `"json"` (default: json) |
| `api_key` | string | No | Override environment API key |

**Example:**
```
Retrieve these papers: DOI:10.1038/nature12373, ARXIV:2106.15928, PMID:32908142
```

**JSON Example:**
```json
{
  "paper_ids": [
    "DOI:10.1038/nature12373",
    "ARXIV:2106.15928",
    "PMID:32908142"
  ]
}
```

---

### 7. `semantic_scholar_status`

Check server health and API connectivity status.

**Parameters:** None

**Example:**
```
Check Semantic Scholar API status
```

**Response:**
```json
{
  "server": "semantic-scholar-mcp",
  "version": "1.1.0",
  "api_key_configured": true,
  "timestamp": "2025-01-15T12:00:00.000000+00:00",
  "api_reachable": true
}
```

---

## Rate Limits

| Tier | Requests/Second | How to Get |
|------|-----------------|------------|
| No API Key | 1 req/sec | Default |
| Free API Key | 1 req/sec | [Sign up](https://www.semanticscholar.org/product/api) |
| Academic Partner | 10-100 req/sec | Apply via S2 |

The server automatically handles rate limiting with:
- Request serialization to enforce minimum intervals
- Exponential backoff retry for 429 (rate limit) and 503 (service unavailable) errors
- Maximum 3 retries with jitter

---

## Architecture

```
+-----------------+     +----------------------+     +-----------------+
|  Claude Desktop |---->|  semantic-scholar-mcp |---->| Semantic Scholar|
|   (MCP Client)  |<----|     (This Server)     |<----+      API        |
+-----------------+     +----------------------+     +-----------------+
        |                         |                          |
        | stdio (JSON-RPC)        | Your API Key             | HTTPS
        | Local process           | Local machine            | 200M+ papers
```

**Your API key never leaves your machine.** The MCP server runs locally.

---

## Development

```bash
# Clone
git clone https://github.com/smaniches/semantic-scholar-mcp.git
cd semantic-scholar-mcp

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=src/semantic_scholar_mcp --cov-report=term-missing

# Type checking
mypy src/
```

---

## License

MIT License - see [LICENSE](LICENSE) file.

---

## Author

**Santiago Maniches**
- Founder & CEO, [TOPOLOGICA LLC](https://topologica.ai)
- ORCID: [0009-0005-6480-1987](https://orcid.org/0009-0005-6480-1987)
- LinkedIn: [santiago-maniches](https://www.linkedin.com/in/santiago-maniches/)
- Website: [topologica.ai](https://topologica.ai)

---

## Contributing

Contributions welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md).

---

## Support

- Issues: [GitHub Issues](https://github.com/smaniches/semantic-scholar-mcp/issues)
- Discussions: [GitHub Discussions](https://github.com/smaniches/semantic-scholar-mcp/discussions)
- Contact: santiago@topologica.ai

---

<p align="center">
  <b>Built by <a href="https://topologica.ai">TOPOLOGICA LLC</a></b><br>
  <i>Advancing computational research through topological intelligence</i>
</p>
