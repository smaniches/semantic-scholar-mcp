# Usage Examples

Sample tool calls for the Semantic Scholar MCP server.

---

## Setup

### Claude Code (one-liner)

```bash
claude mcp add semantic-scholar -- uvx s2-mcp-server
```

### Claude Desktop

Add to your Claude Desktop config:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
**Linux:** `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "semantic-scholar": {
      "command": "uvx",
      "args": ["s2-mcp-server"],
      "env": {
        "SEMANTIC_SCHOLAR_API_KEY": "your-key-here"
      }
    }
  }
}
```

---

## 1. Search Papers

Search for transformer attention papers from 2023 with at least 100 citations.

```json
{
  "tool": "semantic_scholar_search_papers",
  "arguments": {
    "query": "transformer attention",
    "year": "2023",
    "min_citation_count": 100
  }
}
```

**Expected output:** Markdown-formatted list of papers showing title, authors,
year, citation count, and Semantic Scholar link.

## 2. Get Paper by DOI

Retrieve full details for a specific paper using its DOI.

```json
{
  "tool": "semantic_scholar_get_paper",
  "arguments": {
    "paper_id": "DOI:10.1038/s41586-021-03819-2"
  }
}
```

## 3. Citation Graph Traversal

Get a paper with its citations and references in one call.

```json
{
  "tool": "semantic_scholar_get_paper",
  "arguments": {
    "paper_id": "ARXIV:1706.03762",
    "include_citations": true,
    "include_references": true,
    "citations_limit": 20,
    "references_limit": 20
  }
}
```

**Expected output:** Paper details followed by lists of citing papers and
referenced papers, each with title, year, and citation count.

## 4. Author Search

Find researchers by name.

```json
{
  "tool": "semantic_scholar_search_authors",
  "arguments": {
    "query": "Yoshua Bengio"
  }
}
```

## 5. Paper Recommendations

Get AI-powered recommendations based on a seed paper.

```json
{
  "tool": "semantic_scholar_recommendations",
  "arguments": {
    "paper_id": "ARXIV:1706.03762"
  }
}
```

## 6. JSON Output

Any tool supports `response_format: "json"` for structured output.

```json
{
  "tool": "semantic_scholar_search_papers",
  "arguments": {
    "query": "CRISPR gene editing",
    "year": "2023-2024",
    "limit": 5,
    "response_format": "json"
  }
}
```

**Expected output:** A JSON object with `query`, `total`, and `papers` array
instead of Markdown.

## 7. Bulk Paper Retrieval

Retrieve multiple papers in a single request (max 500).

```json
{
  "tool": "semantic_scholar_bulk_papers",
  "arguments": {
    "paper_ids": [
      "DOI:10.1038/nature12373",
      "ARXIV:2106.15928",
      "PMID:32908142"
    ]
  }
}
```

**Expected output:** JSON with `requested`, `retrieved` counts and paper data.
Any IDs not found are listed in `not_found`.

## 8. Multi-Paper Recommendations

Use positive and negative examples to steer recommendations.

```json
{
  "tool": "semantic_scholar_multi_recommend",
  "arguments": {
    "positive_paper_ids": ["ARXIV:1706.03762", "ARXIV:1810.04805"],
    "negative_paper_ids": ["DOI:10.1038/nature14539"],
    "limit": 20
  }
}
```

## 9. Snippet Search

Search within paper full text and get snippets with context.

```json
{
  "tool": "semantic_scholar_snippet_search",
  "arguments": {
    "query": "scaling laws for language models",
    "year": "2022-2024",
    "limit": 10
  }
}
```

**Expected output:** Matching text snippets with source paper title, section
name, and excerpt.

## 10. Export BibTeX Citation

```json
{
  "tool": "semantic_scholar_export_citation",
  "arguments": {
    "paper_id": "DOI:10.1038/s41586-021-03819-2",
    "format": "bibtex"
  }
}
```

**Expected output:** A BibTeX string for the paper.

## 11. Match Paper by Title

Find the best match for a paper title string.

```json
{
  "tool": "semantic_scholar_match_paper",
  "arguments": {
    "query": "Attention Is All You Need"
  }
}
```

**Expected output:** The best-matching paper with a numeric `matchScore`.

## 12. Server Status

```json
{
  "tool": "semantic_scholar_status",
  "arguments": {}
}
```

**Expected output:** JSON with server version, API key status, rate tier,
and API reachability.
