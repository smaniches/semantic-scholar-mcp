# Basic Usage Examples

Sample tool calls for the Semantic Scholar MCP server.

## 1. Paper Search

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

**Expected output:** A markdown-formatted list of papers matching the query, each showing title, authors, year, citation count, and a Semantic Scholar link.

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

**Expected output:** Detailed paper information including title, abstract, authors with affiliations, citation/reference counts, publication venue, and links.

## 3. Author Search

Find researchers by name.

```json
{
  "tool": "semantic_scholar_search_authors",
  "arguments": {
    "query": "Yoshua Bengio"
  }
}
```

**Expected output:** A list of matching authors showing name, h-index, paper count, citation count, and affiliations.

## 4. Paper Recommendations

Get AI-powered recommendations based on a seed paper.

```json
{
  "tool": "semantic_scholar_recommendations",
  "arguments": {
    "paper_id": "ARXIV:1706.03762"
  }
}
```

**Expected output:** A list of recommended papers related to the seed paper ("Attention Is All You Need"), ranked by relevance.
