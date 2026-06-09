# Installing the Semantic Scholar MCP Server

This MCP server provides access to the Semantic Scholar academic graph
(papers, authors, citations, recommendations). It is distributed on PyPI as
`s2-mcp-server` and runs with `uvx` (no manual install step required).

## One-line launch

```bash
uvx s2-mcp-server
```

## MCP client configuration

Add this server to an MCP client by including the following entry in the
client's MCP configuration:

```json
{
  "mcpServers": {
    "semantic-scholar": {
      "command": "uvx",
      "args": ["s2-mcp-server"]
    }
  }
}
```

## Optional API key

The server works without an API key. An optional `SEMANTIC_SCHOLAR_API_KEY`
environment variable can be set to use a Semantic Scholar API key, which
raises rate limits. A free key is available at
https://www.semanticscholar.org/product/api

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

## Remote access (Streamable HTTP)

The server can also be served over the MCP Streamable HTTP transport for
remote clients:

```bash
uvx s2-mcp-server --transport http   # serves http://127.0.0.1:8000/mcp
```

Clients that accept a URL connect with:

```json
{
  "mcpServers": {
    "semantic-scholar": {
      "type": "http",
      "url": "http://127.0.0.1:8000/mcp",
      "headers": { "x-api-key": "your-key-here" }
    }
  }
}
```

The `x-api-key` header (or a `SEMANTIC_SCHOLAR_API_KEY` query parameter) is
optional and scoped to each request; without it the server uses its own
`SEMANTIC_SCHOLAR_API_KEY` environment variable or keyless public access.
