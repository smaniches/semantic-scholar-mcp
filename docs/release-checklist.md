# Release Checklist — v1.0.0

## 1. Push
```bash
git log -1 --stat   # review what you're pushing
git push origin main
```

## 2. GitHub Repository Settings
Go to https://github.com/smaniches/semantic-scholar-mcp, click the gear icon next to "About", and set:
- **Description:** MCP server for Semantic Scholar — 200M+ academic papers in Claude Desktop
- **Website:** https://topologica.ai
- **Topics:** mcp, semantic-scholar, academic-research, claude, python, llm, research-tools, model-context-protocol

## 3. Create GitHub Release
- **Tag:** v1.0.0
- **Title:** Production Release v1.0.0
- **Body:** paste the [1.0.0] section from CHANGELOG.md

## 4. MCP Registry — automated (no manual step)
`server.json` is published to the official MCP Registry automatically by the
`publish-mcp-registry` job in [`publish.yml`](../.github/workflows/publish.yml):
it authenticates with GitHub OIDC (no secrets) and runs after the release's
PyPI publish succeeds. No manual submission is required — after a release, just
confirm the entry at https://registry.modelcontextprotocol.io.
