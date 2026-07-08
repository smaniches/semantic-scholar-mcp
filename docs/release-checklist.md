# Release Checklist — v1.0.0

> **Historical document.** This checklist describes the manual process used for
> the original v1.0.0 release. Releases are now automated by release-please
> (see [CONTRIBUTING.md](../CONTRIBUTING.md#releasing)); the only remaining
> manual step is the SECURITY.md "Supported Versions" table on minor/major
> bumps.

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

## 4. MCP Registry Submission
Submit `server.json` or open a PR using `mcp-registry-submission.md` at:
https://github.com/modelcontextprotocol/servers
