# 🔬 Semantic Scholar MCP Server

[![PyPI version](https://badge.fury.io/py/semantic-scholar-mcp.svg)](https://pypi.org/project/semantic-scholar-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP](https://img.shields.io/badge/MCP-Compatible-blue)](https://modelcontextprotocol.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**The most comprehensive MCP server for academic research.** Direct access to 200M+ papers from [Semantic Scholar](https://www.semanticscholar.org/) within Claude Desktop.

---

## 🚀 Features

| Tool | Description |
|------|-------------|
| `semantic_scholar_search_papers` | Advanced paper search with filters (year, field, citations, open access) |
| `semantic_scholar_get_paper` | Full paper details with optional citations and references |
| `semantic_scholar_search_authors` | Find researchers by name |
| `semantic_scholar_get_author` | Author profiles with h-index, publications, affiliations |
| `semantic_scholar_recommendations` | AI-powered related paper discovery |
| `semantic_scholar_bulk_papers` | Batch retrieval of up to 500 papers |

**Supported ID Formats:**
- Semantic Scholar ID
- DOI (`DOI:10.1038/...`)
- ArXiv (`ARXIV:2106.15928`)
- PubMed (`PMID:32908142`)
- ACL (`ACL:P19-1285`)
- CorpusId (`CorpusId:215416146`)

---

## 📦 Installation

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

## 🔑 Get Your API Key

1. Go to [Semantic Scholar API](https://www.semanticscholar.org/product/api)
2. Sign up for a free API key
3. Note your rate limit (typically 1-100 requests/second depending on tier)

---

## ⚙️ Configuration

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

## 💡 Usage Examples

### Search Papers
```
Search for "transformer attention mechanism" papers from 2023 with at least 100 citations
```

### Get Paper Details
```
Get details for DOI:10.1038/s41586-021-03819-2 including its top 20 citations
```

### Find Related Papers
```
Get recommendations based on paper 649def34f8be52c8b66281af98ae884c09aef38b
```

### Author Search
```
Find author "Yoshua Bengio" and list their recent publications
```

### Bulk Retrieval
```
Retrieve these papers: DOI:10.1038/nature12373, ARXIV:2106.15928, PMID:32908142
```

---

## 📊 Rate Limits

| Tier | Requests/Second | How to Get |
|------|-----------------|------------|
| No API Key | 1 req/sec | Default |
| Free API Key | 1 req/sec | [Sign up](https://www.semanticscholar.org/product/api) |
| Academic Partner | 10-100 req/sec | Apply via S2 |

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│  Claude Desktop │────▶│  semantic-scholar-mcp │────▶│ Semantic Scholar│
│   (MCP Client)  │◀────│     (This Server)     │◀────│      API        │
└─────────────────┘     └──────────────────────┘     └─────────────────┘
        │                         │                          │
        │ stdio (JSON-RPC)        │ Your API Key             │ HTTPS
        │ Local process           │ Local machine            │ 200M+ papers
```

**Your API key never leaves your machine.** The MCP server runs locally.

---

## 🛠️ Development

```bash
# Clone
git clone https://github.com/smaniches/semantic-scholar-mcp.git
cd semantic-scholar-mcp

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

## 👤 Author

**Santiago Maniches**
- 🏢 Founder & CEO, [TOPOLOGICA LLC](https://topologica.ai)
- 🔬 ORCID: [0009-0005-6480-1987](https://orcid.org/0009-0005-6480-1987)
- 💼 LinkedIn: [santiago-maniches](https://www.linkedin.com/in/santiago-maniches/)
- 🌐 Website: [topologica.ai](https://topologica.ai)

---

## 🤝 Contributing

Contributions welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md).

---

## 📬 Support

- 🐛 Issues: [GitHub Issues](https://github.com/smaniches/semantic-scholar-mcp/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/smaniches/semantic-scholar-mcp/discussions)
- 📧 Contact: santiago@topologica.ai

---

## ⭐ Star History

If this tool helps your research, please star the repo!

---

<p align="center">
  <b>Built with ❤️ by <a href="https://topologica.ai">TOPOLOGICA LLC</a></b><br>
  <i>Advancing computational research through topological intelligence</i>
</p>
