"""
Semantic Scholar MCP Server
===========================

MCP (Model Context Protocol) server for accessing Semantic Scholar's 
academic paper database directly within Claude Desktop.

Author: Santiago Maniches (ORCID: 0009-0005-6480-1987)
Organization: TOPOLOGICA LLC (https://topologica.ai)
License: MIT

For documentation: https://github.com/smaniches/semantic-scholar-mcp
"""

from .server import (
    mcp,
    main,
    SemanticScholarError,
    AuthenticationError,
    RateLimitError,
    NotFoundError,
    ValidationError,
    ServerError,
)

__version__ = "1.0.0"
__author__ = "Santiago Maniches"
__email__ = "santiago@topologica.ai"
__org__ = "TOPOLOGICA LLC"
__url__ = "https://topologica.ai"
__orcid__ = "0009-0005-6480-1987"

__all__ = [
    "mcp",
    "main",
    "__version__",
    "SemanticScholarError",
    "AuthenticationError",
    "RateLimitError",
    "NotFoundError",
    "ValidationError",
    "ServerError",
]
