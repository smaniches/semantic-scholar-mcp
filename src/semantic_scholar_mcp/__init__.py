"""
Semantic Scholar MCP Server.

MCP (Model Context Protocol) server providing access to Semantic Scholar's
academic graph (200M+ papers) from any MCP-compatible client.

Author: Santiago Maniches (ORCID: 0009-0005-6480-1987)
Organization: TOPOLOGICA LLC (https://topologica.ai)
License: MIT
Repository: https://github.com/smaniches/semantic-scholar-mcp
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("s2-mcp-server")
except PackageNotFoundError:  # pragma: no cover — only when running uninstalled
    __version__ = "0.0.0+local"

from .errors import (  # after __version__ to avoid circular import
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    SemanticScholarError,
    ServerError,
    ValidationError,
)
from .server import main, mcp  # depends on __version__

__author__ = "Santiago Maniches"
__email__ = "santiago@topologica.ai"
__org__ = "TOPOLOGICA LLC"
__url__ = "https://topologica.ai"
__orcid__ = "0009-0005-6480-1987"

__all__ = [
    "AuthenticationError",
    "NotFoundError",
    "RateLimitError",
    "SemanticScholarError",
    "ServerError",
    "ValidationError",
    "__author__",
    "__email__",
    "__orcid__",
    "__org__",
    "__url__",
    "__version__",
    "main",
    "mcp",
]
