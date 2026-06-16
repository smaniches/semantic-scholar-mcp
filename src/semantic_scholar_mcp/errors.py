"""Typed exception hierarchy for Semantic Scholar API errors.

Mapped from HTTP status codes in :func:`client.handle_error`. Tools surface
these as MCP ``ToolError`` instances so clients see actionable messages.
"""

from __future__ import annotations


class SemanticScholarError(Exception):
    """Base exception for Semantic Scholar MCP."""

    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)


class AuthenticationError(SemanticScholarError):
    """API key invalid or missing (401/403)."""


class RateLimitError(SemanticScholarError):
    """Rate limit exceeded (429)."""

    def __init__(self, message: str, retry_after: float | None = None):
        self.retry_after = retry_after
        super().__init__(message, status_code=429)


class NotFoundError(SemanticScholarError):
    """Paper/author not found (404)."""


class ValidationError(SemanticScholarError):
    """Bad request — invalid parameters (400)."""


class ServerError(SemanticScholarError):
    """Semantic Scholar server error (500/502/503)."""


__all__ = [
    "AuthenticationError",
    "NotFoundError",
    "RateLimitError",
    "SemanticScholarError",
    "ServerError",
    "ValidationError",
]
