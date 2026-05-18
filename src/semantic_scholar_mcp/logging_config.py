"""Structured JSON logger for stdio-transport MCP servers.

Clients that capture stderr (Claude Desktop, Claude Code) get one
JSON-per-line stream that's trivially parseable by log shippers.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging in production."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def get_logger(name: str = "semantic_scholar_mcp") -> logging.Logger:
    """Return the singleton structured logger for the package."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


__all__ = ["StructuredFormatter", "get_logger"]
