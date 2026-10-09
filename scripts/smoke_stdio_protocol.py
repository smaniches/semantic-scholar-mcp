"""Deterministic real MCP stdio process handshake and input-validation smoke.

Unlike scripts/smoke_no_key_mcp.py this never calls the upstream API.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def _smoke():
    env = os.environ.copy()
    env.pop("SEMANTIC_SCHOLAR_API_KEY", None)
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "semantic_scholar_mcp"], env=env
    )
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            listing = await session.list_tools()
            names = {tool.name for tool in listing.tools}
            assert len(names) == 14, f"Expected 14 MCP tools, got {len(names)}"
            result = await session.call_tool(
                "semantic_scholar_get_paper",
                arguments={"params": {"paper_id": "not-a-valid-id"}},
            )
            assert result.isError, "Invalid paper ID unexpectedly accepted"
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "protocol": "MCP stdio",
                        "auth": "none",
                        "tool_count": len(names),
                        "invalid_id_rejected": True,
                        "external_api_calls": 0,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


if __name__ == "__main__":
    asyncio.run(asyncio.wait_for(_smoke(), timeout=60))
