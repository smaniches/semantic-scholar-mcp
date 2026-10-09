"""Unauthenticated end-to-end Streamable HTTP MCP protocol smoke.

Creates a loopback server, negotiates an MCP session, enumerates the public
tools and checks rejection of an invalid ID at the real protocol boundary.
The stdio smoke separately validates real no-key API access. It is deliberate
that this smoke does not send a second anonymous upstream API request: such
calls share Semantic Scholar's external rate-limited public pool.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

def _port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def _wait_until_listening(proc, port):
    for _ in range(60):
        if proc.returncode is not None:
            raise RuntimeError(f"MCP HTTP server exited with code {proc.returncode}")
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
        except OSError:
            await asyncio.sleep(0.25)
            continue
        writer.close()
        await writer.wait_closed()
        return
    raise TimeoutError(f"MCP HTTP server did not accept connections on {port}")


async def _smoke():
    port = _port()
    env = os.environ.copy()
    env.pop("SEMANTIC_SCHOLAR_API_KEY", None)
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "semantic_scholar_mcp",
        "--transport",
        "http",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        env=env,
    )
    try:
        await _wait_until_listening(proc, port)
        async with streamablehttp_client(f"http://127.0.0.1:{port}/mcp") as connection:
            read_stream, write_stream, _ = connection
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                listing = await session.list_tools()
                names = {tool.name for tool in listing.tools}
                assert len(names) == 14, f"Expected 14 MCP tools, got {len(names)}"
                result = await session.call_tool(
                    "semantic_scholar_get_paper",
                    arguments={"params": {"paper_id": "invalid-id"}},
                )
                assert result.isError, "An invalid paper ID should produce a tool error"
                print(
                    json.dumps(
                        {
                            "status": "PASS",
                            "protocol": "MCP Streamable HTTP",
                            "host": "127.0.0.1",
                            "auth": "none",
                            "tool_count": len(names),
                            "invalid_id_rejected": True,
                            "live_upstream_api_test": "covered by separate stdio smoke",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        if proc.returncode is None:
            proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()


if __name__ == "__main__":
    asyncio.run(asyncio.wait_for(_smoke(), timeout=180))
