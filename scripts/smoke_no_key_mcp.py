"""Live no-key smoke for the published MCP stdio protocol and graph pagination.

Run on a networked environment with dependencies installed:
    SEMANTIC_SCHOLAR_API_KEY= python scripts/smoke_no_key_mcp.py

This intentionally fails (rather than reporting a false pass) when unauthenticated
Semantic Scholar availability is blocked or rate-limited. The regular hermetic
test suite is independent of this external availability check.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PAPER_ID = "ARXIV:1706.03762"
EXPECTED_TOOLS = 14


def _decode_tool_output(result):
    if result.isError:
        payload = " | ".join(block.text for block in result.content if hasattr(block, "text"))
        raise RuntimeError(f"MCP tool returned isError: {payload[:500]}")
    content = [block.text for block in result.content if hasattr(block, "text")]
    if len(content) != 1:
        raise AssertionError(f"Expected one text content block, got {len(content)}")
    return json.loads(content[0])


async def _smoke():
    env = os.environ.copy()
    env.pop("SEMANTIC_SCHOLAR_API_KEY", None)
    # No API key is available to the subprocess, even if the runner has one.
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "semantic_scholar_mcp"],
        env=env,
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            listing = await session.list_tools()
            names = {tool.name for tool in listing.tools}
            assert len(names) == EXPECTED_TOOLS, f"Expected 14 tools, got {len(names)}"
            assert "semantic_scholar_get_paper" in names

            first = _decode_tool_output(
                await session.call_tool(
                    "semantic_scholar_get_paper",
                    arguments={
                        "params": {
                            "paper_id": PAPER_ID,
                            "include_citations": True,
                            "include_references": True,
                            "citations_limit": 2,
                            "references_limit": 2,
                            "response_format": "json",
                        }
                    },
                )
            )
            assert first["paper"]["paperId"], "Missing Semantic Scholar paper ID"
            citations = first["citations"]
            references = first["references"]
            assert 0 < len(citations) <= 2
            assert 0 < len(references) <= 2
            citation_next = first["citations_page"]["next"]
            reference_next = first["references_page"]["next"]
            assert isinstance(citation_next, int), f"Missing citation next: {citation_next}"
            assert isinstance(reference_next, int), f"Missing reference next: {reference_next}"

            second = _decode_tool_output(
                await session.call_tool(
                    "semantic_scholar_get_paper",
                    arguments={
                        "params": {
                            "paper_id": PAPER_ID,
                            "include_citations": True,
                            "citations_offset": citation_next,
                            "citations_limit": 2,
                            "response_format": "json",
                        }
                    },
                )
            )
            assert second["citations_page"]["offset"] == citation_next
            old_ids = {x["citingPaper"]["paperId"] for x in citations}
            new_ids = {x["citingPaper"]["paperId"] for x in second["citations"]}
            assert new_ids, "Second citation page was empty"
            assert old_ids.isdisjoint(new_ids), "Citation pages overlap"
            assert "references" not in second, "Unrequested references should be absent"
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "protocol": "MCP stdio",
                        "auth": "none",
                        "tool_count": len(names),
                        "paper_id": first["paper"]["paperId"],
                        "first_citations": len(citations),
                        "first_references": len(references),
                        "next_citations_offset": citation_next,
                        "next_references_offset": reference_next,
                        "second_citations": len(new_ids),
                        "different_citation_pages": True,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


if __name__ == "__main__":
    asyncio.run(asyncio.wait_for(_smoke(), timeout=180))
