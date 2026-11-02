"""Offline regression tests for the anonymous MCP live-probe exit contract."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from scripts import smoke_no_key_mcp as smoke

RATE_LIMIT_MESSAGE = (
    "Error executing tool semantic_scholar_get_paper: Rate limited. "
    "Get a free API key for faster access: https://www.semanticscholar.org/product/api"
)
SERVER_ERROR_MESSAGE = (
    "Error executing tool semantic_scholar_get_paper: Upstream server error (503)"
)
FIRST_PAGE = {
    "paper": {"paperId": "verified-paper"},
    "citations": [{"citingPaper": {"paperId": "p1"}}],
    "references": [{"citedPaper": {"paperId": "r1"}}],
    "citations_page": {"next": 2},
    "references_page": {"next": 2},
}
SECOND_PAGE = {
    "citations_page": {"offset": 2},
    "citations": [{"citingPaper": {"paperId": "p2"}}],
}


def _result(*, error: bool, message: str) -> SimpleNamespace:
    return SimpleNamespace(isError=error, content=[SimpleNamespace(text=message)])


def test_only_exact_tool_rate_limit_is_an_advisory_condition() -> None:
    with pytest.raises(smoke.UpstreamRateLimited):
        smoke._decode_tool_output(_result(error=True, message=RATE_LIMIT_MESSAGE))
    for message in (
        SERVER_ERROR_MESSAGE,
        "Error executing tool semantic_scholar_search_papers: Rate limited.",
        "Rate limited.",
        "Error executing tool semantic_scholar_get_paper: unexpected 429 response",
    ):
        with pytest.raises(RuntimeError) as error:
            smoke._decode_tool_output(_result(error=True, message=message))
        assert type(error.value) is RuntimeError


def test_success_requires_valid_json() -> None:
    parsed = smoke._decode_tool_output(_result(error=False, message=json.dumps(FIRST_PAGE)))
    assert parsed == FIRST_PAGE
    with pytest.raises(json.JSONDecodeError):
        smoke._decode_tool_output(_result(error=False, message="{not JSON}"))


def _install_fake_protocol(
    monkeypatch: pytest.MonkeyPatch,
    *,
    limited_at: int | None = None,
    fail_at: int | None = None,
) -> None:
    @asynccontextmanager
    async def fake_stdio(_params):
        yield object(), object()

    class FakeSession:
        def __init__(self, _reader, _writer):
            self.calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

        async def initialize(self):
            return None

        async def list_tools(self):
            tools = [SimpleNamespace(name=f"other_{index}") for index in range(13)]
            tools.append(SimpleNamespace(name="semantic_scholar_get_paper"))
            return SimpleNamespace(tools=tools)

        async def call_tool(self, _name, *, arguments):
            assert "params" in arguments
            self.calls += 1
            if self.calls == limited_at:
                return _result(error=True, message=RATE_LIMIT_MESSAGE)
            if self.calls == fail_at:
                return _result(error=True, message=SERVER_ERROR_MESSAGE)
            page = FIRST_PAGE if self.calls == 1 else SECOND_PAGE
            return _result(error=False, message=json.dumps(page))

    monkeypatch.setattr(smoke, "stdio_client", fake_stdio)
    monkeypatch.setattr(smoke, "ClientSession", FakeSession)


@pytest.mark.asyncio
@pytest.mark.parametrize("limited_at,stage", [(1, "first_page"), (2, "second_page")])
async def test_throttling_after_valid_handshake_is_inconclusive(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    limited_at: int,
    stage: str,
) -> None:
    _install_fake_protocol(monkeypatch, limited_at=limited_at)
    assert await smoke._smoke() == smoke.UPSTREAM_RATE_LIMITED
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "INCONCLUSIVE"
    assert payload["reason"] == "UPSTREAM_HTTP_429"
    assert payload["handshake"] == "PASS"
    assert payload["tools_list"] == "PASS"
    assert payload["live_pagination"] == "NOT_VERIFIED"
    assert payload["stage"] == stage


@pytest.mark.asyncio
async def test_real_mcp_error_still_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_protocol(monkeypatch, fail_at=1)
    with pytest.raises(RuntimeError, match="MCP tool returned isError"):
        await smoke._smoke()


@pytest.mark.asyncio
async def test_real_two_page_success_remains_a_pass(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_protocol(monkeypatch)
    assert await smoke._smoke() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PASS"
    assert payload["different_citation_pages"] is True
