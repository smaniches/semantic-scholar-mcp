"""
Streamable HTTP transport tests.

Covers:
- CLI/env parsing into TransportConfig (flags, env fallbacks, precedence)
- Per-request API key extraction (header, query params, Smithery ?config=)
- RequestApiKeyMiddleware contextvar binding and reset
- Refcounted _lifespan (shared client survives nested/per-request entries)
- main() transport dispatch and run_http() uvicorn wiring
- Full end-to-end Streamable HTTP round-trip against a real uvicorn server
"""

from __future__ import annotations

import asyncio
import base64
import json
import socket
import warnings

import pytest
import respx
from httpx import Response
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.requests import Request

from semantic_scholar_mcp import __version__
from semantic_scholar_mcp import client as client_mod
from semantic_scholar_mcp import server as server_mod
from semantic_scholar_mcp.client import _request_api_key, get_headers, make_request
from semantic_scholar_mcp.server import SEMANTIC_SCHOLAR_API_BASE, _lifespan, mcp
from semantic_scholar_mcp.transport import (
    DEFAULT_HOST,
    DEFAULT_PATH,
    DEFAULT_PORT,
    RequestApiKeyMiddleware,
    TransportConfig,
    _env_flag,
    _with_configured_hosts,
    build_http_app,
    extract_api_key,
    parse_transport_config,
    run_http,
)

TRANSPORT_ENV_VARS = (
    "MCP_TRANSPORT",
    "MCP_HOST",
    "MCP_PORT",
    "PORT",
    "MCP_PATH",
    "MCP_STATELESS_HTTP",
    "MCP_JSON_RESPONSE",
    "MCP_ALLOWED_HOSTS",
)


@pytest.fixture(autouse=True)
def clean_transport_env(monkeypatch):
    """Strip transport env vars so each test starts from pure defaults."""
    for name in TRANSPORT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def request_key():
    """Bind a request-scoped API key for the duration of a test."""
    token = _request_api_key.set("ctx-key-123")
    yield "ctx-key-123"
    _request_api_key.reset(token)


# ===============================================================================
# ENV FLAG / PORT RESOLUTION
# ===============================================================================


class TestEnvFlag:
    def test_unset_returns_default(self):
        assert _env_flag("MCP_STATELESS_HTTP", default=True) is True
        assert _env_flag("MCP_STATELESS_HTTP", default=False) is False

    def test_truthy_values(self, monkeypatch):
        for value in ("1", "true", "YES", " on "):
            monkeypatch.setenv("MCP_JSON_RESPONSE", value)
            assert _env_flag("MCP_JSON_RESPONSE", default=False) is True

    def test_falsy_values(self, monkeypatch):
        for value in ("0", "false", "No", " OFF "):
            monkeypatch.setenv("MCP_JSON_RESPONSE", value)
            assert _env_flag("MCP_JSON_RESPONSE", default=True) is False

    def test_unrecognized_returns_default(self, monkeypatch):
        monkeypatch.setenv("MCP_STATELESS_HTTP", "maybe")
        assert _env_flag("MCP_STATELESS_HTTP", default=True) is True
        assert _env_flag("MCP_STATELESS_HTTP", default=False) is False


# ===============================================================================
# CLI / ENV PARSING
# ===============================================================================


class TestParseTransportConfig:
    def test_defaults_are_stdio(self):
        config = parse_transport_config([])
        assert config == TransportConfig(
            transport="stdio",
            host=DEFAULT_HOST,
            port=DEFAULT_PORT,
            path=DEFAULT_PATH,
            stateless=True,
            json_response=True,
        )

    def test_http_flags(self):
        config = parse_transport_config(
            ["--transport", "http", "--host", "0.0.0.0", "--port", "9001", "--path", "/api/mcp"]
        )
        assert config.transport == "http"
        assert config.host == "0.0.0.0"
        assert config.port == 9001
        assert config.path == "/api/mcp"

    def test_allowed_host_flag_repeats(self):
        config = parse_transport_config(
            ["--allowed-host", "s2.example.org:*", "--allowed-host", "10.0.0.5:8080"]
        )
        assert config.allowed_hosts == ("s2.example.org:*", "10.0.0.5:8080")

    def test_allowed_hosts_env_csv(self, monkeypatch):
        monkeypatch.setenv("MCP_ALLOWED_HOSTS", " s2.example.org:* , 10.0.0.5:8080 ,, ")
        config = parse_transport_config([])
        assert config.allowed_hosts == ("s2.example.org:*", "10.0.0.5:8080")

    def test_streamable_http_alias_normalizes_to_http(self):
        config = parse_transport_config(["--transport", "streamable-http"])
        assert config.transport == "http"

    def test_env_transport_fallback(self, monkeypatch):
        monkeypatch.setenv("MCP_TRANSPORT", "http")
        assert parse_transport_config([]).transport == "http"

    def test_cli_flag_beats_env(self, monkeypatch):
        monkeypatch.setenv("MCP_TRANSPORT", "http")
        monkeypatch.setenv("MCP_HOST", "10.0.0.1")
        config = parse_transport_config(["--transport", "stdio", "--host", "127.0.0.1"])
        assert config.transport == "stdio"
        assert config.host == "127.0.0.1"

    def test_env_host_and_path_fallbacks(self, monkeypatch):
        monkeypatch.setenv("MCP_HOST", "0.0.0.0")
        monkeypatch.setenv("MCP_PATH", "/custom")
        config = parse_transport_config([])
        assert config.host == "0.0.0.0"
        assert config.path == "/custom"

    def test_invalid_env_transport_rejected(self, monkeypatch):
        """argparse skips `choices` validation for defaults; we must not."""
        monkeypatch.setenv("MCP_TRANSPORT", "bogus")
        with pytest.raises(SystemExit):
            parse_transport_config([])

    def test_invalid_cli_transport_rejected(self):
        with pytest.raises(SystemExit):
            parse_transport_config(["--transport", "carrier-pigeon"])

    def test_mcp_port_env(self, monkeypatch):
        monkeypatch.setenv("MCP_PORT", "9100")
        assert parse_transport_config([]).port == 9100

    def test_platform_port_env(self, monkeypatch):
        """PORT (set by hosting platforms like Smithery) is honored."""
        monkeypatch.setenv("PORT", "8081")
        assert parse_transport_config([]).port == 8081

    def test_mcp_port_beats_platform_port(self, monkeypatch):
        monkeypatch.setenv("MCP_PORT", "9100")
        monkeypatch.setenv("PORT", "8081")
        assert parse_transport_config([]).port == 9100

    def test_cli_port_beats_env(self, monkeypatch):
        monkeypatch.setenv("MCP_PORT", "9100")
        assert parse_transport_config(["--port", "7000"]).port == 7000

    def test_non_integer_port_env_rejected(self, monkeypatch):
        monkeypatch.setenv("MCP_PORT", "not-a-port")
        with pytest.raises(SystemExit):
            parse_transport_config([])

    def test_stateless_and_json_env_opt_outs(self, monkeypatch):
        monkeypatch.setenv("MCP_STATELESS_HTTP", "false")
        monkeypatch.setenv("MCP_JSON_RESPONSE", "0")
        config = parse_transport_config(["--transport", "http"])
        assert config.stateless is False
        assert config.json_response is False

    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            parse_transport_config(["--version"])
        assert excinfo.value.code == 0
        assert __version__ in capsys.readouterr().out


# ===============================================================================
# API KEY EXTRACTION
# ===============================================================================


def _http_scope(
    headers: list[tuple[bytes, bytes]] | None = None,
    query_string: bytes | None = b"",
) -> dict:
    scope: dict = {"type": "http", "query_string": query_string}
    if headers is not None:
        scope["headers"] = headers
    return scope


def _smithery_config_qs(config: object) -> bytes:
    encoded = base64.b64encode(json.dumps(config).encode()).decode()
    return f"config={encoded}".encode()


class TestExtractApiKey:
    def test_x_api_key_header(self):
        scope = _http_scope(headers=[(b"x-api-key", b" header-key ")])
        assert extract_api_key(scope) == "header-key"

    def test_other_headers_ignored(self):
        scope = _http_scope(headers=[(b"authorization", b"Bearer tok")])
        assert extract_api_key(scope) == ""

    def test_no_headers_no_query(self):
        assert extract_api_key({"type": "http"}) == ""

    def test_api_key_query_param(self):
        scope = _http_scope(query_string=b"api_key=query-key")
        assert extract_api_key(scope) == "query-key"

    def test_config_schema_query_param(self):
        """Smithery dot-notation config arrives as top-level query params."""
        scope = _http_scope(query_string=b"SEMANTIC_SCHOLAR_API_KEY=schema-key")
        assert extract_api_key(scope) == "schema-key"

    def test_header_beats_query_param(self):
        scope = _http_scope(
            headers=[(b"x-api-key", b"header-key")],
            query_string=b"api_key=query-key",
        )
        assert extract_api_key(scope) == "header-key"

    def test_blank_header_falls_back_to_query_param(self):
        """A proxy forwarding an empty `x-api-key:` must not mask other sources."""
        scope = _http_scope(
            headers=[(b"x-api-key", b"  ")],
            query_string=b"SEMANTIC_SCHOLAR_API_KEY=query-key",
        )
        assert extract_api_key(scope) == "query-key"

    def test_blank_header_and_no_other_source(self):
        scope = _http_scope(headers=[(b"x-api-key", b"")])
        assert extract_api_key(scope) == ""

    def test_schema_param_beats_reserved_api_key_param(self):
        """Smithery's gateway reserves `api_key` for the Smithery user key, so
        the schema-named param must win when both are present.
        """
        scope = _http_scope(
            query_string=b"api_key=smithery-platform-key&SEMANTIC_SCHOLAR_API_KEY=s2-key"
        )
        assert extract_api_key(scope) == "s2-key"

    def test_blank_query_param_ignored(self):
        scope = _http_scope(query_string=b"api_key=%20%20")
        assert extract_api_key(scope) == ""

    def test_smithery_base64_config(self):
        qs = _smithery_config_qs({"SEMANTIC_SCHOLAR_API_KEY": "smithery-key"})
        assert extract_api_key(_http_scope(query_string=qs)) == "smithery-key"

    def test_smithery_config_api_key_alias(self):
        qs = _smithery_config_qs({"api_key": "alias-key"})
        assert extract_api_key(_http_scope(query_string=qs)) == "alias-key"

    def test_smithery_config_without_key(self):
        qs = _smithery_config_qs({"other": "setting"})
        assert extract_api_key(_http_scope(query_string=qs)) == ""

    def test_smithery_config_not_a_dict(self):
        qs = _smithery_config_qs(["not", "a", "dict"])
        assert extract_api_key(_http_scope(query_string=qs)) == ""

    def test_smithery_config_non_string_key(self):
        qs = _smithery_config_qs({"SEMANTIC_SCHOLAR_API_KEY": 12345})
        assert extract_api_key(_http_scope(query_string=qs)) == ""

    def test_smithery_config_invalid_base64(self):
        scope = _http_scope(query_string=b"config=%21%21not-base64%21%21")
        assert extract_api_key(scope) == ""

    def test_smithery_config_invalid_json(self):
        encoded = base64.b64encode(b"{not json").decode()
        scope = _http_scope(query_string=f"config={encoded}".encode())
        assert extract_api_key(scope) == ""


# ===============================================================================
# MIDDLEWARE
# ===============================================================================


class TestRequestApiKeyMiddleware:
    @pytest.mark.asyncio
    async def test_binds_and_resets_key_for_http_scope(self):
        seen: list[str] = []

        async def inner_app(scope, receive, send):
            seen.append(_request_api_key.get())

        middleware = RequestApiKeyMiddleware(inner_app)
        scope = _http_scope(headers=[(b"x-api-key", b"mw-key")])
        await middleware(scope, None, None)

        assert seen == ["mw-key"]
        assert _request_api_key.get() == ""  # reset after the request

    @pytest.mark.asyncio
    async def test_resets_key_when_inner_app_raises(self):
        async def inner_app(scope, receive, send):
            raise RuntimeError("boom")

        middleware = RequestApiKeyMiddleware(inner_app)
        scope = _http_scope(headers=[(b"x-api-key", b"mw-key")])
        with pytest.raises(RuntimeError):
            await middleware(scope, None, None)
        assert _request_api_key.get() == ""

    @pytest.mark.asyncio
    async def test_non_http_scope_passthrough(self):
        called: list[str] = []

        async def inner_app(scope, receive, send):
            called.append(scope["type"])

        middleware = RequestApiKeyMiddleware(inner_app)
        await middleware({"type": "lifespan"}, None, None)
        assert called == ["lifespan"]


# ===============================================================================
# REQUEST-SCOPED KEY → OUTBOUND HEADERS / RATE TIER
# ===============================================================================


class TestRequestScopedKeyResolution:
    def test_get_headers_uses_request_key_without_deprecation(self, request_key, monkeypatch):
        monkeypatch.setattr(client_mod, "SEMANTIC_SCHOLAR_API_KEY", "")
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any DeprecationWarning fails the test
            headers = get_headers()
        assert headers["x-api-key"] == request_key

    def test_per_call_key_beats_request_key(self, request_key):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            headers = get_headers(api_key="explicit-key")
        assert headers["x-api-key"] == "explicit-key"

    def test_env_key_used_when_no_request_key(self, monkeypatch):
        monkeypatch.setattr(client_mod, "SEMANTIC_SCHOLAR_API_KEY", "env-key")
        assert get_headers()["x-api-key"] == "env-key"

    @respx.mock
    @pytest.mark.asyncio
    async def test_make_request_sends_request_scoped_key(self, reset_all, request_key, monkeypatch):
        monkeypatch.setattr(client_mod, "SEMANTIC_SCHOLAR_API_KEY", "")
        route = respx.get(f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search").mock(
            return_value=Response(200, json={"data": []})
        )
        await make_request("GET", "paper/search", params={"query": "q"})
        assert route.calls[0].request.headers["x-api-key"] == request_key

    @respx.mock
    @pytest.mark.asyncio
    async def test_status_reports_request_scoped_key(self, reset_all, request_key, monkeypatch):
        monkeypatch.setattr(server_mod, "SEMANTIC_SCHOLAR_API_KEY", "")
        respx.get(f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search").mock(
            return_value=Response(200, json={"data": []})
        )
        status = json.loads(await server_mod.server_status())
        assert status["api_key_configured"] is True
        assert status["rate_tier"] == "authenticated (10 req/sec)"


# ===============================================================================
# REFCOUNTED LIFESPAN
# ===============================================================================


class TestRefcountedLifespan:
    @pytest.mark.asyncio
    async def test_nested_entries_keep_client_open(self, reset_client):
        """Per-request lifespan entries must not tear down shared resources
        while an outer holder (run_http) or concurrent request is active.
        """
        async with _lifespan(mcp):
            client = await client_mod.get_client()
            async with _lifespan(mcp):  # simulates one stateless HTTP request
                assert not client.is_closed
            # Inner exit must NOT close the shared client.
            assert not client.is_closed
            assert client_mod._client is client
        # Outermost exit tears down.
        assert client.is_closed
        assert client_mod._client is None


# ===============================================================================
# main() DISPATCH AND run_http()
# ===============================================================================


class TestMainDispatch:
    def test_main_http_dispatches_to_run_http(self, monkeypatch):
        calls: list[tuple] = []
        monkeypatch.setattr(
            server_mod, "run_http", lambda server, config, lifespan: calls.append((server, config))
        )
        server_mod.main(["--transport", "http", "--port", "9009"])
        assert len(calls) == 1
        assert calls[0][0] is mcp
        assert calls[0][1].transport == "http"
        assert calls[0][1].port == 9009

    def test_main_stdio_does_not_touch_http(self, monkeypatch):
        monkeypatch.setattr(
            server_mod.mcp, "run", lambda: None
        )  # stdio path; run_http must not be hit
        monkeypatch.setattr(
            server_mod,
            "run_http",
            lambda *a, **k: pytest.fail("run_http called for stdio transport"),
        )
        server_mod.main([])


class TestRunHttp:
    def test_run_http_applies_settings_and_starts_uvicorn(self, monkeypatch):
        served: dict = {}

        def fake_uvicorn_run(app, host, port, log_level):
            served.update(app=app, host=host, port=port, log_level=log_level)

        monkeypatch.setattr("uvicorn.run", fake_uvicorn_run)
        server = FastMCP("transport-test")
        config = TransportConfig(
            transport="http",
            host="0.0.0.0",
            port=9050,
            path="/custom-mcp",
            stateless=True,
            json_response=True,
        )

        run_http(server, config, _lifespan)

        assert server.settings.host == "0.0.0.0"
        assert server.settings.port == 9050
        assert server.settings.streamable_http_path == "/custom-mcp"
        assert server.settings.stateless_http is True
        assert server.settings.json_response is True
        assert isinstance(served["app"], Starlette)
        assert served["host"] == "0.0.0.0"
        assert served["port"] == 9050


class TestHostValidation:
    @pytest.mark.asyncio
    async def test_configured_remote_host_allowed_and_unapproved_rejected(self):
        from mcp.server.transport_security import (
            TransportSecurityMiddleware,
            TransportSecuritySettings,
        )

        settings = _with_configured_hosts(
            TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"],
                allowed_origins=[],
            ),
            ("s2.example.org:*",),
        )
        security = TransportSecurityMiddleware(settings)

        allowed_request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/mcp",
                "headers": [
                    (b"host", b"s2.example.org:8080"),
                    (b"content-type", b"application/json"),
                ],
            }
        )
        rejected_request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/mcp",
                "headers": [
                    (b"host", b"evil.example:8080"),
                    (b"content-type", b"application/json"),
                ],
            }
        )

        assert await security.validate_request(allowed_request, is_post=True) is None
        rejection = await security.validate_request(rejected_request, is_post=True)
        assert rejection is not None
        assert rejection.status_code == 421

    def test_none_settings_preserves_dns_rebinding_protection(self):
        settings = _with_configured_hosts(None, ("s2.example.org:*",))

        assert settings.enable_dns_rebinding_protection is True
        assert settings.allowed_hosts == ["s2.example.org:*"]


class TestBuildHttpApp:
    @pytest.mark.asyncio
    async def test_lifespan_holds_shared_resources(self, reset_client):
        """The app lifespan enters the refcounted _lifespan once for the whole
        process, so per-request entries can never drop the count to zero
        while the server is up.
        """
        server = FastMCP("transport-test-lifespan")
        app = build_http_app(server, _lifespan)

        assert any(m.cls is RequestApiKeyMiddleware for m in app.user_middleware), (
            "API-key middleware must be installed"
        )

        depth_before = server_mod._lifespan_depth
        async with app.router.lifespan_context(app):
            assert server_mod._lifespan_depth == depth_before + 1
        assert server_mod._lifespan_depth == depth_before


# ===============================================================================
# END-TO-END: REAL UVICORN + STREAMABLE HTTP ROUND-TRIP
# ===============================================================================


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "mcp-protocol-version": "2025-06-18",
}


def _rpc(method: str, params: dict | None = None, id_: int = 1) -> dict:
    body: dict = {"jsonrpc": "2.0", "id": id_, "method": method}
    if params is not None:
        body["params"] = params
    return body


class TestStreamableHttpEndToEnd:
    @pytest.mark.asyncio
    async def test_full_round_trip(self, reset_all, monkeypatch, sample_paper):
        """Boot the real ASGI stack with uvicorn and exercise initialize,
        tools/list, and a tools/call whose per-request x-api-key must reach
        the outbound Semantic Scholar request.
        """
        import httpx
        import uvicorn

        monkeypatch.setattr(client_mod, "SEMANTIC_SCHOLAR_API_KEY", "")
        # The session manager is cached and single-use; force a fresh one with
        # this test's settings.
        monkeypatch.setattr(mcp, "_session_manager", None)
        monkeypatch.setattr(mcp.settings, "stateless_http", True)
        monkeypatch.setattr(mcp.settings, "json_response", True)
        monkeypatch.setattr(mcp.settings, "streamable_http_path", "/mcp")

        app = build_http_app(mcp, _lifespan)
        port = _free_port()
        server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        )
        serve_task = asyncio.create_task(server.serve())
        try:
            for _ in range(200):
                if server.started:
                    break
                await asyncio.sleep(0.05)
            assert server.started, "uvicorn failed to start"

            url = f"http://127.0.0.1:{port}/mcp"
            with respx.mock(assert_all_called=False) as router:
                router.route(host="127.0.0.1").pass_through()
                s2_route = router.get(f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search").mock(
                    return_value=Response(200, json={"total": 1, "data": [sample_paper]})
                )

                async with httpx.AsyncClient(timeout=30.0) as http:
                    # 1. initialize — stateless servers answer without a session.
                    resp = await http.post(
                        url,
                        json=_rpc(
                            "initialize",
                            {
                                "protocolVersion": "2025-06-18",
                                "capabilities": {},
                                "clientInfo": {"name": "e2e-test", "version": "0"},
                            },
                        ),
                        headers=MCP_HEADERS,
                    )
                    assert resp.status_code == 200
                    init = resp.json()["result"]
                    assert init["serverInfo"]["name"] == "semantic_scholar_mcp"
                    assert init["serverInfo"]["version"] == __version__

                    # 2. tools/list — no prior handshake needed in stateless mode.
                    resp = await http.post(url, json=_rpc("tools/list", id_=2), headers=MCP_HEADERS)
                    assert resp.status_code == 200
                    tools = resp.json()["result"]["tools"]
                    assert len(tools) == 14

                    # 3. tools/call with a per-request key.
                    resp = await http.post(
                        url,
                        json=_rpc(
                            "tools/call",
                            {
                                "name": "semantic_scholar_search_papers",
                                "arguments": {"params": {"query": "attention"}},
                            },
                            id_=3,
                        ),
                        headers={**MCP_HEADERS, "x-api-key": "e2e-user-key"},
                    )
                    assert resp.status_code == 200
                    result = resp.json()["result"]
                    assert result["isError"] is False
                    assert "Attention Is All You Need" in result["content"][0]["text"]

                # The remote user's key must reach the Semantic Scholar API…
                assert s2_route.calls[0].request.headers["x-api-key"] == "e2e-user-key"
                # …and the shared client must survive between stateless requests
                # (the lifespan holder keeps the refcount above zero).
                assert client_mod._client is not None
                assert not client_mod._client.is_closed
        finally:
            server.should_exit = True
            await asyncio.wait_for(serve_task, timeout=10)
