"""Streamable HTTP transport for the Semantic Scholar MCP server.

stdio remains the default transport; ``--transport http`` (or
``MCP_TRANSPORT=http``) serves the same FastMCP instance over the MCP
Streamable HTTP transport so remote clients — claude.ai custom connectors,
Smithery-hosted deployments, ``mcp-remote`` bridges — can connect to a URL
instead of spawning a local process.

Design decisions:

- **Stateless by default** (``stateless_http=True``). The server keeps no
  per-session state (the TTL cache and rate limiter are process-global), so
  every request can be served independently. This is what hosted platforms
  expect (tool scanning without a handshake, horizontal scaling) and it is
  what makes per-request API keys sound: each request's key is bound to a
  contextvar that is copied into that request's server task.
- **JSON responses by default** (``json_response=True``). No tool streams
  partial results or emits notifications, so plain JSON responses are
  equivalent to SSE streams while being friendlier to proxies and clients.
  Both knobs have env-var opt-outs (``MCP_STATELESS_HTTP``,
  ``MCP_JSON_RESPONSE``) for clients that want session semantics or SSE.
- **Per-request API keys**: remote users supply their own Semantic Scholar
  key via the ``x-api-key`` header, a ``SEMANTIC_SCHOLAR_API_KEY``/``api_key``
  query parameter (Smithery dot-notation session config), or the legacy
  base64 ``?config=`` parameter. The key is request-scoped (see
  :data:`semantic_scholar_mcp.client._request_api_key`); it is never logged
  and never leaks across concurrent requests.
- **Loopback bind by default** (``127.0.0.1``). Exposing the server on a
  public interface is an explicit opt-in (``--host 0.0.0.0``) because the
  HTTP transport itself performs no authentication.

Author: Santiago Maniches (ORCID: 0009-0005-6480-1987)
Organization: TOPOLOGICA LLC — https://topologica.ai
License: MIT
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from urllib.parse import parse_qs

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.types import ASGIApp, Receive, Scope, Send

from . import __version__
from .client import _request_api_key
from .logging_config import get_logger

logger = get_logger()

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_PATH = "/mcp"

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})


@dataclass(frozen=True)
class TransportConfig:
    """Resolved transport selection for one server invocation."""

    transport: str  # "stdio" | "http"
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    path: str = DEFAULT_PATH
    allowed_hosts: tuple[str, ...] = ()
    stateless: bool = True
    json_response: bool = True


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean env var; unrecognized values fall back to ``default``."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    return default


def _resolve_port(cli_port: int | None, parser: argparse.ArgumentParser) -> int:
    """Resolve the port: CLI flag, then MCP_PORT, then PORT (platform), then 8000."""
    if cli_port is not None:
        return cli_port
    for env_name in ("MCP_PORT", "PORT"):
        raw = os.environ.get(env_name)
        if raw:
            try:
                return int(raw)
            except ValueError:
                parser.error(f"environment variable {env_name} must be an integer, got {raw!r}")
    return DEFAULT_PORT


def parse_transport_config(argv: Sequence[str] | None = None) -> TransportConfig:
    """Parse CLI flags (with env-var fallbacks) into a :class:`TransportConfig`.

    Precedence per option: CLI flag > environment variable > default. The
    ``PORT`` variable (set by hosting platforms such as Smithery) is honored
    when ``--port``/``MCP_PORT`` are absent.
    """
    parser = argparse.ArgumentParser(
        prog="s2-mcp-server",
        description="Semantic Scholar MCP server (stdio by default; --transport http "
        "serves MCP Streamable HTTP for remote clients).",
        epilog="Environment fallbacks: MCP_TRANSPORT, MCP_HOST, MCP_PORT (or PORT), "
        "MCP_PATH, MCP_ALLOWED_HOSTS, MCP_STATELESS_HTTP, MCP_JSON_RESPONSE, "
        "SEMANTIC_SCHOLAR_API_KEY.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "streamable-http"],
        default=os.environ.get("MCP_TRANSPORT", "stdio"),
        help="MCP transport to serve (streamable-http is an alias for http; default: stdio)",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("MCP_HOST", DEFAULT_HOST),
        help=f"bind address for the HTTP transport (default: {DEFAULT_HOST}; "
        "use 0.0.0.0 only behind a trusted proxy)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"bind port for the HTTP transport (default: MCP_PORT, PORT, or {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--path",
        default=os.environ.get("MCP_PATH", DEFAULT_PATH),
        help=f"URL path the MCP endpoint is served on (default: {DEFAULT_PATH})",
    )
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=None,
        help=(
            "additional Host header allowed by the HTTP transport DNS-rebinding "
            "guard; repeat for multiple public hostnames/IPs (env: "
            "MCP_ALLOWED_HOSTS comma-separated)"
        ),
    )
    args = parser.parse_args(argv)

    # argparse validates `choices` only for CLI-provided values, so an env-var
    # default like MCP_TRANSPORT=bogus must be rejected explicitly.
    transport = "http" if args.transport == "streamable-http" else args.transport
    if transport not in ("stdio", "http"):
        parser.error(
            f"invalid transport {args.transport!r} (choose from stdio, http, streamable-http)"
        )

    return TransportConfig(
        transport=transport,
        host=args.host,
        port=_resolve_port(args.port, parser),
        path=args.path,
        allowed_hosts=_resolve_allowed_hosts(args.allowed_host),
        stateless=_env_flag("MCP_STATELESS_HTTP", default=True),
        json_response=_env_flag("MCP_JSON_RESPONSE", default=True),
    )


def _resolve_allowed_hosts(cli_hosts: Sequence[str] | None) -> tuple[str, ...]:
    """Resolve additional DNS-rebinding Host allow-list entries."""
    raw_hosts = (
        cli_hosts if cli_hosts is not None else os.environ.get("MCP_ALLOWED_HOSTS", "").split(",")
    )
    return tuple(host.strip() for host in raw_hosts if host.strip())


def _with_configured_hosts(
    settings: TransportSecuritySettings | None, hosts: Sequence[str]
) -> TransportSecuritySettings:
    """Return transport security settings extended with explicit public hosts."""
    base_settings = settings or TransportSecuritySettings(enable_dns_rebinding_protection=True)
    merged_hosts = list(dict.fromkeys([*base_settings.allowed_hosts, *hosts]))
    return base_settings.model_copy(update={"allowed_hosts": merged_hosts})


def extract_api_key(scope: Scope) -> str:
    """Pull a per-request Semantic Scholar API key from an ASGI scope.

    Sources, highest precedence first: the ``x-api-key`` header, a
    ``SEMANTIC_SCHOLAR_API_KEY`` query parameter (Smithery's dot-notation
    session config), an ``api_key`` query parameter, and the legacy
    base64-encoded ``?config=`` parameter. ``SEMANTIC_SCHOLAR_API_KEY`` is
    checked before ``api_key`` because Smithery's gateway reserves ``api_key``
    for the *Smithery* user key, which must not be mistaken for an S2 key.
    Returns ``""`` when no key is present.
    """
    # ASGI scope values are untyped (Any); pin the types mypy can't infer.
    headers: list[tuple[bytes, bytes]] = scope.get("headers") or []
    for name, value in headers:
        if name.lower() == b"x-api-key":
            header_key = value.decode("latin-1").strip()
            # A blank header (e.g. a proxy forwarding `x-api-key:` unset) must
            # not mask a key supplied via the query-parameter sources below.
            if header_key:
                return header_key
    query_string: bytes = scope.get("query_string") or b""
    query = parse_qs(query_string.decode("latin-1"))
    for param in ("SEMANTIC_SCHOLAR_API_KEY", "api_key"):
        values = query.get(param)
        if values and values[0].strip():
            return values[0].strip()
    return _key_from_smithery_config(query)


def _key_from_smithery_config(query: dict[str, list[str]]) -> str:
    """Decode Smithery's base64 ``?config=`` JSON and pull the API key from it."""
    values = query.get("config")
    if not values:
        return ""
    try:
        decoded = json.loads(base64.b64decode(values[0]))
    except ValueError:  # binascii.Error and JSONDecodeError are ValueErrors
        return ""
    if not isinstance(decoded, dict):
        return ""
    key = decoded.get("SEMANTIC_SCHOLAR_API_KEY") or decoded.get("api_key") or ""
    return key.strip() if isinstance(key, str) else ""


class RequestApiKeyMiddleware:
    """Bind the per-request API key to a contextvar for the request's lifetime.

    Pure ASGI (not ``BaseHTTPMiddleware``) so streamed SSE responses are not
    buffered. The contextvar is copied into the per-request MCP server task
    spawned by the stateless session manager, so the key set here is visible
    to tool handlers for exactly this request.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        token = _request_api_key.set(extract_api_key(scope))
        try:
            await self.app(scope, receive, send)
        finally:
            _request_api_key.reset(token)


def build_http_app(
    server: FastMCP,
    resource_lifespan: Callable[[FastMCP], AbstractAsyncContextManager[None]],
) -> Starlette:
    """Build the Streamable HTTP ASGI app with key middleware and held resources.

    ``resource_lifespan`` (the server's refcounted ``_lifespan``) is entered
    once for the whole process here. The MCP SDK re-enters the same lifespan
    per request in stateless mode; holding this outer reference keeps the
    shared httpx client and TTL cache alive across requests instead of being
    torn down after every response.
    """
    app = server.streamable_http_app()
    session_manager_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def _http_lifespan(started_app: Starlette) -> AsyncIterator[None]:
        async with resource_lifespan(server), session_manager_lifespan(started_app):
            yield

    app.router.lifespan_context = _http_lifespan
    app.add_middleware(RequestApiKeyMiddleware)
    return app


def run_http(
    server: FastMCP,
    config: TransportConfig,
    resource_lifespan: Callable[[FastMCP], AbstractAsyncContextManager[None]],
) -> None:
    """Serve the FastMCP instance over Streamable HTTP with uvicorn (blocking)."""
    import uvicorn

    # Settings must be applied before build_http_app(): the SDK constructs its
    # session manager lazily on first streamable_http_app() call and reads
    # stateless/json_response/path from settings at that moment.
    server.settings.host = config.host
    server.settings.port = config.port
    server.settings.streamable_http_path = config.path
    server.settings.stateless_http = config.stateless
    server.settings.json_response = config.json_response
    server.settings.transport_security = _with_configured_hosts(
        server.settings.transport_security, config.allowed_hosts
    )

    app = build_http_app(server, resource_lifespan)
    logger.info(
        "Serving MCP Streamable HTTP at http://%s:%d%s (stateless=%s, json_response=%s)",
        config.host,
        config.port,
        config.path,
        config.stateless,
        config.json_response,
    )
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PATH",
    "DEFAULT_PORT",
    "RequestApiKeyMiddleware",
    "TransportConfig",
    "_resolve_allowed_hosts",
    "_with_configured_hosts",
    "build_http_app",
    "extract_api_key",
    "parse_transport_config",
    "run_http",
]
