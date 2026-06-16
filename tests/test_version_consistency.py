"""
Version-consistency tests.

Guards against the v1.2.1 regression where ``server.json`` shipped with version
``1.0.0`` while ``pyproject.toml``, the runtime ``__version__``, and the PyPI
artifact were at ``1.2.1``. These tests fail fast in CI if any version source
diverges from the runtime ``__version__``, and assert the README install
command points at the correct PyPI distribution name.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from semantic_scholar_mcp import __version__

REPO_ROOT = Path(__file__).resolve().parent.parent


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_pyproject_version_matches_runtime() -> None:
    text = _read("pyproject.toml")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match is not None, "pyproject.toml has no [project] version"
    assert match.group(1) == __version__, (
        f"pyproject.toml version {match.group(1)!r} != runtime __version__ {__version__!r}"
    )


def test_server_json_version_matches_runtime() -> None:
    data = json.loads(_read("server.json"))
    assert data["version"] == __version__, (
        f"server.json top-level version {data['version']!r} != runtime {__version__!r}"
    )
    assert data["packages"], "server.json has no packages[] entries"
    pkg_version = data["packages"][0]["version"]
    assert pkg_version == __version__, (
        f"server.json packages[0].version {pkg_version!r} != runtime {__version__!r}"
    )


def test_readme_install_command_uses_pypi_distribution_name() -> None:
    # The PyPI distribution name is `s2-mcp-server`; the GitHub repo slug
    # `semantic-scholar-mcp` is held by a different author on PyPI. The README
    # install instructions must point users at the correct package.
    readme = _read("README.md")
    assert "pip install s2-mcp-server" in readme, (
        "README must instruct `pip install s2-mcp-server`, not the GitHub slug"
    )
    assert "uvx s2-mcp-server" in readme, (
        "README must instruct `uvx s2-mcp-server` for the recommended path"
    )
    assert "pip install semantic-scholar-mcp" not in readme, (
        "README must not direct users to the conflicting PyPI package "
        "`semantic-scholar-mcp` (held by a different author)"
    )


def test_citation_cff_version_matches_runtime() -> None:
    text = _read("CITATION.cff")
    match = re.search(r"^version:\s*['\"]?([^'\"\s#]+)", text, re.MULTILINE)
    assert match is not None, "CITATION.cff has no top-level version field"
    assert match.group(1) == __version__, (
        f"CITATION.cff version {match.group(1)!r} != runtime {__version__!r}"
    )


def test_zenodo_json_version_matches_runtime() -> None:
    data = json.loads(_read(".zenodo.json"))
    assert data["version"] == __version__, (
        f".zenodo.json version {data['version']!r} != runtime {__version__!r}"
    )


def test_release_please_tracks_every_versioned_file() -> None:
    """Defense-in-depth: every file the version-consistency suite checks must
    be tracked by release-please. If you add a new version-bearing file, add
    it here AND to release-please-config.json. This keeps releases automatic
    instead of relying on humans to remember to bump N files in lockstep.
    """
    config = json.loads(_read("release-please-config.json"))
    extras = config["packages"]["."]["extra-files"]

    string_paths: set[str] = {e for e in extras if isinstance(e, str)}
    json_paths: set[tuple[str, str]] = {
        (e["path"], e["jsonpath"])
        for e in extras
        if isinstance(e, dict) and e.get("type") == "json"
    }

    # Files updated via the `# x-release-please-version` annotation
    assert "CITATION.cff" in string_paths

    # Files updated via JSON path replacement
    assert ("server.json", "$.version") in json_paths
    assert ("server.json", "$.packages[0].version") in json_paths
    assert (".zenodo.json", "$.version") in json_paths

    # pyproject.toml is implicit (release-type: python handles it). The runtime
    # __version__ is derived from importlib.metadata.version("s2-mcp-server"),
    # which reads pyproject.toml — so bumping pyproject is sufficient for the
    # runtime constant; server.py no longer carries a hardcoded version string.
    assert config["release-type"] == "python"


def test_mcp_handshake_advertises_runtime_version() -> None:
    """The MCP ``initialize`` handshake must report our package version, not the
    bundled SDK's version. Regression guard for ``serverInfo.version`` silently
    defaulting to the FastMCP SDK version.
    """
    from semantic_scholar_mcp.server import mcp

    advertised = mcp._mcp_server.version
    assert advertised == __version__, (
        f"MCP serverInfo version {advertised!r} != runtime __version__ {__version__!r}"
    )


def test_security_md_supported_version_tracks_runtime() -> None:
    """SECURITY.md's supported-versions table must track the current minor line.

    Guards the drift surfaced by the 2026-06-15 quality audit, where the table
    still advertised ``1.3.x`` support while the package had moved to ``1.5.0``.
    The table may list one or more supported ``X.Y.x`` minors; it must include
    the current ``__version__`` minor, and the ``< X.Y | No`` boundary row must
    name the lowest still-supported minor.
    """
    text = _read("SECURITY.md")
    # Rows of the form `| 1.5.x | Yes |` (the `< 1.5 | No` boundary row has no `.x`).
    rows = re.findall(r"\|\s*(\d+\.\d+)\.x\s*\|\s*(Yes|No)\s*\|", text)
    supported = {ver for ver, flag in rows if flag == "Yes"}
    assert supported, "SECURITY.md has no `X.Y.x | Yes` supported-version row"
    current_minor = ".".join(__version__.split(".")[:2])
    assert current_minor in supported, (
        f"SECURITY.md supported minors {sorted(supported)} omit the current minor "
        f"{current_minor!r} (from __version__ {__version__!r}) — update the "
        f"Supported Versions table in SECURITY.md (see CONTRIBUTING.md 'Releasing')."
    )
    # The `< X.Y | No` boundary row must name the lowest still-supported minor,
    # so a half-update (supported row bumped, boundary left stale) is caught too.
    boundary = set(re.findall(r"\|\s*<\s*(\d+\.\d+)\s*\|\s*No\s*\|", text))
    lowest_supported = min(supported, key=lambda m: tuple(map(int, m.split("."))))
    assert boundary == {lowest_supported}, (
        f"SECURITY.md boundary row should read `< {lowest_supported} | No` (the "
        f"lowest supported minor); found {sorted(boundary)}. Update BOTH rows of "
        f"the Supported Versions table."
    )
