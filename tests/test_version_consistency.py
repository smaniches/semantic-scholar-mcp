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
