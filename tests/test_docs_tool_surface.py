"""
Documentation tool-surface tests.

Guards against the v1.2.1 regression where the README "Tools Reference" listed
only 7 of the 14 tools that the server actually registers. These tests fail
fast in CI if a new ``@mcp.tool`` is added to the server but not documented in
the README, or vice versa.
"""

from __future__ import annotations

from pathlib import Path

from semantic_scholar_mcp.server import mcp

REPO_ROOT = Path(__file__).resolve().parent.parent


def _registered_tool_names() -> set[str]:
    return set(mcp._tool_manager._tools.keys())


def test_every_registered_tool_is_documented_in_readme() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    missing = sorted(name for name in _registered_tool_names() if name not in readme)
    assert not missing, (
        "README.md is missing entries for these registered tools: "
        f"{missing}. Add them to the 'Tools Reference' section."
    )


def test_readme_does_not_advertise_unregistered_tools() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    registered = _registered_tool_names()
    # Catch tools advertised in the "Tools Reference" headings that aren't
    # registered on the server. Only headings of the form `### N. ` `name` `
    # are considered, so unrelated occurrences of the package name
    # `semantic_scholar_mcp` in shell or import examples are not matched.
    import re

    documented = set(
        re.findall(r"^###\s*\d+\.\s*`(semantic_scholar_[a-z_]+)`", readme, re.MULTILINE)
    )
    unknown = sorted(documented - registered)
    assert not unknown, (
        "README.md 'Tools Reference' has headings for tools that are not "
        f"registered on the FastMCP server: {unknown}."
    )
    # Sanity: the heading-extraction itself must find every registered tool.
    missing_headings = sorted(registered - documented)
    assert not missing_headings, (
        "README.md is missing 'Tools Reference' headings for these "
        f"registered tools: {missing_headings}."
    )
