"""
Documentation tool-surface tests.

Guards against the v1.2.1 regression where the README "Tools Reference" listed
only 7 of the 14 tools that the server actually registers. These tests fail
fast in CI if a new ``@mcp.tool`` is added to the server but not documented in
the README, or vice versa.
"""

from __future__ import annotations

from pathlib import Path

from semantic_scholar_mcp.server import _get_accepted_params, mcp

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


def test_every_tool_parameter_is_documented_in_readme() -> None:
    """Each tool's input-model fields must appear in the README.

    Extends the tool-name guard to the parameter level: a v1.3.x audit found
    ``from_pool`` (on ``semantic_scholar_recommendations``) wired up and tested
    but absent from the README parameter tables. Parameters are documented as
    backtick code-spans, so this checks for ``\\`field\\```.
    """
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    tool_manager = mcp._tool_manager
    undocumented: dict[str, list[str]] = {}
    for name in _registered_tool_names():
        for param in _get_accepted_params(name, tool_manager):
            if f"`{param}`" not in readme:
                undocumented.setdefault(name, []).append(param)
    assert not undocumented, (
        "README.md 'Tools Reference' is missing parameter documentation for "
        f"{undocumented}. Document each parameter as a `code-span`."
    )
