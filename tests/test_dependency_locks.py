"""Contract tests for the committed dependency locks.

These tests seal three properties of the lock files themselves, independently of
whatever a resolver happens to produce today:

* every requirement is pinned and hash-checked;
* the development lock differs from the authoritative base only by an approved,
  enumerated set of packages;
* the build and release closures carry the tools the workflows rely on.

The parser is marker-aware. A universal lock legitimately pins the same package
more than once under disjoint environment markers (``rpds-py`` is pinned
separately for Python < 3.11 and >= 3.11), so blocks are keyed by canonical name
and compared as marker-tagged groups rather than assumed unique.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import NamedTuple

import pytest

ROOT = Path(__file__).resolve().parent.parent

AUTHORITATIVE_BASE = "5ab7a36e52828f726bec764bbbfb2a881b311273"
CUTOFF = "2026-08-03T15:29:41Z"

DEV_LOCK = ROOT / "requirements-dev.lock"
BUILD_LOCK = ROOT / "requirements-build.lock"
RELEASE_LOCK = ROOT / "requirements-release.lock"
REGENERATE_SCRIPT = ROOT / "scripts" / "regenerate-locks.sh"

# Packages whose development-lock block may change relative to the authoritative
# base, and the only way each is allowed to change.
SEMANTIC_CHANGES = frozenset(
    {
        "backports-asyncio-runner",
        "cryptography",
        "exceptiongroup",
        "pip",
        "rpds-py",
    }
)

# Packages the development lock must pin at an exact version. cryptography is
# held at 50.0.0 because 49.0.0 (the version the authoritative-base seed would
# otherwise carry forward) is affected by CVE-2026-69247; pip is held at 26.2
# because the seeded 26.1.2 is affected by PYSEC-2026-3721. Each pin is produced
# by an --upgrade-package instruction in scripts/regenerate-locks.sh, not by hand.
REQUIRED_DEV_VERSIONS = {
    "cryptography": "50.0.0",
    "pip": "26.2",
}

PROVENANCE_ONLY_CHANGES: dict[str, tuple[str, ...]] = {
    "tomli": (
        "    #   bandit\n",
        "    #   mypy\n",
        "    #   pytest\n",
    ),
    "typing-extensions": (
        "    #   cryptography\n",
        "    #   exceptiongroup\n",
        "    #   pyjwt\n",
        "    #   uvicorn\n",
        "    #   virtualenv\n",
    ),
}

EXPECTED_CHANGED_PACKAGES = SEMANTIC_CHANGES | set(PROVENANCE_ONLY_CHANGES)

# The exact uv command line each lock header must record. uv echoes its own
# arguments verbatim, so this pins the resolver contract into the artifact.
EXPECTED_HEADERS = {
    DEV_LOCK: (
        "uv --no-config pip compile pyproject.toml --extra dev "
        "--python-version 3.10 --universal --generate-hashes "
        f"--exclude-newer {CUTOFF} --output-file requirements-dev.lock"
    ),
    BUILD_LOCK: (
        "uv --no-config pip compile requirements-build.in "
        "--python-version 3.10 --universal --generate-hashes "
        f"--exclude-newer {CUTOFF} --output-file requirements-build.lock"
    ),
    RELEASE_LOCK: (
        "uv --no-config pip compile requirements-release.in "
        "--python-version 3.10 --universal --generate-hashes "
        f"--exclude-newer {CUTOFF} --constraint requirements-build.lock "
        "--output-file requirements-release.lock"
    ),
}

REQUIREMENT_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)\s*(?:;\s*(.*?))?\s*(?:\\)?$")


class Requirement(NamedTuple):
    """One pinned requirement: its identity, its marker, and its raw block."""

    name: str
    version: str
    marker: str
    block: str

    @property
    def hashes(self) -> list[str]:
        return [
            line.strip()
            for line in self.block.splitlines()
            if line.lstrip().startswith("--hash=sha256:")
        ]

    @property
    def provenance(self) -> list[str]:
        return [
            line for line in self.block.splitlines(keepends=True) if line.lstrip().startswith("#")
        ]


def canonical(name: str) -> str:
    """Canonicalize a distribution name per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_lock(text: str) -> dict[str, list[Requirement]]:
    """Parse a lock into canonical-name -> list of marker-tagged requirements."""
    lines = text.splitlines(keepends=True)
    starts = [index for index, line in enumerate(lines) if REQUIREMENT_RE.match(line.rstrip("\n"))]

    parsed: dict[str, list[Requirement]] = {}
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        match = REQUIREMENT_RE.match(lines[start].rstrip("\n"))
        assert match is not None
        name = canonical(match.group(1))
        parsed.setdefault(name, []).append(
            Requirement(
                name=name,
                version=match.group(2),
                marker=(match.group(3) or "").strip(),
                block="".join(lines[start:end]),
            )
        )
    return parsed


def authoritative_dev_lock() -> str | None:
    """Return the base development lock, or None when history is unavailable.

    Shallow CI checkouts do not carry the base blob. The structural gates in
    this module stand alone; only the base-relative comparisons need history.
    """
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{AUTHORITATIVE_BASE}:requirements-dev.lock"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def require_base() -> str:
    base = authoritative_dev_lock()
    if base is None:
        pytest.skip(f"authoritative base {AUTHORITATIVE_BASE[:12]} not present in this checkout")
    return base


@pytest.fixture(scope="module")
def dev_lock() -> dict[str, list[Requirement]]:
    return parse_lock(DEV_LOCK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def build_lock() -> dict[str, list[Requirement]]:
    return parse_lock(BUILD_LOCK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def release_lock() -> dict[str, list[Requirement]]:
    return parse_lock(RELEASE_LOCK.read_text(encoding="utf-8"))


@pytest.mark.parametrize("lock", [DEV_LOCK, BUILD_LOCK, RELEASE_LOCK], ids=lambda p: p.name)
def test_lock_exists_and_every_requirement_is_hash_pinned(lock: Path) -> None:
    assert lock.is_file(), f"{lock.name} is missing"
    parsed = parse_lock(lock.read_text(encoding="utf-8"))
    assert parsed, f"{lock.name} pins nothing"
    for requirements in parsed.values():
        for requirement in requirements:
            assert requirement.hashes, f"{lock.name}: {requirement.name} has no --hash entries"


@pytest.mark.parametrize("lock", [DEV_LOCK, BUILD_LOCK, RELEASE_LOCK], ids=lambda p: p.name)
def test_lock_header_records_the_generation_contract(lock: Path) -> None:
    header = lock.read_text(encoding="utf-8").splitlines()[:2]
    assert header[0].startswith("# This file was autogenerated by uv")
    assert header[1] == f"#    {EXPECTED_HEADERS[lock]}"


@pytest.mark.parametrize("lock", [DEV_LOCK, BUILD_LOCK, RELEASE_LOCK], ids=lambda p: p.name)
def test_duplicate_pins_are_separated_by_markers(lock: Path) -> None:
    """A package may be pinned more than once only under distinct markers."""
    for name, requirements in parse_lock(lock.read_text(encoding="utf-8")).items():
        if len(requirements) == 1:
            continue
        markers = [requirement.marker for requirement in requirements]
        assert all(markers), f"{lock.name}: {name} pinned {len(markers)}x without markers"
        assert len(set(markers)) == len(markers), f"{lock.name}: {name} repeats a marker"


def test_universal_dev_lock_pins_rpds_py_under_disjoint_markers(
    dev_lock: dict[str, list[Requirement]],
) -> None:
    """The duplicate this contract exists to tolerate, asserted explicitly."""
    requirements = dev_lock["rpds-py"]
    assert len(requirements) == 2
    assert {requirement.marker for requirement in requirements} == {
        "python_full_version < '3.11'",
        "python_full_version >= '3.11'",
    }


def test_dev_lock_changed_package_set_is_exactly_the_approved_set() -> None:
    base = parse_lock(require_base())
    current = parse_lock(DEV_LOCK.read_text(encoding="utf-8"))

    changed = {
        name
        for name in set(base) | set(current)
        if [r.block for r in base.get(name, [])] != [r.block for r in current.get(name, [])]
    }
    assert changed == EXPECTED_CHANGED_PACKAGES


def test_every_other_dev_package_is_byte_for_byte_unchanged() -> None:
    base = parse_lock(require_base())
    current = parse_lock(DEV_LOCK.read_text(encoding="utf-8"))

    for name in sorted((set(base) | set(current)) - EXPECTED_CHANGED_PACKAGES):
        assert [r.block for r in base.get(name, [])] == [r.block for r in current.get(name, [])], (
            f"{name} changed but is not an approved change"
        )


@pytest.mark.parametrize("name", sorted(PROVENANCE_ONLY_CHANGES))
def test_provenance_only_packages_change_by_added_comments_alone(name: str) -> None:
    base = parse_lock(require_base())
    current = parse_lock(DEV_LOCK.read_text(encoding="utf-8"))

    assert len(base[name]) == 1
    assert len(current[name]) == 1
    before, after = base[name][0], current[name][0]

    assert after.version == before.version, f"{name}: version changed"
    assert after.marker == before.marker, f"{name}: marker changed"
    assert after.hashes == before.hashes, f"{name}: ordered hash set changed"

    before_lines = before.block.splitlines(keepends=True)
    after_lines = after.block.splitlines(keepends=True)
    assert after_lines[0] == before_lines[0], f"{name}: requirement line changed"

    residual = list(after_lines)
    for approved in PROVENANCE_ONLY_CHANGES[name]:
        assert residual.count(approved) == before_lines.count(approved) + 1, (
            f"{name}: {approved.strip()!r} was not added exactly once"
        )
        residual.remove(approved)

    assert residual == before_lines, (
        f"{name}: removing the approved provenance lines does not restore the base block"
    )


@pytest.mark.parametrize("name", sorted(REQUIRED_DEV_VERSIONS))
def test_dev_lock_pins_required_security_versions(
    dev_lock: dict[str, list[Requirement]], name: str
) -> None:
    """A security-critical pin must not drift back to a vulnerable version."""
    requirements = dev_lock[name]
    assert len(requirements) == 1, f"{name}: expected exactly one pin"
    assert requirements[0].version == REQUIRED_DEV_VERSIONS[name]


def test_regeneration_script_selects_the_required_security_versions() -> None:
    """The pin must be reproducible, not hand-applied to the generated lock."""
    script = REGENERATE_SCRIPT.read_text(encoding="utf-8")
    for name in REQUIRED_DEV_VERSIONS:
        assert f"--upgrade-package {name}" in script, (
            f"{name} is pinned in the lock but nothing in the regeneration "
            f"script selects it; --check would resolve it back to the seed"
        )


def test_build_and_release_locks_pin_editables(
    build_lock: dict[str, list[Requirement]],
    release_lock: dict[str, list[Requirement]],
) -> None:
    """editables backs PEP 660 editable installs under --no-build-isolation."""
    assert "editables" in build_lock
    assert "editables" in release_lock


def test_release_lock_pins_cyclonedx_bom(release_lock: dict[str, list[Requirement]]) -> None:
    """The SBOM tool must come from the locked release closure, never live."""
    assert "cyclonedx-bom" in release_lock


def test_release_closure_preserves_the_build_closure_versions(
    build_lock: dict[str, list[Requirement]],
    release_lock: dict[str, list[Requirement]],
) -> None:
    """The release lock is constrained by the build lock, so versions must agree."""
    for name, requirements in build_lock.items():
        assert name in release_lock, f"{name} present in build closure but missing from release"
        build_versions = {requirement.version for requirement in requirements}
        release_versions = {requirement.version for requirement in release_lock[name]}
        assert build_versions == release_versions, f"{name}: build/release version disagreement"


def test_no_bootstrap_workflow_or_constraints_file_exists() -> None:
    """Locks are generated by a maintainer, never by a self-pushing workflow."""
    assert not (ROOT / ".github" / "workflows" / "lock-bootstrap.yml").exists()
    assert not (ROOT / "requirements-dev.constraints").exists()


def test_regeneration_script_encodes_the_deterministic_contract() -> None:
    script = REGENERATE_SCRIPT.read_text(encoding="utf-8")

    assert f"BASE={AUTHORITATIVE_BASE}" in script
    assert f"CUTOFF={CUTOFF}" in script
    assert '"uv 0.11.29"|"uv 0.11.29 "*' in script, "version gate must accept build metadata"
    assert 'git show "$BASE:requirements-dev.lock"' in script, "dev lock must be seeded from git"
    assert 'test ! -e "$WORK/requirements-build.lock"' in script
    assert 'test ! -e "$WORK/requirements-release.lock"' in script
    assert "--constraint requirements-build.lock" in script
    assert "cmp -s" in script, "committed locks must be compared byte-for-byte"
    assert "diff -u" in script, "drift must be printed as a diff"
    # A bare --upgrade would re-resolve every pin and defeat the seed. Targeted
    # --upgrade-package exemptions are allowed and are asserted separately.
    assert re.search(r"--upgrade(?!-package)", script) is None, (
        "regeneration must never blanket-upgrade pins"
    )


def test_no_changed_workflow_can_write_to_the_repository() -> None:
    """The workflows this change touches must stay read-only toward git."""
    forbidden = (
        "contents: write",
        "git commit",
        "git push",
        "update-ref",
        "checkout -B",
        "switch -C",
        "--force",
    )
    for name in (
        "ci.yml",
        "dependency-audit.yml",
        "publish.yml",
        "test-api-compat.yml",
    ):
        text = (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, f"{name} must not contain {needle!r}"
