"""Contract tests for the release workflow's SBOM trust boundary.

The publish workflow separates three concerns that must never share a job:

* ``build`` holds attestation authority (OIDC + attestations) and therefore
  executes only the hash-locked release closure — no live dependency
  resolution of any kind;
* ``sbom`` resolves the wheel's runtime dependency graph — necessarily from
  the network — and is therefore stripped to ``contents: read``;
* ``attest-sbom`` holds attestation authority again and therefore installs
  and resolves nothing: it only signs the already-verified SBOM against the
  already-built wheel.

These tests seal that boundary structurally (workflow shape and permissions)
and behaviorally (the ``scripts/release_sbom.py`` binding tool, exercised
through the same CLI the workflow invokes).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "publish.yml"
SBOM_TOOL = ROOT / "scripts" / "release_sbom.py"
PUBLISHER_INSTALLER = ROOT / "scripts" / "install-mcp-publisher.sh"

# The digest-pinned mcp-publisher installer is outside this boundary's blast
# radius and must not drift under a workflow-only change. Sealed by content
# hash: any future edit has to update this constant consciously.
PUBLISHER_INSTALLER_SHA256 = "932ee908e2307e78cff047d907cb8d7f36174b69da51942ef45f161a623b4e0a"

RELEASE_REF = "${{ inputs.tag || github.ref }}"

PRIVILEGED_PERMISSIONS = {
    "contents": "read",
    "id-token": "write",
    "attestations": "write",
}


@pytest.fixture(scope="module")
def jobs() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]


def needs(job: dict[str, Any]) -> set[str]:
    """The job's ``needs`` as a set, whether it was written scalar or list."""
    declared = job.get("needs", [])
    return {declared} if isinstance(declared, str) else set(declared)


def run_text(job: dict[str, Any]) -> str:
    """Every ``run`` script in the job, concatenated."""
    return "\n".join(step.get("run", "") for step in job["steps"])


def steps_using(job: dict[str, Any], action: str) -> list[dict[str, Any]]:
    return [step for step in job["steps"] if step.get("uses", "").startswith(f"{action}@")]


# ── Workflow structure: privilege boundaries ─────────────────────────────────


def test_build_retains_provenance_permissions(jobs: dict[str, Any]) -> None:
    assert jobs["build"]["permissions"] == PRIVILEGED_PERMISSIONS


def test_build_performs_no_live_dependency_resolution(jobs: dict[str, Any]) -> None:
    """Every install in the privileged build job must be hash-locked."""
    text = run_text(jobs["build"])
    for line in text.splitlines():
        if "pip install" in line:
            assert "--require-hashes" in line, f"unlocked install in build: {line.strip()!r}"
    for forbidden in ("venv", ".whl", "cyclonedx", "--no-deps"):
        assert forbidden not in text, f"build must not contain {forbidden!r}"
    assert "--no-isolation" in text, "build must keep the offline build backend contract"


def test_build_attests_provenance_and_uploads_dist(jobs: dict[str, Any]) -> None:
    (provenance,) = steps_using(jobs["build"], "actions/attest-build-provenance")
    assert provenance["with"]["subject-path"] == "dist/*"
    (upload,) = steps_using(jobs["build"], "actions/upload-artifact")
    assert upload["with"]["name"] == "dist"


def test_build_no_longer_generates_or_attests_the_sbom(jobs: dict[str, Any]) -> None:
    assert not steps_using(jobs["build"], "actions/attest-sbom")
    assert "sbom" not in run_text(jobs["build"]).lower()


def test_sbom_permissions_are_exactly_contents_read(jobs: dict[str, Any]) -> None:
    assert jobs["sbom"]["permissions"] == {"contents": "read"}


def test_sbom_needs_build_and_checks_out_the_same_release_identity(
    jobs: dict[str, Any],
) -> None:
    assert needs(jobs["sbom"]) == {"build"}
    (build_checkout,) = steps_using(jobs["build"], "actions/checkout")
    (sbom_checkout,) = steps_using(jobs["sbom"], "actions/checkout")
    assert build_checkout["with"]["ref"] == RELEASE_REF
    assert sbom_checkout["with"]["ref"] == RELEASE_REF
    assert sbom_checkout["uses"] == build_checkout["uses"]


def test_sbom_asserts_exactly_one_wheel_before_installing(jobs: dict[str, Any]) -> None:
    text = run_text(jobs["sbom"])
    assert "wheels=(dist/*.whl)" in text
    assert '"${#wheels[@]}" -ne 1' in text


def test_sbom_performs_exact_wheel_runtime_installation(jobs: dict[str, Any]) -> None:
    """The runtime graph is resolved from the built wheel, in a clean venv."""
    text = run_text(jobs["sbom"])
    assert "python -m venv /tmp/sbom-runtime" in text
    assert "/tmp/sbom-runtime/bin/pip install --quiet dist/*.whl" in text
    assert "cyclonedx-py environment /tmp/sbom-runtime/bin/python" in text


def test_sbom_tooling_comes_from_the_locked_release_closure(jobs: dict[str, Any]) -> None:
    assert "pip install --require-hashes -r requirements-release.lock" in run_text(jobs["sbom"])


def test_sbom_binds_then_verifies_then_uploads(jobs: dict[str, Any]) -> None:
    runs = [step.get("run", "") for step in jobs["sbom"]["steps"]]
    bind_at = next(i for i, run in enumerate(runs) if "release_sbom.py bind" in run)
    verify_at = next(i for i, run in enumerate(runs) if "release_sbom.py verify" in run)
    (upload,) = steps_using(jobs["sbom"], "actions/upload-artifact")
    upload_at = jobs["sbom"]["steps"].index(upload)
    assert bind_at < verify_at < upload_at
    assert upload["with"] == {"name": "sbom", "path": "sbom.cdx.json"}


def test_attest_sbom_has_only_its_minimal_permissions(jobs: dict[str, Any]) -> None:
    assert jobs["attest-sbom"]["permissions"] == PRIVILEGED_PERMISSIONS


def test_attest_sbom_needs_build_and_sbom(jobs: dict[str, Any]) -> None:
    assert needs(jobs["attest-sbom"]) == {"build", "sbom"}


def test_attest_sbom_installs_nothing_and_resolves_nothing(jobs: dict[str, Any]) -> None:
    """The attesting job may only download artifacts and sign them."""
    job = jobs["attest-sbom"]
    assert not any("run" in step for step in job["steps"])
    uses = [step["uses"].split("@")[0] for step in job["steps"]]
    assert sorted(uses) == [
        "actions/attest-sbom",
        "actions/download-artifact",
        "actions/download-artifact",
    ]
    assert not steps_using(job, "actions/checkout")
    assert not steps_using(job, "actions/setup-python")


def test_sbom_attestation_subject_is_the_wheel_only(jobs: dict[str, Any]) -> None:
    """A wheel-derived SBOM must never be attested against the sdist."""
    attest_steps = [
        step for job in jobs.values() for step in steps_using(job, "actions/attest-sbom")
    ]
    (attest,) = attest_steps
    assert attest["with"]["subject-path"] == "dist/*.whl"
    assert attest["with"]["sbom-path"] == "sbom.cdx.json"


def test_publish_pypi_cannot_precede_the_integrity_chain(jobs: dict[str, Any]) -> None:
    assert needs(jobs["publish-pypi"]) == {"build", "sbom", "attest-sbom"}
    assert needs(jobs["publish-mcp-registry"]) == {"publish-pypi"}


def test_publish_pypi_trusted_publishing_is_preserved(jobs: dict[str, Any]) -> None:
    job = jobs["publish-pypi"]
    assert job["permissions"] == {"id-token": "write"}
    assert job["environment"]["name"] == "pypi"
    (publish,) = steps_using(job, "pypa/gh-action-pypi-publish")
    assert publish["with"]["attestations"] is True


def test_pinned_publisher_path_remains_unchanged(jobs: dict[str, Any]) -> None:
    digest = hashlib.sha256(PUBLISHER_INSTALLER.read_bytes()).hexdigest()
    assert digest == PUBLISHER_INSTALLER_SHA256
    assert "scripts/install-mcp-publisher.sh" in run_text(jobs["publish-mcp-registry"])


# ── Binding tool behavior: the SBOM must prove it describes the exact wheel ──


def make_wheel(
    dist: Path,
    name: str = "s2-mcp-server",
    version: str = "1.7.1",
    payload: bytes = b"print('release')\n",
) -> Path:
    """Write a minimal but METADATA-complete wheel into ``dist``."""
    dist.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[-_.]+", "_", name)
    wheel = dist / f"{stem}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"{stem}/__init__.py", payload)
        archive.writestr(
            f"{stem}-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        )
        archive.writestr(f"{stem}-{version}.dist-info/WHEEL", "Wheel-Version: 1.0\n")
    return wheel


def make_sbom(
    path: Path,
    name: str = "s2-mcp-server",
    version: str = "1.7.1",
    hashes: list[dict[str, str]] | None = None,
) -> None:
    root: dict[str, Any] = {"type": "application", "name": name, "version": version}
    if hashes is not None:
        root["hashes"] = hashes
    document = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {"component": root},
        "components": [],
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def sbom_tool(command: str, sbom: Path, dist: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SBOM_TOOL), command, "--sbom", str(sbom), "--dist", str(dist)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_bind_records_wheel_identity_and_exact_digest(tmp_path: Path) -> None:
    wheel = make_wheel(tmp_path / "dist")
    sbom = tmp_path / "sbom.cdx.json"
    make_sbom(sbom)

    assert sbom_tool("bind", sbom, tmp_path / "dist").returncode == 0
    root = json.loads(sbom.read_text(encoding="utf-8"))["metadata"]["component"]
    assert root["hashes"] == [
        {"alg": "SHA-256", "content": hashlib.sha256(wheel.read_bytes()).hexdigest()}
    ]
    assert sbom_tool("verify", sbom, tmp_path / "dist").returncode == 0


def test_binding_accepts_canonically_equal_names(tmp_path: Path) -> None:
    """PEP 503 canonicalization, not string equality, decides name identity."""
    make_wheel(tmp_path / "dist", name="S2_MCP.Server")
    sbom = tmp_path / "sbom.cdx.json"
    make_sbom(sbom, name="s2-mcp-server")
    assert sbom_tool("bind", sbom, tmp_path / "dist").returncode == 0
    assert sbom_tool("verify", sbom, tmp_path / "dist").returncode == 0


@pytest.mark.parametrize(
    ("sbom_name", "sbom_version"),
    [("some-other-package", "1.7.1"), ("s2-mcp-server", "9.9.9")],
    ids=["name-mismatch", "version-mismatch"],
)
def test_bind_rejects_root_identity_not_matching_wheel_metadata(
    tmp_path: Path, sbom_name: str, sbom_version: str
) -> None:
    make_wheel(tmp_path / "dist")
    sbom = tmp_path / "sbom.cdx.json"
    make_sbom(sbom, name=sbom_name, version=sbom_version)
    result = sbom_tool("bind", sbom, tmp_path / "dist")
    assert result.returncode == 1
    assert "does not match wheel METADATA" in result.stderr


def test_bind_fails_closed_on_conflicting_preexisting_digest(tmp_path: Path) -> None:
    make_wheel(tmp_path / "dist")
    sbom = tmp_path / "sbom.cdx.json"
    make_sbom(sbom, hashes=[{"alg": "SHA-256", "content": "0" * 64}])
    result = sbom_tool("bind", sbom, tmp_path / "dist")
    assert result.returncode == 1
    assert "conflicting SHA-256 evidence" in result.stderr


def test_verify_rejects_a_substituted_wheel(tmp_path: Path) -> None:
    """Same name, same version, different bytes: the binding must not hold."""
    make_wheel(tmp_path / "dist")
    sbom = tmp_path / "sbom.cdx.json"
    make_sbom(sbom)
    assert sbom_tool("bind", sbom, tmp_path / "dist").returncode == 0

    make_wheel(tmp_path / "dist", payload=b"print('tampered')\n")
    result = sbom_tool("verify", sbom, tmp_path / "dist")
    assert result.returncode == 1
    assert "does not match wheel" in result.stderr


def test_verify_fails_closed_on_multiple_sha256_evidence(tmp_path: Path) -> None:
    """Even two *agreeing* digests fail: the contract is exactly one."""
    wheel = make_wheel(tmp_path / "dist")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    sbom = tmp_path / "sbom.cdx.json"
    make_sbom(
        sbom,
        hashes=[
            {"alg": "SHA-256", "content": digest},
            {"alg": "SHA-256", "content": digest},
        ],
    )
    result = sbom_tool("verify", sbom, tmp_path / "dist")
    assert result.returncode == 1
    assert "exactly one SHA-256" in result.stderr


@pytest.mark.parametrize(
    "hashes",
    [None, [{"alg": "SHA-256", "content": "f" * 64}]],
    ids=["digest-absent", "digest-wrong"],
)
def test_verify_fails_closed_without_one_matching_digest(
    tmp_path: Path, hashes: list[dict[str, str]] | None
) -> None:
    make_wheel(tmp_path / "dist")
    sbom = tmp_path / "sbom.cdx.json"
    make_sbom(sbom, hashes=hashes)
    assert sbom_tool("verify", sbom, tmp_path / "dist").returncode == 1


@pytest.mark.parametrize("command", ["bind", "verify"])
@pytest.mark.parametrize("wheel_count", [0, 2], ids=["no-wheel", "two-wheels"])
def test_anything_but_exactly_one_wheel_fails_closed(
    tmp_path: Path, command: str, wheel_count: int
) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    for index in range(wheel_count):
        make_wheel(dist, version=f"1.7.{index}")
    sbom = tmp_path / "sbom.cdx.json"
    make_sbom(sbom)
    result = sbom_tool(command, sbom, dist)
    assert result.returncode == 1
    assert "exactly one wheel" in result.stderr
