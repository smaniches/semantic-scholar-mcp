"""Contract tests for the release workflow's SBOM trust boundary.

The publish workflow separates three concerns that must never share a job:

* ``build`` holds attestation authority (OIDC + attestations) and therefore
  executes only the hash-locked release closure — no live dependency
  resolution of any kind;
* ``sbom`` resolves the wheel's runtime dependency graph — necessarily from
  the network — and is therefore stripped to ``contents: read``. The graph
  is data, never code: wheels are downloaded with ``--only-binary=:all:``
  (source distributions fail closed, so no build backend runs), nothing from
  the wheelhouse is installed or imported, and no interpreter populated with
  dependency code is ever started (no ``.pth``/``sitecustomize`` hook can
  run). The document is derived from static wheel METADATA only;
* ``attest-sbom`` holds attestation authority again and therefore installs
  and resolves nothing: it only signs the already-verified SBOM against the
  already-built wheel.

These tests seal that boundary structurally (workflow shape and permissions)
and behaviorally (the ``scripts/release_sbom.py`` tool and the wheels-only
resolution shape, exercised through the same CLIs the workflow invokes).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
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


def test_sbom_asserts_exactly_one_wheel_before_resolving(jobs: dict[str, Any]) -> None:
    text = run_text(jobs["sbom"])
    assert "wheels=(dist/*.whl)" in text
    assert '"${#wheels[@]}" -ne 1' in text


def test_sbom_resolves_wheels_only_and_never_executes_them(jobs: dict[str, Any]) -> None:
    """The dependency graph is downloaded as data; no resolved code may run."""
    text = run_text(jobs["sbom"])
    assert (
        "python -m pip download --only-binary=:all: --dest /tmp/sbom-wheelhouse dist/*.whl"
    ) in text
    for line in text.splitlines():
        if "pip install" in line:
            assert "--require-hashes" in line, f"unlocked install in sbom: {line.strip()!r}"
    for forbidden in ("pip install dist", "venv", "cyclonedx-py environment"):
        assert forbidden not in text, f"sbom must not contain {forbidden!r}"


def test_sbom_document_is_generated_from_the_static_manifest(jobs: dict[str, Any]) -> None:
    """CycloneDX renders the metadata-derived manifest, never an interpreter."""
    text = run_text(jobs["sbom"])
    assert (
        "python scripts/release_sbom.py lock --dist dist "
        "--wheelhouse /tmp/sbom-wheelhouse --output sbom-requirements.txt"
    ) in text
    assert "cyclonedx-py requirements sbom-requirements.txt" in text
    everywhere = "\n".join(run_text(job) for job in jobs.values())
    assert "cyclonedx-py environment" not in everywhere


def test_sbom_tooling_comes_from_the_locked_release_closure(jobs: dict[str, Any]) -> None:
    assert "pip install --require-hashes -r requirements-release.lock" in run_text(jobs["sbom"])


def test_sbom_pipeline_orders_resolve_lock_generate_bind_verify_upload(
    jobs: dict[str, Any],
) -> None:
    runs = [step.get("run", "") for step in jobs["sbom"]["steps"]]

    def at(needle: str) -> int:
        return next(i for i, run in enumerate(runs) if needle in run)

    resolve_at = at("pip download --only-binary=:all:")
    lock_at = at("release_sbom.py lock")
    generate_at = at("cyclonedx-py requirements")
    bind_at = at("release_sbom.py bind")
    verify_at = at("release_sbom.py verify")
    (upload,) = steps_using(jobs["sbom"], "actions/upload-artifact")
    upload_at = jobs["sbom"]["steps"].index(upload)
    assert resolve_at < lock_at < generate_at < bind_at < verify_at < upload_at
    assert "--requirements sbom-requirements.txt" in runs[verify_at]
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
    requires: tuple[str, ...] = (),
    extra_files: dict[str, str] | None = None,
) -> Path:
    """Write a minimal but METADATA-complete wheel into ``dist``."""
    dist.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[-_.]+", "_", name)
    wheel = dist / f"{stem}-{version}-py3-none-any.whl"
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    metadata += "".join(f"Requires-Dist: {requirement}\n" for requirement in requires)
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"{stem}/__init__.py", payload)
        for member, content in (extra_files or {}).items():
            archive.writestr(member, content)
        archive.writestr(f"{stem}-{version}.dist-info/METADATA", metadata)
        archive.writestr(f"{stem}-{version}.dist-info/WHEEL", "Wheel-Version: 1.0\n")
    return wheel


def make_sdist(directory: Path, sentinel: Path, name: str = "evil", version: str = "1.0") -> Path:
    """A source distribution whose build backend would betray any execution."""
    directory.mkdir(parents=True, exist_ok=True)
    sdist = directory / f"{name}-{version}.tar.gz"
    members = {
        f"{name}-{version}/setup.py": f"open({str(sentinel)!r}, 'w').write('executed')\n",
        f"{name}-{version}/PKG-INFO": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
    }
    with tarfile.open(sdist, "w:gz") as archive:
        for member, content in members.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(member)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return sdist


def make_sbom(
    path: Path,
    name: str = "s2-mcp-server",
    version: str = "1.7.1",
    hashes: list[dict[str, str]] | None = None,
    components: list[dict[str, Any]] | None = None,
) -> None:
    root: dict[str, Any] = {"type": "application", "name": name, "version": version}
    if hashes is not None:
        root["hashes"] = hashes
    document = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {"component": root},
        "components": components or [],
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def component(
    name: str, version: str, digest: str | None, via_external_ref: bool = False
) -> dict[str, Any]:
    """A CycloneDX component carrying its digest the way cyclonedx-py does."""
    entry: dict[str, Any] = {"type": "library", "name": name, "version": version}
    if digest is not None:
        hashes = [{"alg": "SHA-256", "content": digest}]
        if via_external_ref:
            entry["externalReferences"] = [
                {"type": "distribution", "url": "https://pypi.org/simple/", "hashes": hashes}
            ]
        else:
            entry["hashes"] = hashes
    return entry


def sbom_tool(
    command: str, sbom: Path, dist: Path, requirements: Path | None = None
) -> subprocess.CompletedProcess[str]:
    argv = [sys.executable, str(SBOM_TOOL), command, "--sbom", str(sbom), "--dist", str(dist)]
    if requirements is not None:
        argv += ["--requirements", str(requirements)]
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def lock_tool(dist: Path, wheelhouse: Path, output: Path) -> subprocess.CompletedProcess[str]:
    argv = [sys.executable, str(SBOM_TOOL), "lock", "--dist", str(dist)]
    argv += ["--wheelhouse", str(wheelhouse), "--output", str(output)]
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def pip_download(
    root_wheel: Path, find_links: Path, dest: Path
) -> subprocess.CompletedProcess[str]:
    """The workflow's wheels-only resolution shape, against a local index."""
    argv = [sys.executable, "-m", "pip", "download", "--only-binary=:all:"]
    argv += ["--no-index", "--find-links", str(find_links), "--dest", str(dest), str(root_wheel)]
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
    )


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


# ── Non-execution: resolving and describing the graph runs none of its code ──


def test_wheels_only_resolution_never_executes_dependency_code(tmp_path: Path) -> None:
    """A dependency armed with import/.pth/sitecustomize hooks stays inert."""
    sentinel = tmp_path / "executed"
    betrayal = f"open({str(sentinel)!r}, 'w').write('executed')\n"
    deps = tmp_path / "deps"
    evil = make_wheel(
        deps,
        name="evil",
        version="1.0",
        payload=betrayal.encode("utf-8"),
        extra_files={"evil_hook.pth": "import evil\n", "sitecustomize.py": betrayal},
    )
    dist = tmp_path / "dist"
    release = make_wheel(dist, requires=("evil",))

    wheelhouse = tmp_path / "wheelhouse"
    result = pip_download(release, deps, wheelhouse)
    assert result.returncode == 0, result.stderr
    assert (wheelhouse / evil.name).is_file()

    manifest = tmp_path / "sbom-requirements.txt"
    assert lock_tool(dist, wheelhouse, manifest).returncode == 0
    assert manifest.read_text(encoding="utf-8") == (f"evil==1.0 --hash=sha256:{sha256_of(evil)}\n")
    assert not sentinel.exists(), "dependency code executed during SBOM construction"


def test_sdist_only_dependency_fails_closed_without_building(tmp_path: Path) -> None:
    """--only-binary=:all: must refuse the sdist, not run its build backend."""
    sentinel = tmp_path / "executed"
    deps = tmp_path / "deps"
    make_sdist(deps, sentinel)
    dist = tmp_path / "dist"
    release = make_wheel(dist, requires=("evil",))

    result = pip_download(release, deps, tmp_path / "wheelhouse")
    assert result.returncode != 0
    assert not sentinel.exists(), "sdist build backend executed"


# ── lock: the manifest mirrors static wheel metadata, fail-closed ────────────


def test_lock_derives_manifest_and_excludes_the_release_wheel(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    release = make_wheel(dist)
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    (wheelhouse / release.name).write_bytes(release.read_bytes())
    beta = make_wheel(wheelhouse, name="beta", version="2.0")
    alpha = make_wheel(wheelhouse, name="Alpha_pkg", version="1.0")

    manifest = tmp_path / "sbom-requirements.txt"
    assert lock_tool(dist, wheelhouse, manifest).returncode == 0
    assert manifest.read_text(encoding="utf-8").splitlines() == [
        f"Alpha_pkg==1.0 --hash=sha256:{sha256_of(alpha)}",
        f"beta==2.0 --hash=sha256:{sha256_of(beta)}",
    ]


def test_lock_fails_closed_on_a_source_distribution_in_the_wheelhouse(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    release = make_wheel(dist)
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    (wheelhouse / release.name).write_bytes(release.read_bytes())
    make_sdist(wheelhouse, tmp_path / "executed")

    result = lock_tool(dist, wheelhouse, tmp_path / "out.txt")
    assert result.returncode == 1
    assert "non-wheel" in result.stderr


def test_lock_fails_closed_without_the_release_wheel_in_the_wheelhouse(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    make_wheel(dist)
    wheelhouse = tmp_path / "wheelhouse"
    make_wheel(wheelhouse, name="beta", version="2.0")

    result = lock_tool(dist, wheelhouse, tmp_path / "out.txt")
    assert result.returncode == 1
    assert "exactly once" in result.stderr


def test_lock_fails_closed_on_release_identity_reuse(tmp_path: Path) -> None:
    """A wheel claiming the release identity with different bytes must fail."""
    dist = tmp_path / "dist"
    release = make_wheel(dist)
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    (wheelhouse / release.name).write_bytes(release.read_bytes())
    impostor = make_wheel(tmp_path / "elsewhere", payload=b"print('impostor')\n")
    (wheelhouse / "s2_mcp_server-1.7.1-py2.py3-none-any.whl").write_bytes(impostor.read_bytes())

    result = lock_tool(dist, wheelhouse, tmp_path / "out.txt")
    assert result.returncode == 1
    assert "reuses the release wheel identity" in result.stderr


def test_lock_fails_closed_on_duplicate_dependency_pins(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    release = make_wheel(dist)
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    (wheelhouse / release.name).write_bytes(release.read_bytes())
    make_wheel(wheelhouse, name="evil", version="1.0")
    make_wheel(wheelhouse, name="evil", version="2.0")

    result = lock_tool(dist, wheelhouse, tmp_path / "out.txt")
    assert result.returncode == 1
    assert "more than once" in result.stderr


# ── verify --requirements: components must mirror the manifest exactly ───────

ALPHA_DIGEST = "a" * 64
BETA_DIGEST = "b" * 64

GOOD_COMPONENTS = [
    component("alpha", "1.0", ALPHA_DIGEST),
    component("beta", "2.0", BETA_DIGEST, via_external_ref=True),
]


def cross_check_setup(tmp_path: Path, components: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    dist = tmp_path / "dist"
    make_wheel(dist)
    manifest = tmp_path / "sbom-requirements.txt"
    manifest.write_text(
        f"alpha==1.0 --hash=sha256:{ALPHA_DIGEST}\nbeta==2.0 --hash=sha256:{BETA_DIGEST}\n",
        encoding="utf-8",
    )
    sbom = tmp_path / "sbom.cdx.json"
    make_sbom(sbom, components=components)
    assert sbom_tool("bind", sbom, dist).returncode == 0
    return sbom, dist, manifest


def test_verify_accepts_components_matching_the_manifest(tmp_path: Path) -> None:
    """Both digest carriers cyclonedx-py uses (hashes, externalReferences) pass."""
    sbom, dist, manifest = cross_check_setup(tmp_path, GOOD_COMPONENTS)
    result = sbom_tool("verify", sbom, dist, manifest)
    assert result.returncode == 0, result.stderr
    assert "2 manifest components" in result.stdout


@pytest.mark.parametrize(
    ("components", "message"),
    [
        (GOOD_COMPONENTS[:1], "do not match the manifest"),
        ([*GOOD_COMPONENTS, component("gamma", "3.0", "c" * 64)], "do not match the manifest"),
        ([component("alpha", "1.1", ALPHA_DIGEST), GOOD_COMPONENTS[1]], "does not match"),
        ([component("alpha", "1.0", "d" * 64), GOOD_COMPONENTS[1]], "SHA-256 evidence"),
        ([component("alpha", "1.0", None), GOOD_COMPONENTS[1]], "SHA-256 evidence"),
        ([*GOOD_COMPONENTS, GOOD_COMPONENTS[0]], "more than once"),
        (
            [*GOOD_COMPONENTS, component("s2-mcp-server", "1.7.1", "e" * 64)],
            "must not appear as a component",
        ),
    ],
    ids=[
        "component-missing",
        "component-extra",
        "version-mismatch",
        "digest-conflict",
        "digest-absent",
        "duplicate-component",
        "release-listed-as-component",
    ],
)
def test_verify_fails_closed_on_manifest_divergence(
    tmp_path: Path, components: list[dict[str, Any]], message: str
) -> None:
    sbom, dist, manifest = cross_check_setup(tmp_path, components)
    result = sbom_tool("verify", sbom, dist, manifest)
    assert result.returncode == 1
    assert message in result.stderr


def test_verify_fails_closed_on_a_malformed_manifest(tmp_path: Path) -> None:
    """A manifest line without its hash pin is evidence tampering, not noise."""
    sbom, dist, manifest = cross_check_setup(tmp_path, GOOD_COMPONENTS)
    manifest.write_text("alpha==1.0\n", encoding="utf-8")
    result = sbom_tool("verify", sbom, dist, manifest)
    assert result.returncode == 1
    assert "malformed manifest line" in result.stderr
