#!/usr/bin/env python3
"""Derive, bind, and verify the release SBOM without executing dependencies.

The release workflow (``.github/workflows/publish.yml``) constructs a
CycloneDX SBOM inside an unprivileged job, and a privileged ``attest-sbom``
job later signs that document against the wheel. No code belonging to the
resolved runtime dependency graph may execute anywhere on that path: the
graph is resolved as wheel artifacts only (``pip download
--only-binary=:all:``), and this script reads their metadata statically —
nothing from the wheelhouse is ever installed, imported, or started.

``lock`` derives a hash-pinned dependency manifest from the wheelhouse's
static ``METADATA`` files, excluding the release wheel itself (matched by
byte digest) and failing closed on any non-wheel file, duplicate
distribution, or wheel that reuses the release wheel's identity with
different bytes. ``cyclonedx-py requirements`` then renders that manifest
without touching any interpreter.

``bind`` checks the root component's identity fields against the release
wheel's own ``METADATA`` and writes the wheel digest into the root:

* canonical root component ``name``  == canonical wheel ``METADATA`` ``Name``
* root component ``version``         == wheel ``METADATA`` ``Version``
* root component SHA-256             == SHA-256 of the exact wheel bytes

``verify`` independently recomputes every value from the wheel bytes and
fails closed on any mismatch, on a missing digest, and on multiple or
conflicting SHA-256 evidence; with ``--requirements`` it additionally
requires the SBOM's component set to match the manifest exactly (canonical
name, version, and every recorded SHA-256). All commands require ``--dist``
to contain exactly one wheel.

Usage::

    python scripts/release_sbom.py lock   --dist dist --wheelhouse WH --output REQS
    python scripts/release_sbom.py bind   --sbom sbom.cdx.json --dist dist
    python scripts/release_sbom.py verify --sbom sbom.cdx.json --dist dist \\
        [--requirements REQS]
"""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

SHA256_ALG = "SHA-256"


class BindingError(Exception):
    """A violated SBOM/wheel binding invariant. Every instance is fatal."""


def canonical(name: str) -> str:
    """Canonicalize a distribution name per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def resolve_exact_wheel(dist: Path) -> Path:
    """Return the sole wheel in ``dist``, failing closed on zero or many."""
    wheels = sorted(dist.glob("*.whl"))
    if len(wheels) != 1:
        found = ", ".join(wheel.name for wheel in wheels) or "none"
        raise BindingError(f"expected exactly one wheel in {dist}, found {len(wheels)}: {found}")
    return wheels[0]


def wheel_identity(wheel: Path) -> tuple[str, str]:
    """Read ``Name`` and ``Version`` from the wheel's own ``METADATA`` file."""
    with zipfile.ZipFile(wheel) as archive:
        members = [
            member
            for member in archive.namelist()
            if re.fullmatch(r"[^/]+\.dist-info/METADATA", member)
        ]
        if len(members) != 1:
            raise BindingError(
                f"{wheel.name}: expected exactly one *.dist-info/METADATA, found {members}"
            )
        metadata = email.parser.Parser().parsestr(archive.read(members[0]).decode("utf-8"))
    name, version = metadata.get("Name"), metadata.get("Version")
    if not name or not version:
        raise BindingError(f"{wheel.name}: METADATA is missing Name and/or Version")
    return name, version


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_root_component(sbom_path: Path) -> tuple[dict, dict]:
    """Return ``(document, metadata.component)``, failing closed if absent."""
    document = json.loads(sbom_path.read_text(encoding="utf-8"))
    root = document.get("metadata", {}).get("component")
    if not isinstance(root, dict):
        raise BindingError(f"{sbom_path}: SBOM has no metadata.component root")
    return document, root


def sha256_evidence(root: dict) -> list[str]:
    """Every SHA-256 digest recorded on the root component, lowercased."""
    hashes = root.get("hashes", [])
    if not isinstance(hashes, list):
        raise BindingError("root component 'hashes' is not a list")
    return [
        str(entry.get("content", "")).lower()
        for entry in hashes
        if isinstance(entry, dict) and entry.get("alg") == SHA256_ALG
    ]


def check_identity(root: dict, name: str, version: str) -> None:
    root_name = str(root.get("name", ""))
    root_version = str(root.get("version", ""))
    if canonical(root_name) != canonical(name):
        raise BindingError(
            f"root component name {root_name!r} does not match wheel METADATA Name {name!r}"
        )
    if root_version != version:
        raise BindingError(
            f"root component version {root_version!r} does not match "
            f"wheel METADATA Version {version!r}"
        )


MANIFEST_LINE = re.compile(
    r"^(?P<name>\S+)==(?P<version>\S+) --hash=sha256:(?P<digest>[0-9a-f]{64})$"
)


def lock(dist: Path, wheelhouse: Path, output: Path) -> None:
    """Derive the hash-pinned dependency manifest from static wheel metadata."""
    release_wheel = resolve_exact_wheel(dist)
    release_name, _ = wheel_identity(release_wheel)
    release_digest = sha256_of(release_wheel)

    if not wheelhouse.is_dir():
        raise BindingError(f"wheelhouse {wheelhouse} does not exist")
    non_wheels = sorted(entry.name for entry in wheelhouse.iterdir() if entry.suffix != ".whl")
    if non_wheels:
        raise BindingError(
            f"wheelhouse contains non-wheel entries {non_wheels}; a source distribution "
            f"here means --only-binary=:all: was not enforced"
        )

    dependencies: dict[str, tuple[str, str, str]] = {}
    release_copies = 0
    for wheel in sorted(wheelhouse.glob("*.whl")):
        digest = sha256_of(wheel)
        if digest == release_digest:
            release_copies += 1
            continue
        name, version = wheel_identity(wheel)
        if canonical(name) == canonical(release_name):
            raise BindingError(
                f"{wheel.name} reuses the release wheel identity {release_name!r} "
                f"with different bytes (sha256:{digest})"
            )
        if canonical(name) in dependencies:
            raise BindingError(f"wheelhouse pins {canonical(name)} more than once")
        dependencies[canonical(name)] = (name, version, digest)
    if release_copies != 1:
        raise BindingError(
            f"wheelhouse must contain the release wheel exactly once, found {release_copies}"
        )

    lines = [
        f"{name}=={version} --hash=sha256:{digest}"
        for name, version, digest in (dependencies[key] for key in sorted(dependencies))
    ]
    output.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    print(f"locked {len(lines)} dependency wheels from {wheelhouse} into {output}")


def parse_manifest(requirements: Path) -> dict[str, tuple[str, str]]:
    """Parse a ``lock`` manifest into canonical name -> (version, digest)."""
    manifest: dict[str, tuple[str, str]] = {}
    for line in requirements.read_text(encoding="utf-8").splitlines():
        match = MANIFEST_LINE.fullmatch(line)
        if match is None:
            raise BindingError(f"{requirements}: malformed manifest line {line!r}")
        name = canonical(match["name"])
        if name in manifest:
            raise BindingError(f"{requirements}: {name} pinned more than once")
        manifest[name] = (match["version"], match["digest"])
    return manifest


def component_sha256_evidence(component: dict) -> set[str]:
    """Every SHA-256 digest recorded anywhere on a component, lowercased."""
    holders = [component, *component.get("externalReferences", [])]
    return {
        digest
        for holder in holders
        if isinstance(holder, dict)
        for digest in sha256_evidence(holder)
    }


def check_components(document: dict, requirements: Path, release_name: str) -> int:
    """Fail closed unless the SBOM components exactly mirror the manifest."""
    manifest = parse_manifest(requirements)
    components: dict[str, dict] = {}
    for component in document.get("components", []):
        name = canonical(str(component.get("name", "")))
        if name == canonical(release_name):
            raise BindingError(f"release wheel {release_name!r} must not appear as a component")
        if name in components:
            raise BindingError(f"SBOM lists component {name} more than once")
        components[name] = component
    if set(components) != set(manifest):
        missing = sorted(set(manifest) - set(components))
        extra = sorted(set(components) - set(manifest))
        raise BindingError(
            f"SBOM components do not match the manifest (missing {missing}, extra {extra})"
        )
    for name, (version, digest) in manifest.items():
        component = components[name]
        if str(component.get("version", "")) != version:
            raise BindingError(
                f"component {name} version {component.get('version')!r} does not match "
                f"manifest version {version!r}"
            )
        evidence = component_sha256_evidence(component)
        if evidence != {digest}:
            raise BindingError(
                f"component {name} SHA-256 evidence {sorted(evidence)} does not match "
                f"manifest digest {digest}"
            )
    return len(manifest)


def bind(sbom_path: Path, dist: Path) -> None:
    """Record the exact wheel's SHA-256 on an identity-matching root component."""
    wheel = resolve_exact_wheel(dist)
    name, version = wheel_identity(wheel)
    digest = sha256_of(wheel)

    document, root = load_root_component(sbom_path)
    check_identity(root, name, version)

    conflicting = [entry for entry in sha256_evidence(root) if entry != digest]
    if conflicting:
        raise BindingError(
            f"root component already carries conflicting SHA-256 evidence {conflicting}; "
            f"refusing to overwrite with {digest}"
        )

    other_algs = [
        entry
        for entry in root.get("hashes", [])
        if not (isinstance(entry, dict) and entry.get("alg") == SHA256_ALG)
    ]
    root["hashes"] = [*other_algs, {"alg": SHA256_ALG, "content": digest}]
    sbom_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"bound {sbom_path} root component to {wheel.name} (sha256:{digest})")


def verify(sbom_path: Path, dist: Path, requirements: Path | None = None) -> None:
    """Recompute the binding from the wheel bytes; any deviation fails closed."""
    wheel = resolve_exact_wheel(dist)
    name, version = wheel_identity(wheel)
    digest = sha256_of(wheel)

    document, root = load_root_component(sbom_path)
    check_identity(root, name, version)

    evidence = sha256_evidence(root)
    if len(evidence) != 1:
        raise BindingError(
            f"root component must carry exactly one SHA-256 digest, found {len(evidence)}: "
            f"{evidence}"
        )
    if evidence[0] != digest:
        raise BindingError(
            f"root component SHA-256 {evidence[0]} does not match wheel {wheel.name} "
            f"digest {digest}"
        )
    checked = check_components(document, requirements, name) if requirements else None
    suffix = f" and {checked} manifest components" if checked is not None else ""
    print(f"verified {sbom_path} root component against {wheel.name} (sha256:{digest}){suffix}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    lock_parser = subparsers.add_parser("lock")
    lock_parser.add_argument("--dist", required=True, type=Path)
    lock_parser.add_argument("--wheelhouse", required=True, type=Path)
    lock_parser.add_argument("--output", required=True, type=Path)
    for command in ("bind", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--sbom", required=True, type=Path)
        subparser.add_argument("--dist", required=True, type=Path)
    subparsers.choices["verify"].add_argument("--requirements", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.command == "lock":
            lock(args.dist, args.wheelhouse, args.output)
        elif args.command == "bind":
            bind(args.sbom, args.dist)
        else:
            verify(args.sbom, args.dist, args.requirements)
    except BindingError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
