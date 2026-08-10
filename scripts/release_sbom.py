#!/usr/bin/env python3
"""Bind and verify the release SBOM against the exact built wheel.

The release workflow (``.github/workflows/publish.yml``) generates a CycloneDX
SBOM from a clean runtime environment inside an unprivileged job. Before the
privileged ``attest-sbom`` job signs that document against the wheel, the
SBOM's root component must be bound to the one wheel the release built:

* canonical root component ``name``  == canonical wheel ``METADATA`` ``Name``
* root component ``version``         == wheel ``METADATA`` ``Version``
* root component SHA-256             == SHA-256 of the exact wheel bytes

``bind`` checks the identity fields and writes the wheel digest into the root
component. ``verify`` independently recomputes every value from the wheel
bytes and fails closed on any mismatch, on a missing digest, and on multiple
or conflicting SHA-256 evidence. Both commands require ``--dist`` to contain
exactly one wheel.

Usage::

    python scripts/release_sbom.py bind   --sbom sbom.cdx.json --dist dist
    python scripts/release_sbom.py verify --sbom sbom.cdx.json --dist dist
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


def verify(sbom_path: Path, dist: Path) -> None:
    """Recompute the binding from the wheel bytes; any deviation fails closed."""
    wheel = resolve_exact_wheel(dist)
    name, version = wheel_identity(wheel)
    digest = sha256_of(wheel)

    _, root = load_root_component(sbom_path)
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
    print(f"verified {sbom_path} root component against {wheel.name} (sha256:{digest})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("bind", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--sbom", required=True, type=Path)
        subparser.add_argument("--dist", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        (bind if args.command == "bind" else verify)(args.sbom, args.dist)
    except BindingError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
