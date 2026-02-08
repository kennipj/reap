#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError as exc:
    raise SystemExit(
        "validation failed: tomllib is unavailable; run with Python 3.11+"
    ) from exc


RELEASE_TAG_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
REQUIRED_CARGO_FIELDS = (
    "name",
    "version",
    "description",
    "license",
    "readme",
    "repository",
    "homepage",
)


def fail(message: str) -> None:
    raise SystemExit(f"validation failed: {message}")


def set_output(key: str, value: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as out:
            out.write(f"{key}={value}\n")
    print(f"{key}={value}")


def load_toml(path: Path) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def validate_cargo_metadata(cargo: dict) -> str:
    package = cargo.get("package")
    if not isinstance(package, dict):
        fail("Cargo.toml is missing a [package] table")

    missing = [field for field in REQUIRED_CARGO_FIELDS if not package.get(field)]
    if missing:
        fail(f"Cargo.toml [package] missing required fields: {', '.join(missing)}")

    version = package["version"]
    if not isinstance(version, str):
        fail("Cargo.toml [package].version must be a string")
    if not RELEASE_TAG_PATTERN.fullmatch(version):
        fail("Cargo.toml [package].version must match X.Y.Z")

    return version


def validate_pyproject_metadata(pyproject: dict) -> None:
    project = pyproject.get("project")
    if not isinstance(project, dict):
        fail("pyproject.toml is missing a [project] table")

    if "version" in project:
        fail("pyproject.toml [project] must not set version explicitly")

    dynamic = project.get("dynamic")
    if not isinstance(dynamic, list) or "version" not in dynamic:
        fail("pyproject.toml [project].dynamic must include 'version'")

    for field in ("description", "readme", "license"):
        if field not in project:
            fail(f"pyproject.toml [project] missing required field: {field}")

    urls = project.get("urls")
    if not isinstance(urls, dict) or "Repository" not in urls:
        fail("pyproject.toml [project.urls] must include Repository")

    tool = pyproject.get("tool")
    if not isinstance(tool, dict):
        fail("pyproject.toml missing [tool] table")
    maturin = tool.get("maturin")
    if not isinstance(maturin, dict):
        fail("pyproject.toml missing [tool.maturin] table")

    features = maturin.get("features")
    if not isinstance(features, list) or "pyo3" not in features:
        fail("pyproject.toml [tool.maturin].features must include 'pyo3'")
    if maturin.get("python-source") != "python":
        fail("pyproject.toml [tool.maturin].python-source must be 'python'")


def validate_tag(tag: str, version: str) -> None:
    if not RELEASE_TAG_PATTERN.fullmatch(tag):
        fail("release tag must match X.Y.Z with no 'v' prefix")
    if tag != version:
        fail(
            f"release tag '{tag}' does not match Cargo version '{version}' in Cargo.toml"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate crate and Python package release metadata."
    )
    parser.add_argument(
        "--tag",
        dest="tag",
        help="release tag to validate (must match X.Y.Z)",
    )
    parser.add_argument(
        "--repo-root",
        dest="repo_root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="path to repository root (default: auto-detected)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    cargo_path = repo_root / "Cargo.toml"
    pyproject_path = repo_root / "pyproject.toml"

    if not cargo_path.exists():
        fail(f"missing file: {cargo_path}")
    if not pyproject_path.exists():
        fail(f"missing file: {pyproject_path}")

    cargo = load_toml(cargo_path)
    pyproject = load_toml(pyproject_path)

    version = validate_cargo_metadata(cargo)
    validate_pyproject_metadata(pyproject)

    if args.tag is not None:
        validate_tag(args.tag, version)

    set_output("version", version)
    print("release metadata validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
