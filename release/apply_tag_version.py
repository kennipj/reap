#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
PACKAGE_VERSION_LINE = re.compile(r'^(\s*version\s*=\s*")([^"]+)(".*)$')


def fail(message: str) -> None:
    raise SystemExit(f"version apply failed: {message}")


def set_output(key: str, value: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as out:
            out.write(f"{key}={value}\n")
    print(f"{key}={value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply release tag version to Cargo.toml [package].version."
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="release tag (must match X.Y.Z; no v-prefix)",
    )
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="path to repository root (default: auto-detected)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and report changes without writing Cargo.toml",
    )
    return parser.parse_args()


def update_cargo_version(cargo_path: Path, target_version: str, dry_run: bool) -> str:
    text = cargo_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    in_package = False
    current_version = None

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_package = stripped == "[package]"
            continue
        if not in_package:
            continue

        match = PACKAGE_VERSION_LINE.match(line)
        if match:
            current_version = match.group(2)
            lines[idx] = f"{match.group(1)}{target_version}{match.group(3)}\n"
            break

    if current_version is None:
        fail("could not find [package].version in Cargo.toml")

    if not dry_run and current_version != target_version:
        cargo_path.write_text("".join(lines), encoding="utf-8")

    return current_version


def main() -> int:
    args = parse_args()
    tag = args.tag.strip()
    if not VERSION_PATTERN.fullmatch(tag):
        fail("tag must match X.Y.Z with no 'v' prefix")

    repo_root = args.repo_root.resolve()
    cargo_path = repo_root / "Cargo.toml"
    if not cargo_path.exists():
        fail(f"missing file: {cargo_path}")

    previous_version = update_cargo_version(cargo_path, tag, args.dry_run)

    if args.dry_run:
        print(
            "dry-run: "
            f"would update Cargo.toml [package].version {previous_version} -> {tag}"
        )
    elif previous_version == tag:
        print(f"Cargo.toml already set to release version {tag}")
    else:
        print(f"updated Cargo.toml version: {previous_version} -> {tag}")

    set_output("version", tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
