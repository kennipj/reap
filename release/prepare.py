#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
PACKAGE_VERSION_LINE = re.compile(r'^(\s*version\s*=\s*")([^"]+)(".*)$')


def fail(message: str) -> None:
    raise SystemExit(f"release prepare failed: {message}")


def ensure_python_version() -> None:
    if sys.version_info < (3, 11):
        fail("Python 3.11+ is required")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a new release by bumping Cargo version and running checks."
    )
    parser.add_argument("version", help="new version (X.Y.Z)")
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="repository root (default: auto-detected)",
    )
    parser.add_argument(
        "--skip-package-checks",
        action="store_true",
        help="skip cargo package and maturin sdist commands",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="also run cargo test",
    )
    parser.add_argument(
        "--sdist-out",
        default=Path("/tmp/reap-release-check"),
        type=Path,
        help="output directory for local sdist check (default: /tmp/reap-release-check)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would change without writing files",
    )
    return parser.parse_args()


def update_cargo_version(
    cargo_path: Path, target_version: str, dry_run: bool
) -> tuple[str, bool]:
    text = cargo_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    in_package = False
    found_version = False
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
            found_version = True
            break

    if not found_version or current_version is None:
        fail("could not find [package].version in Cargo.toml")

    changed = current_version != target_version
    if changed and not dry_run:
        cargo_path.write_text("".join(lines), encoding="utf-8")

    return current_version, changed


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> int:
    ensure_python_version()
    args = parse_args()

    version = args.version.strip()
    if not VERSION_PATTERN.fullmatch(version):
        fail("version must match X.Y.Z")

    repo_root = args.repo_root.resolve()
    cargo_path = repo_root / "Cargo.toml"
    validator_path = repo_root / "release" / "validate.py"

    if not cargo_path.exists():
        fail(f"missing file: {cargo_path}")
    if not validator_path.exists():
        fail(f"missing file: {validator_path}")

    current_version, changed = update_cargo_version(cargo_path, version, args.dry_run)
    if changed:
        if args.dry_run:
            print(
                f"dry-run: would update Cargo.toml version from {current_version} to {version}"
            )
        else:
            print(f"updated Cargo.toml version: {current_version} -> {version}")
    else:
        print(f"Cargo.toml already at version {version}")

    if args.dry_run and changed:
        print(
            "dry-run: skipping release/validate.py --tag check because Cargo.toml was not modified"
        )
    else:
        run_cmd([sys.executable, str(validator_path), "--tag", version], repo_root)

    if not args.skip_package_checks:
        run_cmd(
            ["cargo", "package", "--no-verify", "--allow-dirty"], repo_root
        )
        run_cmd(
            ["uv", "run", "maturin", "sdist", "--out", str(args.sdist_out)],
            repo_root,
        )

    if args.run_tests:
        run_cmd(["cargo", "test"], repo_root)

    print("")
    print("next steps:")
    print(f"  git add Cargo.toml")
    print(f'  git commit -m "release: {version}"')
    print("  git push")
    print(f"  create and publish GitHub Release tag {version}")
    print("  approve the publish-pypi job in environment 'pypi'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
