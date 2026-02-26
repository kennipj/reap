#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


def fail(message: str) -> None:
    raise SystemExit(f"release preflight failed: {message}")


def ensure_python_version() -> None:
    if sys.version_info < (3, 11):
        fail("Python 3.11+ is required")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run non-mutating release preflight checks for a release tag (X.Y.Z)."
        )
    )
    parser.add_argument("version", help="release tag/version (X.Y.Z)")
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
    return parser.parse_args()


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> int:
    ensure_python_version()
    args = parse_args()

    version = args.version.strip()
    if not VERSION_PATTERN.fullmatch(version):
        fail("version must match X.Y.Z (no 'v' prefix)")

    repo_root = args.repo_root.resolve()
    apply_script = repo_root / "release" / "apply_tag_version.py"
    validator_path = repo_root / "release" / "validate.py"

    if not apply_script.exists():
        fail(f"missing file: {apply_script}")
    if not validator_path.exists():
        fail(f"missing file: {validator_path}")

    run_cmd(
        [
            sys.executable,
            str(apply_script),
            "--tag",
            version,
            "--repo-root",
            str(repo_root),
            "--dry-run",
        ],
        repo_root,
    )

    run_cmd(
        [
            sys.executable,
            str(validator_path),
            "--repo-root",
            str(repo_root),
        ],
        repo_root,
    )

    if not args.skip_package_checks:
        run_cmd(["cargo", "package", "--no-verify", "--allow-dirty"], repo_root)
        run_cmd(
            ["uv", "run", "maturin", "sdist", "--out", str(args.sdist_out)],
            repo_root,
        )

    if args.run_tests:
        run_cmd(["cargo", "test"], repo_root)

    print("")
    print("next steps:")
    print("  ensure package-check.yml is green on default branch")
    print(f"  create and publish GitHub Release tag {version}")
    print("  approve the publish-pypi job in environment 'pypi'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
