# Release Checklist

## 1) Optional local preflight
From repo root:

```sh
python3.11 release/prepare.py {VERSION}
```

What this does:
1. Validates `{VERSION}` format (`X.Y.Z`, no `v` prefix).
2. Dry-runs version injection into `Cargo.toml` without changing tracked files.
3. Runs metadata checks and local package checks (`cargo package --no-verify --allow-dirty` and `maturin sdist`).


## 2) Ensure default branch is green
Wait for `package-check.yml` to pass on the default branch commit you want to release.

## 3) Create GitHub release
Create and publish a GitHub Release with tag `{VERSION}`.

This triggers `.github/workflows/release-publish.yml`.
The workflow applies `{VERSION}` to `Cargo.toml` inside each release job before validation/builds.

## 4) Approve PyPI publish
In Actions, approve the `publish-pypi` job in environment `pypi`.

## 5) Verify release
Check:
1. GitHub Release has `*.tar.gz` and `*.whl` assets.
2. Wheels were built for Linux `x86_64`, Linux `aarch64`, and macOS `universal2` (Intel + Apple Silicon) for each CPython `3.10` through `3.14`.
3. PyPI has `reap-pdf=={VERSION}`.

Smoke check:

```sh
python -m pip install reap-pdf=={VERSION}
python -c "import reap; print(reap.__all__)"
```
