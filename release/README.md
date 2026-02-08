# Release Checklist

## 1) Prepare release locally
From repo root:

```sh
python3.11 release/prepare.py {VERSION}
```

What this does:
1. Updates `Cargo.toml` version.
2. Runs `release/validate.py --tag {VERSION}`.
3. Runs local package checks (`cargo package --no-verify --allow-dirty` and `maturin sdist`).


## 2) Push release commit
```sh
git add Cargo.toml
git commit -m "release: {VERSION}"
git push
```

Wait for `package-check.yml` to pass.

## 3) Create GitHub release
Create and publish a GitHub Release with tag `{VERSION}`.

This triggers `.github/workflows/release-publish.yml`.

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
