# Performance Benchmarking

`reap` uses two complementary performance layers:

1. Criterion benchmarks for granular measurements and hotspot discovery.
2. Release-mode regression tests against baseline JSON snapshots for CI gating.

## Benchmark Corpus

Configuration lives in `benchmarks/corpus.toml`.
Committed benchmark fixtures:

- `fw2`
- `edge_incremental_update_objstm`
- `edge_encrypted_empty_password` (empty password)

Deterministic query/scenario parameters (search/scoped/regex) are also defined there.

## Criterion Benchmarks

Run core benchmarks:

```bash
cargo bench --bench criterion_core
```

This benchmark includes committed corpus and, if configured, private local PDFs.

## Regression Gate

Run release-mode regression checks:

```bash
cargo test --release --test perf_regression -- --nocapture
```

The test compares current metrics against:

- `tests/fixtures/perf_baseline.macos-aarch64.json`

Platform selection is automatic (`<os>-<arch>`).

## Baseline Refresh

Record (overwrite) baseline snapshots explicitly:

```bash
REAP_PERF_RECORD_BASELINE=1 cargo test --release --test perf_regression -- --nocapture
```

- Committed baseline file is updated for the current platform.
- Private baseline is written only when private datasets are configured.

## Flamegraph Profiling

Use the flamegraph bench target:

```bash
cargo bench --bench flamegraph --features bench-profiler
```

This writes a summary artifact at `target/perf/flamegraph_summary.json` and emits profiler output through Criterion/pprof.

## Private Local Suites

Private datasets are loaded from `REAP_BENCH_PRIVATE_DIRS` using OS path-list semantics (`std::env::split_paths`).

Example:

```bash
REAP_BENCH_PRIVATE_DIRS="/abs/private/pdfs:/abs/more/pdfs" cargo bench --bench criterion_core
```

Optional controls:

- `REAP_BENCH_PRIVATE_MAX_FILES` (default: `50`)
- `REAP_PERF_PRIVATE_BASELINE` (default: `target/perf/private_baseline.<platform>.json`)
- `REAP_PERF_WRITE_NAME_MAP=1` (writes `target/perf/private_name_map.<platform>.json`)

Private dataset identifiers are anonymized as `priv_<sha256_prefix>` in reports.
Encrypted private PDFs without credentials are skipped with a warning.
