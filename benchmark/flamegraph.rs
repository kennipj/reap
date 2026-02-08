#[path = "../tests/perf_support/mod.rs"]
mod perf_support;

use std::fs;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Duration;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use reap::rtree::TextBlockIndex;
use reap::text::extract_text_blocks;
use serde::Serialize;

#[cfg(feature = "bench-profiler")]
use pprof::criterion::{Output, PProfProfiler};

fn corpus() -> &'static perf_support::PerfCorpus {
    static CORPUS: OnceLock<perf_support::PerfCorpus> = OnceLock::new();
    CORPUS.get_or_init(|| {
        perf_support::PerfCorpus::load_default().expect("failed to load benchmark corpus")
    })
}

#[derive(Serialize)]
struct FlamegraphSummary<'a> {
    platform: String,
    benchmarks: &'a [&'a str],
    datasets: Vec<String>,
}

fn write_summary_once() {
    static WRITTEN: OnceLock<()> = OnceLock::new();
    WRITTEN.get_or_init(|| {
        let corpus = corpus();
        let summary = FlamegraphSummary {
            platform: perf_support::platform_tag(),
            benchmarks: &[
                "parse_fw2",
                "extract_blocks_fw2",
                "search_mixed_fw2",
                "pipeline_total_fw2",
            ],
            datasets: corpus
                .parse_extract_index_cases()
                .iter()
                .map(|case| case.id.clone())
                .collect(),
        };
        let out = Path::new("target/perf/flamegraph_summary.json");
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)
                .unwrap_or_else(|err| panic!("failed to create {}: {}", parent.display(), err));
        }
        let raw = serde_json::to_vec_pretty(&summary)
            .unwrap_or_else(|err| panic!("failed to serialize flamegraph summary: {}", err));
        fs::write(out, raw)
            .unwrap_or_else(|err| panic!("failed to write {}: {}", out.display(), err));
    });
}

fn bench_parse_fw2(c: &mut Criterion) {
    write_summary_once();
    let corpus = corpus();
    let case = corpus.committed_case("fw2").expect("missing fw2 dataset");
    c.bench_function("parse_fw2", move |b| {
        b.iter(|| {
            let doc = perf_support::parse_case(case).expect("parse_fw2 should succeed");
            black_box(doc.objects.len());
        })
    });
}

fn bench_extract_blocks_fw2(c: &mut Criterion) {
    write_summary_once();
    let corpus = corpus();
    let case = corpus.committed_case("fw2").expect("missing fw2 dataset");
    let doc = perf_support::parse_case(case).expect("extract_blocks_fw2 parse should succeed");
    c.bench_function("extract_blocks_fw2", move |b| {
        b.iter(|| {
            let blocks = extract_text_blocks(&doc);
            black_box(blocks.len());
        })
    });
}

fn bench_search_mixed_fw2(c: &mut Criterion) {
    write_summary_once();
    let corpus = corpus();
    let cfg = corpus.config.scenarios.search_mixed.clone();
    let case = corpus
        .committed_case(&cfg.dataset)
        .expect("missing search_mixed dataset");
    let doc = perf_support::parse_case(case).expect("search_mixed_fw2 parse should succeed");
    let blocks = extract_text_blocks(&doc);
    let mut idx = TextBlockIndex::new(blocks);
    let doc_rect = idx.doc_rect();
    let mut query_index = 0usize;

    c.bench_function("search_mixed_fw2", move |b| {
        b.iter(|| {
            let i = query_index;
            query_index = query_index.wrapping_add(1);
            let q = perf_support::build_query(
                doc_rect,
                ((i * cfg.x_mul) % 100) as f64 / 100.0,
                ((i * cfg.y_mul) % 100) as f64 / 100.0,
                cfg.w_base + (((i * cfg.w_mul) % cfg.w_mod) as f64 / 100.0),
                cfg.h_base + (((i * cfg.h_mul) % cfg.h_mod) as f64 / 100.0),
            );
            let overlap = if i % cfg.overlap_mod == 0 {
                0.0
            } else {
                cfg.overlap_alt
            };
            let out = idx.search(q, overlap);
            black_box(out.len());
        })
    });
}

fn bench_pipeline_total_fw2(c: &mut Criterion) {
    write_summary_once();
    let corpus = corpus();
    let case = corpus.committed_case("fw2").expect("missing fw2 dataset");

    c.bench_function("pipeline_total_fw2", move |b| {
        b.iter(|| {
            let doc =
                perf_support::parse_case(case).expect("pipeline_total_fw2 parse should succeed");
            let blocks = extract_text_blocks(&doc);
            let idx = TextBlockIndex::new(blocks);
            black_box(idx.doc_rect());
        })
    });
}

#[cfg(feature = "bench-profiler")]
fn criterion_config() -> Criterion {
    Criterion::default()
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)))
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5))
        .sample_size(20)
}

#[cfg(not(feature = "bench-profiler"))]
fn criterion_config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5))
        .sample_size(20)
}

criterion_group!(
    name = flamegraph_benches;
    config = criterion_config();
    targets =
        bench_parse_fw2,
        bench_extract_blocks_fw2,
        bench_search_mixed_fw2,
        bench_pipeline_total_fw2
);
criterion_main!(flamegraph_benches);
