#[path = "../tests/perf_support/mod.rs"]
mod perf_support;

use std::sync::OnceLock;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use reap::parser::ParseError;
use reap::rtree::{ScopedMergeRules, TextBlockIndex};
use reap::text::{extract_char_bboxes, extract_text_blocks};

fn corpus() -> &'static perf_support::PerfCorpus {
    static CORPUS: OnceLock<perf_support::PerfCorpus> = OnceLock::new();
    CORPUS.get_or_init(|| {
        perf_support::PerfCorpus::load_default().expect("failed to load benchmark corpus")
    })
}

fn parseable_case(case: &perf_support::PdfCase) -> bool {
    match perf_support::parse_case(case) {
        Ok(_) => true,
        Err(ParseError::InvalidPassword) if case.private => {
            eprintln!(
                "warning: skipping private benchmark dataset {} (password required)",
                case.id
            );
            false
        }
        Err(err) => panic!(
            "failed to parse benchmark dataset {} ({}): {}",
            case.id,
            case.path.display(),
            err
        ),
    }
}

fn bench_parse(c: &mut Criterion) {
    let corpus = corpus();
    let mut group = c.benchmark_group("parse");
    for case in corpus.parse_extract_index_cases() {
        if !parseable_case(case) {
            continue;
        }
        group.bench_with_input(BenchmarkId::from_parameter(&case.id), case, |b, case| {
            b.iter(|| {
                let doc = perf_support::parse_case(case).expect("parse benchmark should succeed");
                black_box(doc.objects.len());
            })
        });
    }
    group.finish();
}

fn bench_extract_blocks(c: &mut Criterion) {
    let corpus = corpus();
    let mut group = c.benchmark_group("extract_blocks");
    for case in corpus.parse_extract_index_cases() {
        if !parseable_case(case) {
            continue;
        }
        let doc = perf_support::parse_case(case).expect("extract benchmark parse should succeed");
        group.bench_function(BenchmarkId::from_parameter(&case.id), move |b| {
            b.iter(|| {
                let blocks = extract_text_blocks(&doc);
                black_box(blocks.len());
            })
        });
    }
    group.finish();
}

fn bench_index_build(c: &mut Criterion) {
    let corpus = corpus();
    let mut group = c.benchmark_group("index_build");
    for case in corpus.parse_extract_index_cases() {
        if !parseable_case(case) {
            continue;
        }
        let doc = perf_support::parse_case(case).expect("index benchmark parse should succeed");
        let blocks = extract_text_blocks(&doc);
        group.bench_function(BenchmarkId::from_parameter(&case.id), move |b| {
            b.iter(|| {
                let idx = TextBlockIndex::new(blocks.clone());
                black_box(idx.doc_rect());
            })
        });
    }
    group.finish();
}

fn bench_search_cached(c: &mut Criterion) {
    let corpus = corpus();
    let cfg = &corpus.config.scenarios.search_cached;
    let case = corpus
        .committed_case(&cfg.dataset)
        .expect("missing search_cached dataset");
    let doc = perf_support::parse_case(case).expect("search_cached parse should succeed");
    let blocks = extract_text_blocks(&doc);
    let mut idx = TextBlockIndex::new(blocks);
    let query = perf_support::build_query(
        idx.doc_rect(),
        cfg.x_frac,
        cfg.y_frac,
        cfg.w_frac,
        cfg.h_frac,
    );

    c.bench_function("search_cached/fw2", move |b| {
        b.iter(|| {
            let out = idx.search(query, cfg.overlap);
            black_box(out.len());
        })
    });
}

fn bench_search_mixed(c: &mut Criterion) {
    let corpus = corpus();
    let cfg = corpus.config.scenarios.search_mixed.clone();
    let case = corpus
        .committed_case(&cfg.dataset)
        .expect("missing search_mixed dataset");
    let doc = perf_support::parse_case(case).expect("search_mixed parse should succeed");
    let blocks = extract_text_blocks(&doc);
    let mut idx = TextBlockIndex::new(blocks);
    let doc_rect = idx.doc_rect();
    let mut query_index = 0usize;

    c.bench_function("search_mixed/fw2", move |b| {
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

fn bench_search_regex_cached(c: &mut Criterion) {
    let corpus = corpus();
    let cfg = corpus.config.scenarios.search_regex_cached.clone();
    let case = corpus
        .committed_case(&cfg.dataset)
        .expect("missing search_regex_cached dataset");
    let doc = perf_support::parse_case(case).expect("search_regex_cached parse should succeed");
    let blocks = extract_text_blocks(&doc);
    let mut idx = TextBlockIndex::new(blocks);

    c.bench_function("search_regex_cached/fw2", move |b| {
        b.iter(|| {
            let out = idx
                .search_regex(&cfg.pattern)
                .expect("search_regex_cached pattern should be valid");
            black_box(out.len());
        })
    });
}

fn bench_scoped_merge_rules(c: &mut Criterion) {
    let corpus = corpus();
    let cfg = corpus.config.scenarios.scoped_merge_rules.clone();
    let case = corpus
        .committed_case(&cfg.dataset)
        .expect("missing scoped_merge_rules dataset");
    let doc = perf_support::parse_case(case).expect("scoped_merge_rules parse should succeed");
    let blocks = extract_text_blocks(&doc);
    let idx = TextBlockIndex::new(blocks);
    let rect = idx.doc_rect();

    c.bench_function("scoped_merge_rules/fw2", move |b| {
        b.iter(|| {
            let scoped = idx.scoped(
                rect,
                cfg.overlap,
                cfg.merge_threshold,
                cfg.normalize,
                ScopedMergeRules {
                    no_numeric_pair_merge: true,
                    no_date_pair_merge: true,
                },
            );
            black_box(scoped.block_len());
        })
    });
}

fn bench_parse_encrypted_empty_password(c: &mut Criterion) {
    let corpus = corpus();
    let case = corpus
        .committed_case("edge_encrypted_empty_password")
        .expect("missing encrypted benchmark dataset");

    c.bench_function("parse_encrypted_empty_password", move |b| {
        b.iter(|| {
            let doc =
                perf_support::parse_case(case).expect("encrypted parse benchmark should succeed");
            black_box(doc.objects.len());
        })
    });
}

fn bench_extract_chars(c: &mut Criterion) {
    let corpus = corpus();
    let mut group = c.benchmark_group("extract_chars");
    for case in corpus.parse_extract_index_cases() {
        if !parseable_case(case) {
            continue;
        }
        let doc = perf_support::parse_case(case).expect("extract_chars parse should succeed");
        group.bench_function(BenchmarkId::from_parameter(&case.id), move |b| {
            b.iter(|| {
                let chars = extract_char_bboxes(&doc);
                black_box(chars.len());
            })
        });
    }
    group.finish();
}

fn bench_pipeline_total(c: &mut Criterion) {
    let corpus = corpus();
    let mut group = c.benchmark_group("pipeline_total");
    for case in corpus.parse_extract_index_cases() {
        if !parseable_case(case) {
            continue;
        }
        group.bench_with_input(BenchmarkId::from_parameter(&case.id), case, |b, case| {
            b.iter(|| {
                let doc = perf_support::parse_case(case).expect("pipeline parse should succeed");
                let blocks = extract_text_blocks(&doc);
                let idx = TextBlockIndex::new(blocks);
                black_box(idx.doc_rect());
            })
        });
    }
    group.finish();
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(8))
        .sample_size(50)
}

criterion_group!(
    name = benches;
    config = criterion_config();
    targets =
        bench_parse,
        bench_extract_blocks,
        bench_index_build,
        bench_search_cached,
        bench_search_mixed,
        bench_search_regex_cached,
        bench_scoped_merge_rules,
        bench_parse_encrypted_empty_password,
        bench_extract_chars,
        bench_pipeline_total
);
criterion_main!(benches);
