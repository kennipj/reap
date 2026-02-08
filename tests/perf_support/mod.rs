#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use reap::parser::{ParseError, Parser, PdfDoc};
use reap::rtree::{ScopedMergeRules, TextBlockIndex};
use reap::text::{Rectangle, extract_char_bboxes, extract_text_blocks};
use reap::tokenizer::Lexer;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub type PerfResult<T> = Result<T, String>;

const CORE_WARMUP: usize = 2;
const CORE_ITERS: usize = 8;
const SEARCH_WARMUP: usize = 100;
const SEARCH_ITERS: usize = 600;
const MIXED_WARMUP: usize = 50;
const SCOPED_WARMUP: usize = 20;
const PRIVATE_MAX_FILES_DEFAULT: usize = 50;

#[derive(Debug, Clone, Deserialize)]
pub struct CorpusConfig {
    pub committed: BTreeMap<String, CorpusEntry>,
    pub scenarios: ScenarioConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CorpusEntry {
    pub path: String,
    #[serde(default)]
    pub password: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ScenarioConfig {
    pub search_cached: SearchCachedConfig,
    pub search_mixed: SearchMixedConfig,
    pub search_regex_cached: SearchRegexConfig,
    pub scoped_merge_rules: ScopedMergeRulesConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearchCachedConfig {
    pub dataset: String,
    pub x_frac: f64,
    pub y_frac: f64,
    pub w_frac: f64,
    pub h_frac: f64,
    pub overlap: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearchMixedConfig {
    pub dataset: String,
    pub iterations: usize,
    pub x_mul: usize,
    pub y_mul: usize,
    pub w_mul: usize,
    pub h_mul: usize,
    pub w_base: f64,
    pub w_mod: usize,
    pub h_base: f64,
    pub h_mod: usize,
    pub overlap_mod: usize,
    pub overlap_alt: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearchRegexConfig {
    pub dataset: String,
    pub pattern: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ScopedMergeRulesConfig {
    pub dataset: String,
    pub overlap: f64,
    pub merge_threshold: f64,
    pub normalize: bool,
    pub iterations: usize,
}

#[derive(Debug, Clone)]
pub struct PdfCase {
    pub id: String,
    pub path: PathBuf,
    pub bytes: Vec<u8>,
    pub password: Option<Vec<u8>>,
    pub private: bool,
}

#[derive(Debug, Clone)]
pub struct PerfCorpus {
    pub config: CorpusConfig,
    committed: BTreeMap<String, PdfCase>,
    private: Vec<PdfCase>,
}

impl PerfCorpus {
    pub fn load_default() -> PerfResult<Self> {
        let config_path = corpus_config_path();
        let config = load_corpus_config(&config_path)?;
        let committed = load_committed_cases(&config)?;
        let private = load_private_cases()?;
        maybe_write_private_name_map(&private)?;
        Ok(Self {
            config,
            committed,
            private,
        })
    }

    pub fn committed_case(&self, id: &str) -> PerfResult<&PdfCase> {
        self.committed
            .get(id)
            .ok_or_else(|| format!("missing committed corpus entry: {}", id))
    }

    pub fn committed_cases(&self) -> Vec<&PdfCase> {
        self.committed.values().collect()
    }

    pub fn private_cases(&self) -> &[PdfCase] {
        &self.private
    }

    pub fn parse_extract_index_cases(&self) -> Vec<&PdfCase> {
        let mut out: Vec<&PdfCase> = self.committed.values().collect();
        out.extend(self.private.iter());
        out
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StatisticKind {
    MedianMs,
    P95Ms,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ThresholdClass {
    Core,
    MixedScopedEncryption,
    Cached,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerfMetric {
    pub scenario_id: String,
    pub dataset_id: String,
    pub stat: StatisticKind,
    pub metric_ms: f64,
    pub threshold_class: ThresholdClass,
}

impl PerfMetric {
    pub fn key(&self) -> String {
        format!("{}::{}", self.scenario_id, self.dataset_id)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineFile {
    pub platform: String,
    pub metrics: Vec<PerfMetric>,
}

#[derive(Debug, Clone, Copy)]
pub struct ThresholdPolicy {
    pub core_relative: f64,
    pub core_min_delta_ms: f64,
    pub mixed_relative: f64,
    pub mixed_min_delta_ms: f64,
    pub cached_relative: f64,
    pub cached_min_delta_ms: f64,
}

impl Default for ThresholdPolicy {
    fn default() -> Self {
        Self {
            core_relative: 0.15,
            core_min_delta_ms: 0.50,
            mixed_relative: 0.20,
            mixed_min_delta_ms: 0.50,
            cached_relative: 0.30,
            cached_min_delta_ms: 0.05,
        }
    }
}

impl ThresholdPolicy {
    pub fn limits(self, class: ThresholdClass) -> (f64, f64) {
        match class {
            ThresholdClass::Core => (self.core_relative, self.core_min_delta_ms),
            ThresholdClass::MixedScopedEncryption => (self.mixed_relative, self.mixed_min_delta_ms),
            ThresholdClass::Cached => (self.cached_relative, self.cached_min_delta_ms),
        }
    }
}

pub fn corpus_config_path() -> PathBuf {
    PathBuf::from("benchmark/corpus.toml")
}

pub fn platform_tag() -> String {
    format!("{}-{}", env::consts::OS, env::consts::ARCH)
}

pub fn should_record_baseline() -> bool {
    env::var("REAP_PERF_RECORD_BASELINE")
        .map(|v| v == "1")
        .unwrap_or(false)
}

pub fn committed_baseline_path() -> PathBuf {
    PathBuf::from(format!(
        "tests/fixtures/perf_baseline.{}.json",
        platform_tag()
    ))
}

pub fn private_baseline_path() -> PathBuf {
    if let Ok(path) = env::var("REAP_PERF_PRIVATE_BASELINE") {
        return PathBuf::from(path);
    }
    PathBuf::from(format!(
        "target/perf/private_baseline.{}.json",
        platform_tag()
    ))
}

pub fn load_corpus_config(path: &Path) -> PerfResult<CorpusConfig> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {}", path.display(), err))?;
    toml::from_str(&raw).map_err(|err| format!("invalid corpus TOML {}: {}", path.display(), err))
}

fn load_committed_cases(config: &CorpusConfig) -> PerfResult<BTreeMap<String, PdfCase>> {
    let mut out = BTreeMap::new();
    for (id, entry) in &config.committed {
        let path = PathBuf::from(&entry.path);
        let bytes = fs::read(&path)
            .map_err(|err| format!("failed to read committed PDF {}: {}", path.display(), err))?;
        out.insert(
            id.clone(),
            PdfCase {
                id: id.clone(),
                path,
                bytes,
                password: entry.password.as_ref().map(|pwd| pwd.as_bytes().to_vec()),
                private: false,
            },
        );
    }
    Ok(out)
}

fn load_private_cases() -> PerfResult<Vec<PdfCase>> {
    let Some(raw_dirs) = env::var_os("REAP_BENCH_PRIVATE_DIRS") else {
        return Ok(Vec::new());
    };

    let max_files = env::var("REAP_BENCH_PRIVATE_MAX_FILES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(PRIVATE_MAX_FILES_DEFAULT);

    let mut pdf_paths = Vec::new();
    for root in env::split_paths(&raw_dirs) {
        if root.as_os_str().is_empty() {
            continue;
        }
        collect_pdf_paths(&root, &mut pdf_paths)?;
    }
    pdf_paths.sort();

    let mut seen_ids: HashSet<String> = HashSet::new();
    let mut out = Vec::new();
    for path in pdf_paths {
        if out.len() >= max_files {
            break;
        }
        let bytes = fs::read(&path)
            .map_err(|err| format!("failed to read private PDF {}: {}", path.display(), err))?;
        let id = private_id_from_bytes(&bytes);
        if !seen_ids.insert(id.clone()) {
            continue;
        }
        out.push(PdfCase {
            id,
            path,
            bytes,
            password: None,
            private: true,
        });
    }

    Ok(out)
}

fn collect_pdf_paths(root: &Path, out: &mut Vec<PathBuf>) -> PerfResult<()> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(err) => {
                return Err(format!(
                    "failed to list private benchmark directory {}: {}",
                    dir.display(),
                    err
                ));
            }
        };
        for entry in entries {
            let entry = entry.map_err(|err| {
                format!(
                    "failed to inspect private benchmark directory entry in {}: {}",
                    dir.display(),
                    err
                )
            })?;
            let path = entry.path();
            let file_type = entry.file_type().map_err(|err| {
                format!(
                    "failed to inspect file type for {}: {}",
                    path.display(),
                    err
                )
            })?;
            if file_type.is_dir() {
                stack.push(path);
            } else if file_type.is_file() && is_pdf(&path) {
                out.push(path);
            }
        }
    }
    Ok(())
}

fn is_pdf(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("pdf"))
        .unwrap_or(false)
}

fn private_id_from_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut short = String::with_capacity(10);
    for byte in digest.iter().take(5) {
        let _ = write!(&mut short, "{:02x}", byte);
    }
    format!("priv_{}", short)
}

fn maybe_write_private_name_map(private: &[PdfCase]) -> PerfResult<()> {
    if private.is_empty() {
        return Ok(());
    }
    let write_map = env::var("REAP_PERF_WRITE_NAME_MAP")
        .map(|v| v == "1")
        .unwrap_or(false);
    if !write_map {
        return Ok(());
    }

    #[derive(Serialize)]
    struct NameMapEntry {
        id: String,
        path: String,
    }

    let entries: Vec<NameMapEntry> = private
        .iter()
        .map(|case| NameMapEntry {
            id: case.id.clone(),
            path: case.path.display().to_string(),
        })
        .collect();

    let path = PathBuf::from(format!(
        "target/perf/private_name_map.{}.json",
        platform_tag()
    ));
    write_json(&path, &entries)
}

pub fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

pub fn p95(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let idx = ((values.len() as f64) * 0.95).ceil() as usize - 1;
    values[idx.min(values.len().saturating_sub(1))]
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn timed_samples<F>(warmup: usize, iterations: usize, mut op: F) -> PerfResult<Vec<f64>>
where
    F: FnMut() -> PerfResult<()>,
{
    let mut out = Vec::with_capacity(iterations);
    for i in 0..(warmup + iterations) {
        let start = Instant::now();
        op()?;
        if i >= warmup {
            out.push(start.elapsed().as_secs_f64() * 1000.0);
        }
    }
    Ok(out)
}

pub fn build_query(
    doc_rect: Rectangle,
    x_frac: f64,
    y_frac: f64,
    w_frac: f64,
    h_frac: f64,
) -> Rectangle {
    let width = (doc_rect.right - doc_rect.left).max(1.0);
    let height = (doc_rect.bottom - doc_rect.top).max(1.0);
    let query_w = (width * w_frac).max(1.0);
    let query_h = (height * h_frac).max(1.0);
    let left = doc_rect.left + (width - query_w) * x_frac;
    let top = doc_rect.top + (height - query_h) * y_frac;
    Rectangle {
        top,
        left,
        bottom: top + query_h,
        right: left + query_w,
    }
}

pub fn parse_case(case: &PdfCase) -> Result<PdfDoc, ParseError> {
    let lexer = Lexer::new(&case.bytes);
    Parser::new(lexer).parse_with_password(case.password.as_deref())
}

fn parse_case_or_err(case: &PdfCase) -> PerfResult<PdfDoc> {
    parse_case(case).map_err(|err| {
        format!(
            "failed to parse {} ({}) for benchmark: {}",
            case.id,
            case.path.display(),
            err
        )
    })
}

fn parse_case_private_opt(case: &PdfCase) -> PerfResult<Option<PdfDoc>> {
    match parse_case(case) {
        Ok(doc) => Ok(Some(doc)),
        Err(ParseError::InvalidPassword) if case.private && case.password.is_none() => {
            eprintln!(
                "warning: skipping private dataset {} because it requires a password",
                case.id
            );
            Ok(None)
        }
        Err(err) => Err(format!(
            "failed to parse {} ({}) for benchmark: {}",
            case.id,
            case.path.display(),
            err
        )),
    }
}

fn metric(
    scenario_id: &str,
    dataset_id: &str,
    stat: StatisticKind,
    metric_ms: f64,
    class: ThresholdClass,
) -> PerfMetric {
    PerfMetric {
        scenario_id: scenario_id.to_string(),
        dataset_id: dataset_id.to_string(),
        stat,
        metric_ms,
        threshold_class: class,
    }
}

fn measure_parse_median_ms(case: &PdfCase) -> PerfResult<f64> {
    let mut samples = timed_samples(CORE_WARMUP, CORE_ITERS, || {
        let _doc = parse_case(case).map_err(|err| {
            format!(
                "failed to parse {} ({}) during parse benchmark: {}",
                case.id,
                case.path.display(),
                err
            )
        })?;
        Ok(())
    })?;
    Ok(median(&mut samples))
}

fn measure_extract_blocks_median_ms(doc: &PdfDoc) -> PerfResult<f64> {
    let mut samples = timed_samples(CORE_WARMUP, CORE_ITERS, || {
        let blocks = extract_text_blocks(doc);
        if blocks.is_empty() {
            return Err("extract_text_blocks produced zero blocks".to_string());
        }
        black_box(blocks.len());
        Ok(())
    })?;
    Ok(median(&mut samples))
}

fn measure_index_build_median_ms(blocks: &[reap::text::TextBlock]) -> PerfResult<f64> {
    let mut samples = timed_samples(CORE_WARMUP, CORE_ITERS, || {
        let idx = TextBlockIndex::new(blocks.to_vec());
        black_box(idx.doc_rect());
        Ok(())
    })?;
    Ok(median(&mut samples))
}

fn measure_search_cached_median_ms(
    idx: &mut TextBlockIndex,
    query: Rectangle,
    overlap: f64,
) -> PerfResult<f64> {
    let mut samples = timed_samples(SEARCH_WARMUP, SEARCH_ITERS, || {
        let out = idx.search(query, overlap);
        black_box(out.len());
        Ok(())
    })?;
    Ok(median(&mut samples))
}

fn measure_search_mixed_p95_ms(
    idx: &mut TextBlockIndex,
    doc_rect: Rectangle,
    cfg: &SearchMixedConfig,
) -> PerfResult<f64> {
    let mut query_index = 0usize;
    let mut samples = timed_samples(MIXED_WARMUP, cfg.iterations, || {
        let i = query_index;
        query_index += 1;
        let q = build_query(
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
        Ok(())
    })?;
    Ok(p95(&mut samples))
}

fn measure_search_regex_cached_median_ms(
    idx: &mut TextBlockIndex,
    pattern: &str,
) -> PerfResult<f64> {
    let mut samples = timed_samples(SEARCH_WARMUP, SEARCH_ITERS, || {
        let out = idx
            .search_regex(pattern)
            .map_err(|err| format!("invalid regex pattern {}: {:?}", pattern, err))?;
        black_box(out.len());
        Ok(())
    })?;
    Ok(median(&mut samples))
}

fn measure_scoped_merge_rules_median_ms(
    idx: &TextBlockIndex,
    cfg: &ScopedMergeRulesConfig,
) -> PerfResult<(f64, f64)> {
    let mut baseline_samples = Vec::with_capacity(cfg.iterations);
    let mut guarded_samples = Vec::with_capacity(cfg.iterations);
    let rect = idx.doc_rect();

    for i in 0..(SCOPED_WARMUP + cfg.iterations) {
        let baseline_start = Instant::now();
        let baseline = idx.scoped(
            rect,
            cfg.overlap,
            cfg.merge_threshold,
            cfg.normalize,
            ScopedMergeRules::default(),
        );
        black_box(baseline.block_len());
        let baseline_ms = baseline_start.elapsed().as_secs_f64() * 1000.0;

        let guarded_start = Instant::now();
        let guarded = idx.scoped(
            rect,
            cfg.overlap,
            cfg.merge_threshold,
            cfg.normalize,
            ScopedMergeRules {
                no_numeric_pair_merge: true,
                no_date_pair_merge: true,
            },
        );
        black_box(guarded.block_len());
        let guarded_ms = guarded_start.elapsed().as_secs_f64() * 1000.0;

        if i >= SCOPED_WARMUP {
            baseline_samples.push(baseline_ms);
            guarded_samples.push(guarded_ms);
        }
    }

    let baseline_mean = mean(&baseline_samples);
    let guarded_mean = mean(&guarded_samples);
    let ratio = if baseline_mean > 0.0 {
        guarded_mean / baseline_mean
    } else {
        1.0
    };

    let mut guarded = guarded_samples;
    Ok((median(&mut guarded), ratio))
}

pub fn run_committed_regression(corpus: &PerfCorpus) -> PerfResult<Vec<PerfMetric>> {
    let fw2 = corpus.committed_case("fw2")?;
    let edge_inc = corpus.committed_case("edge_incremental_update_objstm")?;
    let edge_enc = corpus.committed_case("edge_encrypted_empty_password")?;

    let fw2_doc = parse_case_or_err(fw2)?;
    let edge_inc_doc = parse_case_or_err(edge_inc)?;

    let fw2_blocks = extract_text_blocks(&fw2_doc);
    if fw2_blocks.is_empty() {
        return Err("fw2 extracted zero blocks".to_string());
    }

    let mut out = Vec::new();

    out.push(metric(
        "parse",
        &fw2.id,
        StatisticKind::MedianMs,
        measure_parse_median_ms(fw2)?,
        ThresholdClass::Core,
    ));
    out.push(metric(
        "parse",
        &edge_inc.id,
        StatisticKind::MedianMs,
        measure_parse_median_ms(edge_inc)?,
        ThresholdClass::Core,
    ));
    out.push(metric(
        "extract_blocks",
        &fw2.id,
        StatisticKind::MedianMs,
        measure_extract_blocks_median_ms(&fw2_doc)?,
        ThresholdClass::Core,
    ));
    out.push(metric(
        "extract_blocks",
        &edge_inc.id,
        StatisticKind::MedianMs,
        measure_extract_blocks_median_ms(&edge_inc_doc)?,
        ThresholdClass::Core,
    ));
    out.push(metric(
        "index_build",
        &fw2.id,
        StatisticKind::MedianMs,
        measure_index_build_median_ms(&fw2_blocks)?,
        ThresholdClass::Core,
    ));

    let search_cfg = &corpus.config.scenarios.search_cached;
    if search_cfg.dataset != fw2.id {
        return Err(format!(
            "search_cached dataset mismatch: expected fw2, got {}",
            search_cfg.dataset
        ));
    }
    let mut cached_idx = TextBlockIndex::new(fw2_blocks.clone());
    let query = build_query(
        cached_idx.doc_rect(),
        search_cfg.x_frac,
        search_cfg.y_frac,
        search_cfg.w_frac,
        search_cfg.h_frac,
    );
    out.push(metric(
        "search_cached",
        &fw2.id,
        StatisticKind::MedianMs,
        measure_search_cached_median_ms(&mut cached_idx, query, search_cfg.overlap)?,
        ThresholdClass::Cached,
    ));

    let mixed_cfg = &corpus.config.scenarios.search_mixed;
    if mixed_cfg.dataset != fw2.id {
        return Err(format!(
            "search_mixed dataset mismatch: expected fw2, got {}",
            mixed_cfg.dataset
        ));
    }
    let mut mixed_idx = TextBlockIndex::new(fw2_blocks.clone());
    let mixed_doc_rect = mixed_idx.doc_rect();
    out.push(metric(
        "search_mixed",
        &fw2.id,
        StatisticKind::P95Ms,
        measure_search_mixed_p95_ms(&mut mixed_idx, mixed_doc_rect, mixed_cfg)?,
        ThresholdClass::MixedScopedEncryption,
    ));

    let regex_cfg = &corpus.config.scenarios.search_regex_cached;
    if regex_cfg.dataset != fw2.id {
        return Err(format!(
            "search_regex_cached dataset mismatch: expected fw2, got {}",
            regex_cfg.dataset
        ));
    }
    let mut regex_idx = TextBlockIndex::new(fw2_blocks.clone());
    out.push(metric(
        "search_regex_cached",
        &fw2.id,
        StatisticKind::MedianMs,
        measure_search_regex_cached_median_ms(&mut regex_idx, &regex_cfg.pattern)?,
        ThresholdClass::Cached,
    ));

    let scoped_cfg = &corpus.config.scenarios.scoped_merge_rules;
    if scoped_cfg.dataset != fw2.id {
        return Err(format!(
            "scoped_merge_rules dataset mismatch: expected fw2, got {}",
            scoped_cfg.dataset
        ));
    }
    let scoped_idx = TextBlockIndex::new(fw2_blocks.clone());
    let (scoped_median, scoped_ratio) =
        measure_scoped_merge_rules_median_ms(&scoped_idx, scoped_cfg)?;
    println!(
        "scoped_merge_rules diagnostic ratio (guarded/baseline): {:.3}",
        scoped_ratio
    );
    out.push(metric(
        "scoped_merge_rules",
        &fw2.id,
        StatisticKind::MedianMs,
        scoped_median,
        ThresholdClass::MixedScopedEncryption,
    ));

    out.push(metric(
        "parse_encrypted_empty_password",
        &edge_enc.id,
        StatisticKind::MedianMs,
        measure_parse_median_ms(edge_enc)?,
        ThresholdClass::MixedScopedEncryption,
    ));

    Ok(out)
}

pub fn run_private_regression(corpus: &PerfCorpus) -> PerfResult<Vec<PerfMetric>> {
    let mut out = Vec::new();
    for case in corpus.private_cases() {
        let Some(doc) = parse_case_private_opt(case)? else {
            continue;
        };

        out.push(metric(
            "parse",
            &case.id,
            StatisticKind::MedianMs,
            measure_parse_median_ms(case)?,
            ThresholdClass::Core,
        ));

        let extract_ms = measure_extract_blocks_median_ms(&doc)?;
        out.push(metric(
            "extract_blocks",
            &case.id,
            StatisticKind::MedianMs,
            extract_ms,
            ThresholdClass::Core,
        ));

        let blocks = extract_text_blocks(&doc);
        if blocks.is_empty() {
            return Err(format!(
                "private dataset {} extracted zero blocks ({})",
                case.id,
                case.path.display()
            ));
        }
        out.push(metric(
            "index_build",
            &case.id,
            StatisticKind::MedianMs,
            measure_index_build_median_ms(&blocks)?,
            ThresholdClass::Core,
        ));
    }

    Ok(out)
}

pub fn write_baseline(path: &Path, metrics: &[PerfMetric]) -> PerfResult<()> {
    let payload = BaselineFile {
        platform: platform_tag(),
        metrics: metrics.to_vec(),
    };
    write_json(path, &payload)
}

pub fn read_baseline(path: &Path) -> PerfResult<BaselineFile> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read baseline {}: {}", path.display(), err))?;
    serde_json::from_str(&raw)
        .map_err(|err| format!("invalid baseline JSON {}: {}", path.display(), err))
}

pub fn write_metrics_json(path: &Path, metrics: &[PerfMetric]) -> PerfResult<()> {
    write_json(path, metrics)
}

fn write_json<T: Serialize + ?Sized>(path: &Path, value: &T) -> PerfResult<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create directory {}: {}", parent.display(), err))?;
    }
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|err| format!("failed to serialize JSON {}: {}", path.display(), err))?;
    fs::write(path, bytes).map_err(|err| format!("failed to write {}: {}", path.display(), err))
}

pub fn validate_against_baseline(
    current: &[PerfMetric],
    baseline: &[PerfMetric],
    policy: ThresholdPolicy,
    strict_missing: bool,
) -> PerfResult<()> {
    let mut baseline_by_key: HashMap<String, &PerfMetric> = HashMap::new();
    for metric in baseline {
        baseline_by_key.insert(metric.key(), metric);
    }

    let mut missing = Vec::new();
    let mut failures = Vec::new();

    for metric in current {
        let key = metric.key();
        let Some(base) = baseline_by_key.get(&key) else {
            missing.push(key);
            continue;
        };

        let (relative_limit, min_delta_ms) = policy.limits(metric.threshold_class);
        let delta = metric.metric_ms - base.metric_ms;
        let relative_regression = if base.metric_ms <= f64::EPSILON {
            metric.metric_ms > min_delta_ms
        } else {
            metric.metric_ms > base.metric_ms * (1.0 + relative_limit)
        };
        let absolute_regression = delta > min_delta_ms;

        if relative_regression && absolute_regression {
            failures.push(format!(
                "{} regressed: current {:.4}ms vs baseline {:.4}ms (delta +{:.4}ms, limit +{:.0}% and +{:.2}ms)",
                key,
                metric.metric_ms,
                base.metric_ms,
                delta,
                relative_limit * 100.0,
                min_delta_ms,
            ));
        }
    }

    if strict_missing && !missing.is_empty() {
        failures.push(format!(
            "missing baseline metrics for keys: {}",
            missing.join(", ")
        ));
    }

    if failures.is_empty() {
        return Ok(());
    }

    Err(failures.join("\n"))
}

pub fn summarize(metrics: &[PerfMetric], title: &str) {
    println!("{}", title);
    let mut rows = metrics.to_vec();
    rows.sort_by(|a, b| {
        a.scenario_id
            .cmp(&b.scenario_id)
            .then(a.dataset_id.cmp(&b.dataset_id))
    });
    for metric in rows {
        let stat = match metric.stat {
            StatisticKind::MedianMs => "median",
            StatisticKind::P95Ms => "p95",
        };
        println!(
            "  {:30} {:20} {:>7.3}ms ({})",
            metric.scenario_id, metric.dataset_id, metric.metric_ms, stat
        );
    }
}

pub fn benchmark_extract_chars(case: &PdfCase) -> PerfResult<f64> {
    let doc = parse_case_or_err(case)?;
    let mut samples = timed_samples(CORE_WARMUP, CORE_ITERS, || {
        let chars = extract_char_bboxes(&doc);
        black_box(chars.len());
        Ok(())
    })?;
    Ok(median(&mut samples))
}

pub fn benchmark_pipeline_total(case: &PdfCase) -> PerfResult<f64> {
    let mut samples = timed_samples(CORE_WARMUP, CORE_ITERS, || {
        let doc = parse_case(case).map_err(|err| {
            format!(
                "failed to parse {} ({}) during pipeline benchmark: {}",
                case.id,
                case.path.display(),
                err
            )
        })?;
        let blocks = extract_text_blocks(&doc);
        let idx = TextBlockIndex::new(blocks);
        black_box(idx.doc_rect());
        Ok(())
    })?;
    Ok(median(&mut samples))
}
