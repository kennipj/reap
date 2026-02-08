mod perf_support;

use perf_support::{
    PerfCorpus, ThresholdPolicy, committed_baseline_path, private_baseline_path, read_baseline,
    run_committed_regression, run_private_regression, should_record_baseline, summarize,
    validate_against_baseline, write_baseline,
};

#[test]
fn perf_regression_baselines() {
    if cfg!(debug_assertions) {
        println!("skipping perf_regression in debug profile; run with --release");
        return;
    }

    let corpus = PerfCorpus::load_default().expect("failed to load performance corpus");
    let record = should_record_baseline();

    let committed_metrics =
        run_committed_regression(&corpus).expect("failed to run committed regression scenarios");
    summarize(&committed_metrics, "committed performance metrics:");

    let committed_path = committed_baseline_path();
    if record {
        write_baseline(&committed_path, &committed_metrics)
            .expect("failed to write committed baseline");
        println!(
            "recorded committed baseline at {}",
            committed_path.display()
        );
    } else {
        let committed_baseline = read_baseline(&committed_path)
            .unwrap_or_else(|err| panic!("failed to load committed baseline: {}", err));
        validate_against_baseline(
            &committed_metrics,
            &committed_baseline.metrics,
            ThresholdPolicy::default(),
            true,
        )
        .unwrap_or_else(|err| panic!("performance regression detected:\n{}", err));
    }

    let private_metrics =
        run_private_regression(&corpus).expect("failed to run private regression scenarios");
    if private_metrics.is_empty() {
        println!("no private benchmark datasets detected");
        return;
    }

    summarize(&private_metrics, "private performance metrics:");

    let private_path = private_baseline_path();
    if record {
        write_baseline(&private_path, &private_metrics).expect("failed to write private baseline");
        println!("recorded private baseline at {}", private_path.display());
    } else if private_path.exists() {
        let private_baseline =
            read_baseline(&private_path).expect("failed to load private baseline JSON");
        validate_against_baseline(
            &private_metrics,
            &private_baseline.metrics,
            ThresholdPolicy::default(),
            false,
        )
        .unwrap_or_else(|err| panic!("private performance regression detected:\n{}", err));
    } else {
        println!(
            "private datasets found, but no private baseline at {}. run with REAP_PERF_RECORD_BASELINE=1 to create it.",
            private_path.display()
        );
    }
}
