"""Run the reproducible phase-1 baseline pipeline end to end."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("phones_aligned.csv"))
    parser.add_argument("--alignment-quality", default="pass")
    args = parser.parse_args()

    py = sys.executable
    run(
        [
            py,
            "scripts/run_majority_baseline.py",
            "--input",
            str(args.input),
            "--alignment-quality",
            args.alignment_quality,
            "--metrics-output",
            "reports/majority_baseline_metrics.csv",
            "--predictions-output",
            "reports/majority_baseline_predictions.csv",
        ]
    )
    run(
        [
            py,
            "scripts/run_feature_baseline.py",
            "--input",
            str(args.input),
            "--alignment-quality",
            args.alignment_quality,
            "--metrics-output",
            "reports/feature_baseline_metrics.csv",
            "--predictions-output",
            "reports/feature_baseline_predictions.csv",
            "--model-dir",
            "models",
        ]
    )
    run(
        [
            py,
            "scripts/calibrate_proxy_gop_thresholds.py",
            "--input",
            "reports/feature_baseline_predictions.csv",
            "--thresholds-output",
            "reports/proxy_gop_group_thresholds.csv",
            "--target-thresholds-output",
            "reports/proxy_gop_target_phone_thresholds.csv",
            "--all-thresholds-output",
            "reports/proxy_gop_thresholds.csv",
            "--metrics-output",
            "reports/proxy_gop_metrics.csv",
            "--predictions-output",
            "reports/proxy_gop_predictions.csv",
            "--objective",
            "macro_f1",
        ]
    )
    run(
        [
            py,
            "scripts/run_enhanced_model.py",
            "--input",
            str(args.input),
            "--alignment-quality",
            args.alignment_quality,
            "--metrics-output",
            "reports/enhanced_model_metrics.csv",
            "--predictions-output",
            "reports/enhanced_model_predictions.csv",
            "--model-dir",
            "models",
        ]
    )
    run(
        [
            py,
            "scripts/run_fusion_model.py",
            "--input",
            "reports/proxy_gop_predictions.csv",
            "--metrics-output",
            "reports/fusion_model_metrics.csv",
            "--predictions-output",
            "reports/fusion_model_predictions.csv",
            "--model-dir",
            "models",
        ]
    )
    run(
        [
            py,
            "scripts/evaluate_model_outputs.py",
            "--predictions",
            "reports/proxy_gop_predictions.csv",
            "--metrics-output",
            "reports/formal_eval_metrics.csv",
            "--confusion-output",
            "reports/formal_eval_confusion_matrix.csv",
            "--summary-output",
            "reports/formal_eval_summary.md",
        ]
    )
    run(
        [
            py,
            "scripts/export_prediction_samples.py",
            "--input",
            "reports/proxy_gop_predictions.csv",
            "--output",
            "reports/prediction_samples_100_utterances.csv",
            "--utterance-list-output",
            "reports/prediction_sample_utterances.csv",
            "--model",
            "feature_random_forest",
            "--split",
            "test",
            "--num-utterances",
            "100",
        ]
    )
    run(
        [
            py,
            "scripts/analyze_prediction_errors.py",
            "--input",
            "reports/proxy_gop_predictions.csv",
            "--model",
            "feature_random_forest",
            "--split",
            "test",
            "--cases-output",
            "reports/error_cases.csv",
            "--group-output",
            "reports/error_analysis_by_phone_group.csv",
            "--phone-output",
            "reports/error_analysis_by_target_phone.csv",
            "--summary-output",
            "reports/error_analysis_summary.md",
            "--max-cases-per-type",
            "50",
        ]
    )
    run(
        [
            py,
            "scripts/build_model_comparison.py",
            "--majority-metrics",
            "reports/majority_baseline_metrics.csv",
            "--feature-metrics",
            "reports/feature_baseline_metrics.csv",
            "--proxy-gop-metrics",
            "reports/proxy_gop_metrics.csv",
            "--enhanced-metrics",
            "reports/enhanced_model_metrics.csv",
            "--fusion-metrics",
            "reports/fusion_model_metrics.csv",
            "--output",
            "reports/model_comparison.csv",
            "--summary-output",
            "reports/model_comparison.md",
        ]
    )
    run([py, "scripts/generate_phase1_artifacts.py"])
    run(
        [
            py,
            "scripts/generate_split_artifacts.py",
            "--input",
            str(args.input),
            "--alignment-quality",
            args.alignment_quality,
            "--split-output",
            "reports/split_manifest.csv",
            "--speaker-check-output",
            "reports/speaker_split_check.csv",
        ]
    )


if __name__ == "__main__":
    main()
