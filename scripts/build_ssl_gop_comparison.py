"""Compare SSL phone classifier metrics against proxy GOP metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load(path: Path, family: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["model_family"] = family
    if "calibration" not in df.columns:
        df["calibration"] = "global_threshold"
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gop-metrics", type=Path, default=Path("reports/proxy_gop_metrics.csv"))
    parser.add_argument(
        "--acoustic-gop-metrics",
        type=Path,
        default=Path("reports/acoustic_gop_metrics.csv"),
    )
    parser.add_argument("--ssl-metrics", type=Path, default=Path("reports/ssl_phone_classifier_metrics.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/ssl_vs_gop_comparison.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("reports/ssl_vs_gop_comparison.md"))
    args = parser.parse_args()

    frames = [load(args.gop_metrics, "proxy_gop")]
    if args.acoustic_gop_metrics.exists():
        frames.append(load(args.acoustic_gop_metrics, "acoustic_gop"))
    if args.ssl_metrics.exists():
        frames.append(load(args.ssl_metrics, "ssl_phone_classifier"))
    comparison = pd.concat(frames, ignore_index=True)
    comparison = comparison[comparison["split"].isin(["dev", "test"])].copy()
    comparison = comparison.sort_values(
        ["split", "balanced_accuracy", "macro_f1", "auc"],
        ascending=[True, False, False, False],
    )
    comparison["rank"] = comparison.groupby("split").cumcount() + 1
    cols = [
        "rank",
        "split",
        "model_family",
        "model",
        "calibration",
        "n_samples",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "macro_f1",
        "error_precision",
        "error_recall",
        "error_f1",
        "auc",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison[cols].to_csv(args.output, index=False, encoding="utf-8-sig")

    test = comparison[comparison["split"] == "test"]
    lines = [
        "# SSL vs GOP Comparison",
        "",
        "This report compares proxy GOP threshold baselines against SSL phone-segment classifiers when SSL metrics are available.",
        "",
        "| rank | family | model | calibration | accuracy | balanced_accuracy | macro_f1 | error_f1 | auc |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in test.iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['model_family']} | {row['model']} | {row['calibration']} | "
            f"{row['accuracy']:.6f} | {row['balanced_accuracy']:.6f} | {row['macro_f1']:.6f} | "
            f"{row['error_f1']:.6f} | {row['auc']:.6f} |"
        )
    args.summary_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote SSL/GOP comparison to {args.output}")


if __name__ == "__main__":
    main()
