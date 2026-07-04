"""Build a compact model-comparison table from experiment metric files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = [
    "rank",
    "split",
    "model",
    "calibration",
    "n_samples",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "macro_f1",
    "auc",
    "tn",
    "fp",
    "fn",
    "tp",
    "source_file",
]


def load_metrics(path: Path, calibration: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "calibration" not in df.columns:
        df["calibration"] = calibration or "global_threshold"
    if "threshold" in df.columns and calibration is None:
        df["calibration"] = "global_threshold"
    df["source_file"] = str(path)
    return df


def write_markdown(path: Path, table: pd.DataFrame) -> None:
    test = table[table["split"] == "test"].copy()
    lines = [
        "# Model Comparison",
        "",
        "Sorted by test Balanced Accuracy, then Macro-F1 and AUC.",
        "",
        "| rank | model | calibration | accuracy | balanced_accuracy | macro_f1 | auc | tn | fp | fn | tp |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in test.iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['model']} | {row['calibration']} | "
            f"{row['accuracy']:.6f} | {row['balanced_accuracy']:.6f} | "
            f"{row['macro_f1']:.6f} | {row['auc']:.6f} | "
            f"{int(row['tn'])} | {int(row['fp'])} | {int(row['fn'])} | {int(row['tp'])} |"
        )

    best = test.iloc[0]
    lines.extend(
        [
            "",
            "## Current Best",
            "",
            (
                f"- Best test model by Balanced Accuracy: {best['model']} "
                f"({best['calibration']}), Balanced Accuracy={best['balanced_accuracy']:.6f}, "
                f"Macro-F1={best['macro_f1']:.6f}, AUC={best['auc']:.6f}."
            ),
            "- Majority-class accuracy is high because the dataset is imbalanced; Balanced Accuracy and Macro-F1 are the primary comparison metrics.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--majority-metrics",
        type=Path,
        default=Path("reports/majority_baseline_metrics.csv"),
    )
    parser.add_argument(
        "--feature-metrics",
        type=Path,
        default=Path("reports/feature_baseline_metrics.csv"),
    )
    parser.add_argument(
        "--proxy-gop-metrics",
        type=Path,
        default=Path("reports/proxy_gop_metrics.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("reports/model_comparison.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("reports/model_comparison.md"))
    args = parser.parse_args()

    frames = [
        load_metrics(args.majority_metrics, "majority_class"),
        load_metrics(args.feature_metrics, "global_threshold"),
        load_metrics(args.proxy_gop_metrics, "phone_group_threshold"),
    ]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["split"].isin(["dev", "test"])].copy()
    combined = combined.sort_values(
        ["split", "balanced_accuracy", "macro_f1", "auc"],
        ascending=[True, False, False, False],
    )
    combined["rank"] = combined.groupby("split").cumcount() + 1

    output = combined[OUTPUT_COLUMNS].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    write_markdown(args.summary_output, output)

    print(f"Wrote comparison table to {args.output}")
    print(f"Wrote summary to {args.summary_output}")
    test_best = output[output["split"] == "test"].iloc[0]
    print(
        f"Best test model: {test_best['model']} ({test_best['calibration']}), "
        f"balanced_accuracy={test_best['balanced_accuracy']}, "
        f"macro_f1={test_best['macro_f1']}, auc={test_best['auc']}"
    )


if __name__ == "__main__":
    main()
