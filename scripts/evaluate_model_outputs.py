"""Create formal evaluation reports from calibrated prediction CSV files."""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


METRIC_FIELDS = [
    "split",
    "model",
    "calibration",
    "group_by",
    "group_value",
    "n_samples",
    "positive_rate",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "macro_f1",
    "error_precision",
    "error_recall",
    "error_f1",
    "auc",
    "tn",
    "fp",
    "fn",
    "tp",
]

CONFUSION_FIELDS = [
    "split",
    "model",
    "calibration",
    "group_by",
    "group_value",
    "actual_label",
    "predicted_label",
    "count",
]


def score_column(frame: pd.DataFrame) -> str:
    if "proxy_gop_score" in frame.columns:
        return "proxy_gop_score"
    if "gop_score" in frame.columns:
        return "gop_score"
    raise ValueError("Prediction file must include proxy_gop_score or gop_score.")


def safe_auc(y_true: pd.Series, scores: pd.Series) -> float:
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return 0.5


def metrics_for_frame(
    frame: pd.DataFrame,
    group_by: str,
    group_value: str,
) -> dict[str, object]:
    y_true = frame["gold_binary"].astype(int)
    y_pred = frame["prediction"].astype(int)
    scores = frame[score_column(frame)].astype(float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "split": frame["split"].iloc[0],
        "model": frame["model"].iloc[0],
        "calibration": frame.get("calibration", pd.Series(["global_threshold"])).iloc[0],
        "group_by": group_by,
        "group_value": group_value,
        "n_samples": len(frame),
        "positive_rate": round(float(y_true.mean()), 6),
        "accuracy": round(accuracy_score(y_true, y_pred), 6),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 6),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 6),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 6),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 6),
        "error_precision": round(precision_score(y_true, y_pred, pos_label=0, zero_division=0), 6),
        "error_recall": round(recall_score(y_true, y_pred, pos_label=0, zero_division=0), 6),
        "error_f1": round(f1_score(y_true, y_pred, pos_label=0, zero_division=0), 6),
        "auc": round(safe_auc(y_true, scores), 6),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def confusion_rows_for_frame(
    frame: pd.DataFrame,
    group_by: str,
    group_value: str,
) -> list[dict[str, object]]:
    y_true = frame["gold_binary"].astype(int)
    y_pred = frame["prediction"].astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    rows: list[dict[str, object]] = []
    for actual_index, actual_label in enumerate([0, 1]):
        for predicted_index, predicted_label in enumerate([0, 1]):
            rows.append(
                {
                    "split": frame["split"].iloc[0],
                    "model": frame["model"].iloc[0],
                    "calibration": frame.get("calibration", pd.Series(["global_threshold"])).iloc[0],
                    "group_by": group_by,
                    "group_value": group_value,
                    "actual_label": actual_label,
                    "predicted_label": predicted_label,
                    "count": int(matrix[actual_index, predicted_index]),
                }
            )
    return rows


def add_metrics(
    frame: pd.DataFrame,
    metrics: list[dict[str, object]],
    confusion: list[dict[str, object]],
    group_by: str,
) -> None:
    group_columns = ["split", "model"]
    if "calibration" in frame.columns:
        group_columns.append("calibration")
    if group_by != "all":
        group_columns.append(group_by)

    for _, group in frame.groupby(group_columns, dropna=False):
        if group_by == "all":
            group_value = "all"
        else:
            group_value = str(group[group_by].iloc[0])
        metrics.append(metrics_for_frame(group, group_by, group_value))
        confusion.extend(confusion_rows_for_frame(group, group_by, group_value))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, summary: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Formal Test Evaluation",
        "",
        "The table below reports test-split performance for the current calibrated models.",
        "",
        "| model | calibration | n | accuracy | balanced_accuracy | precision | recall | macro_f1 | error_precision | error_recall | error_f1 | auc | tn | fp | fn | tp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| "
            f"{row['model']} | {row['calibration']} | {int(row['n_samples'])} | "
            f"{row['accuracy']:.6f} | {row['balanced_accuracy']:.6f} | "
            f"{row['precision']:.6f} | {row['recall']:.6f} | "
            f"{row['macro_f1']:.6f} | {row['error_precision']:.6f} | "
            f"{row['error_recall']:.6f} | {row['error_f1']:.6f} | "
            f"{row['auc']:.6f} | "
            f"{int(row['tn'])} | {int(row['fp'])} | {int(row['fn'])} | {int(row['tp'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path("reports/proxy_gop_predictions.csv"))
    parser.add_argument("--metrics-output", type=Path, default=Path("reports/formal_eval_metrics.csv"))
    parser.add_argument(
        "--confusion-output",
        type=Path,
        default=Path("reports/formal_eval_confusion_matrix.csv"),
    )
    parser.add_argument("--summary-output", type=Path, default=Path("reports/formal_eval_summary.md"))
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions, encoding="utf-8-sig", keep_default_na=False)
    frame["gold_binary"] = frame["gold_binary"].astype(int)
    frame["prediction"] = frame["prediction"].astype(int)

    metrics: list[dict[str, object]] = []
    confusion: list[dict[str, object]] = []
    add_metrics(frame, metrics, confusion, "all")
    add_metrics(frame, metrics, confusion, "phone_group")
    add_metrics(frame, metrics, confusion, "target_phone")

    write_csv(args.metrics_output, metrics, METRIC_FIELDS)
    write_csv(args.confusion_output, confusion, CONFUSION_FIELDS)

    metrics_df = pd.DataFrame(metrics)
    summary = metrics_df[
        (metrics_df["split"] == "test")
        & (metrics_df["group_by"] == "all")
    ].sort_values(["balanced_accuracy", "macro_f1", "auc"], ascending=False)
    write_summary(args.summary_output, summary)

    print(f"Wrote metrics to {args.metrics_output}")
    print(f"Wrote confusion matrix to {args.confusion_output}")
    print(f"Wrote summary to {args.summary_output}")
    for _, row in summary.iterrows():
        print(
            f"{row['model']} {row['calibration']} test: "
            f"balanced_accuracy={row['balanced_accuracy']} "
            f"macro_f1={row['macro_f1']} auc={row['auc']}"
        )


if __name__ == "__main__":
    main()
