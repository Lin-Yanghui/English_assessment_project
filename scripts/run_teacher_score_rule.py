"""Evaluate the source phone-score rule as a teacher-score upper bound.

The project labels are derived from source phone-level scores. This script keeps
that rule explicit and separate from deployable acoustic models so reports can
distinguish numerical label-reconstruction performance from model performance.
"""

from __future__ import annotations

import argparse
import csv
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
    "n_samples",
    "positive_rate",
    "threshold",
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


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def best_threshold(y_true: pd.Series, scores: pd.Series) -> float:
    best_t = 0.5
    best_key = (-1.0, -1.0)
    candidates = sorted(set(float(value) for value in scores.dropna().unique()))
    grid = sorted(set(candidates + [step / 100 for step in range(0, 301)]))
    for threshold in grid:
        y_pred = (scores >= threshold).astype(int)
        key = (
            f1_score(y_true, y_pred, average="macro", zero_division=0),
            balanced_accuracy_score(y_true, y_pred),
        )
        if key > best_key:
            best_key = key
            best_t = threshold
    return best_t


def evaluate(split: str, y_true: pd.Series, scores: pd.Series, threshold: float) -> dict[str, object]:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = 0.5
    return {
        "split": split,
        "model": "teacher_score_rule",
        "n_samples": len(y_true),
        "positive_rate": round(float(y_true.mean()), 6),
        "threshold": round(float(threshold), 6),
        "accuracy": round(accuracy_score(y_true, y_pred), 6),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 6),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 6),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 6),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 6),
        "error_precision": round(precision_score(y_true, y_pred, pos_label=0, zero_division=0), 6),
        "error_recall": round(recall_score(y_true, y_pred, pos_label=0, zero_division=0), 6),
        "error_f1": round(f1_score(y_true, y_pred, pos_label=0, zero_division=0), 6),
        "auc": round(float(auc), 6),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_teacher_score(df: pd.DataFrame) -> pd.Series:
    score = pd.to_numeric(df.get("source_score", ""), errors="coerce")
    if "gold_three_class" in df.columns:
        label_score = df["gold_three_class"].map(
            {
                "correct": 2.0,
                "acceptable": 1.2,
                "incorrect": 0.0,
            }
        )
        score = score.fillna(label_score)
    if "error_type" in df.columns:
        error_type_score = df["error_type"].astype(str).str.lower().map(
            lambda value: 2.0 if value in {"", "correct", "none"} else 0.0
        )
        score = score.fillna(error_type_score)
    return score.fillna(0.0)


def filter_alignment_quality(df: pd.DataFrame, alignment_quality: str) -> pd.DataFrame:
    if alignment_quality.lower() == "all":
        return df.copy()
    allowed = {
        "" if item.strip().lower() in {"blank", "empty"} else item.strip().lower()
        for item in alignment_quality.split(",")
    }
    return df[df["alignment_quality"].str.lower().isin(allowed)].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("phones_aligned.csv"))
    parser.add_argument("--alignment-quality", default="pass")
    parser.add_argument("--dataset-source", default="SpeechOcean762")
    parser.add_argument("--metrics-output", type=Path, default=Path("reports/teacher_score_rule_metrics.csv"))
    parser.add_argument(
        "--predictions-output", type=Path, default=Path("reports/teacher_score_rule_predictions.csv")
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    if args.dataset_source and args.dataset_source.lower() != "all":
        df = df[df["dataset_source"] == args.dataset_source].copy()
    if "alignment_quality" in df.columns:
        df = filter_alignment_quality(df, args.alignment_quality)
    df["gold_binary"] = df["gold_binary"].astype(int)
    df["teacher_score"] = build_teacher_score(df)

    dev = df[df["split"] == "dev"].copy()
    test = df[df["split"] == "test"].copy()
    if dev.empty or test.empty:
        raise SystemExit("Input must contain dev and test splits.")
    threshold = best_threshold(dev["gold_binary"], dev["teacher_score"])
    metrics = [
        evaluate("dev", dev["gold_binary"], dev["teacher_score"], threshold),
        evaluate("test", test["gold_binary"], test["teacher_score"], threshold),
    ]

    predictions: list[dict[str, object]] = []
    for split_frame in [dev, test]:
        for _, row in split_frame.iterrows():
            score = float(row["teacher_score"])
            out = row.to_dict()
            out["prediction"] = 1 if score >= threshold else 0
            out["confidence"] = round(max(score / 2.0, 1 - score / 2.0), 6)
            out["gop_score"] = score
            out["model"] = "teacher_score_rule"
            out["threshold"] = threshold
            predictions.append(out)

    write_csv(args.metrics_output, metrics, METRIC_FIELDS)
    write_csv(args.predictions_output, predictions, list(predictions[0].keys()))
    print(f"Wrote teacher-score metrics to {args.metrics_output}")
    print(f"Wrote teacher-score predictions to {args.predictions_output}")
    print(f"Selected threshold={threshold}")


if __name__ == "__main__":
    main()
