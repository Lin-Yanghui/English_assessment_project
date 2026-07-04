"""Run a majority-class baseline for the phone correctness CSV."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

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
    "predicted_class",
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
]


def read_rows(path: Path, alignment_quality: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if alignment_quality.lower() != "all":
        rows = [
            row
            for row in rows
            if row.get("alignment_quality", "").lower() == alignment_quality.lower()
        ]
    return rows


def labels_for_split(rows: list[dict[str, str]], split: str) -> list[int]:
    labels = []
    for row in rows:
        if row["split"] == split:
            labels.append(int(row["gold_binary"]))
    return labels


def evaluate_split(split: str, y_true: list[int], predicted_class: int) -> dict[str, str | int | float]:
    if not y_true:
        raise ValueError(f"No rows found for split: {split}")

    y_pred = [predicted_class] * len(y_true)
    y_score = [float(predicted_class)] * len(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = 0.5

    return {
        "split": split,
        "model": "majority_class",
        "n_samples": len(y_true),
        "positive_rate": round(sum(y_true) / len(y_true), 6),
        "predicted_class": predicted_class,
        "accuracy": round(accuracy_score(y_true, y_pred), 6),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 6),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 6),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 6),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 6),
        "auc": round(auc, 6),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def write_metrics(path: Path, metrics: list[dict[str, str | int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(metrics)


def write_predictions(
    input_rows: list[dict[str, str]], path: Path, predicted_class: int, splits: set[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(input_rows[0].keys()) + ["prediction", "confidence", "gop_score", "model"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in input_rows:
            if row["split"] not in splits:
                continue
            out = dict(row)
            out["prediction"] = predicted_class
            out["confidence"] = 1.0
            out["gop_score"] = float(predicted_class)
            out["model"] = "majority_class"
            writer.writerow(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("phones_aligned.csv"))
    parser.add_argument("--metrics-output", type=Path, default=Path("reports/majority_baseline_metrics.csv"))
    parser.add_argument(
        "--predictions-output", type=Path, default=Path("reports/majority_baseline_predictions.csv")
    )
    parser.add_argument("--eval-splits", nargs="+", default=["dev", "test"])
    parser.add_argument(
        "--alignment-quality",
        default="pass",
        help="Filter by alignment_quality. Use 'all' to disable filtering.",
    )
    args = parser.parse_args()

    rows = read_rows(args.input, args.alignment_quality)
    if not rows:
        raise SystemExit(
            f"No rows found after alignment_quality={args.alignment_quality!r} filtering."
        )
    train_labels = labels_for_split(rows, "train")
    if not train_labels:
        raise SystemExit("No train rows found. Cannot choose a majority class.")

    majority_class = Counter(train_labels).most_common(1)[0][0]
    metrics = [
        evaluate_split(split, labels_for_split(rows, split), majority_class)
        for split in args.eval_splits
    ]

    write_metrics(args.metrics_output, metrics)
    write_predictions(rows, args.predictions_output, majority_class, set(args.eval_splits))

    print(f"Train majority class: {majority_class}")
    print(f"Wrote metrics to {args.metrics_output}")
    print(f"Wrote predictions to {args.predictions_output}")
    for row in metrics:
        print(
            f"{row['split']}: accuracy={row['accuracy']} "
            f"balanced_accuracy={row['balanced_accuracy']} macro_f1={row['macro_f1']}"
        )


if __name__ == "__main__":
    main()
