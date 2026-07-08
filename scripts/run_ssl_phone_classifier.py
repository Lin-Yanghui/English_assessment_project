"""Train a phone-segment classifier from SSL embeddings."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


def best_threshold(y_true: pd.Series, scores: list[float]) -> tuple[float, float]:
    best_t = 0.5
    best_score = -1.0
    for step in range(5, 96):
        threshold = step / 100
        y_pred = [1 if score >= threshold else 0 for score in scores]
        score = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if score > best_score:
            best_t = threshold
            best_score = float(score)
    return best_t, best_score


def evaluate(split: str, y_true: pd.Series, scores: list[float], threshold: float) -> dict[str, object]:
    y_pred = [1 if score >= threshold else 0 for score in scores]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = 0.5
    return {
        "split": split,
        "model": "ssl_phone_logreg",
        "n_samples": len(y_true),
        "positive_rate": round(float(y_true.mean()), 6),
        "threshold": round(threshold, 6),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, default=Path("reports/ssl_phone_embeddings.csv"))
    parser.add_argument("--metrics-output", type=Path, default=Path("reports/ssl_phone_classifier_metrics.csv"))
    parser.add_argument("--predictions-output", type=Path, default=Path("reports/ssl_phone_classifier_predictions.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    df = pd.read_csv(args.embeddings, encoding="utf-8-sig", keep_default_na=False)
    embedding_cols = [col for col in df.columns if col.startswith("ssl_")]
    if not embedding_cols:
        raise SystemExit("No ssl_* embedding columns found.")
    df["gold_binary"] = df["gold_binary"].astype(int)

    train = df[df["split"] == "train"].copy()
    dev = df[df["split"] == "dev"].copy()
    test = df[df["split"] == "test"].copy()
    if train.empty or dev.empty or test.empty:
        raise SystemExit("Embeddings must contain train, dev, and test splits.")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(train[embedding_cols], train["gold_binary"])
    classes = list(model.named_steps["model"].classes_)
    positive_index = classes.index(1)

    def scores(frame: pd.DataFrame) -> list[float]:
        return [float(score[positive_index]) for score in model.predict_proba(frame[embedding_cols])]

    dev_scores = scores(dev)
    threshold, _ = best_threshold(dev["gold_binary"], dev_scores)

    metrics = [
        evaluate("dev", dev["gold_binary"], dev_scores, threshold),
        evaluate("test", test["gold_binary"], scores(test), threshold),
    ]

    predictions: list[dict[str, object]] = []
    for split_frame in [dev, test]:
        split_scores = scores(split_frame)
        for (_, row), score in zip(split_frame.iterrows(), split_scores):
            out = row.drop(labels=embedding_cols).to_dict()
            out["ssl_score"] = round(score, 6)
            out["prediction"] = 1 if score >= threshold else 0
            out["confidence"] = round(max(score, 1 - score), 6)
            out["model"] = "ssl_phone_logreg"
            out["threshold"] = threshold
            predictions.append(out)

    write_csv(args.metrics_output, metrics, METRIC_FIELDS)
    write_csv(args.predictions_output, predictions, list(predictions[0].keys()))
    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_dir / "ssl_phone_logreg.joblib")

    print(f"Wrote SSL metrics to {args.metrics_output}")
    print(f"Wrote SSL predictions to {args.predictions_output}")


if __name__ == "__main__":
    main()
