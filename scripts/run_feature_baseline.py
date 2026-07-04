"""Train simple feature baselines on the phone-level manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_FEATURES = ["target_phone", "phone_group"]
NUMERIC_FEATURES = ["duration_ms", "phone_index"]

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
    "auc",
    "tn",
    "fp",
    "fn",
    "tp",
]


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def candidate_models(random_state: int) -> dict[str, Pipeline]:
    return {
        "feature_logreg": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "feature_random_forest": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=10,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def find_best_threshold(y_true: pd.Series, positive_scores: list[float]) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0
    for step in range(5, 96):
        threshold = step / 100
        y_pred = [1 if score >= threshold else 0 for score in positive_scores]
        score = balanced_accuracy_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold, best_score


def evaluate(
    split: str,
    model_name: str,
    y_true: pd.Series,
    positive_scores: list[float],
    threshold: float,
) -> dict[str, int | float | str]:
    y_pred = [1 if score >= threshold else 0 for score in positive_scores]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_true, positive_scores)
    except ValueError:
        auc = 0.5
    return {
        "split": split,
        "model": model_name,
        "n_samples": len(y_true),
        "positive_rate": round(float(y_true.mean()), 6),
        "threshold": round(threshold, 6),
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


def positive_class_scores(model: Pipeline, frame: pd.DataFrame) -> list[float]:
    classes = list(model.named_steps["model"].classes_)
    positive_index = classes.index(1)
    return [float(score[positive_index]) for score in model.predict_proba(frame)]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prediction_rows(
    frame: pd.DataFrame,
    model_name: str,
    positive_scores: list[float],
    threshold: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (_, row), score in zip(frame.iterrows(), positive_scores):
        out = row.to_dict()
        out["prediction"] = 1 if score >= threshold else 0
        out["confidence"] = round(max(score, 1 - score), 6)
        out["gop_score"] = round(score, 6)
        out["model"] = model_name
        out["threshold"] = threshold
        rows.append(out)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("phones_aligned.csv"))
    parser.add_argument("--metrics-output", type=Path, default=Path("reports/feature_baseline_metrics.csv"))
    parser.add_argument(
        "--predictions-output", type=Path, default=Path("reports/feature_baseline_predictions.csv")
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--alignment-quality",
        default="pass",
        help="Filter by alignment_quality. Use 'all' to disable filtering.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    if args.alignment_quality.lower() != "all":
        if "alignment_quality" not in df.columns:
            raise SystemExit("--alignment-quality was set, but input has no alignment_quality column.")
        df = df[df["alignment_quality"].str.lower() == args.alignment_quality.lower()].copy()
    if df.empty:
        raise SystemExit(
            f"No rows found after alignment_quality={args.alignment_quality!r} filtering."
        )
    df["gold_binary"] = df["gold_binary"].astype(int)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("").astype(str)

    train = df[df["split"] == "train"].copy()
    dev = df[df["split"] == "dev"].copy()
    test = df[df["split"] == "test"].copy()
    if train.empty or dev.empty or test.empty:
        raise SystemExit("Manifest must contain train, dev, and test splits.")

    metrics: list[dict[str, object]] = []
    all_prediction_rows: list[dict[str, object]] = []
    args.model_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in candidate_models(args.random_state).items():
        model.fit(train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train["gold_binary"])
        dev_scores = positive_class_scores(model, dev[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
        threshold, _ = find_best_threshold(dev["gold_binary"], dev_scores)

        for split_name, split_frame in [("dev", dev), ("test", test)]:
            scores = positive_class_scores(model, split_frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
            metrics.append(
                evaluate(split_name, model_name, split_frame["gold_binary"], scores, threshold)
            )
            all_prediction_rows.extend(prediction_rows(split_frame, model_name, scores, threshold))

        joblib.dump(model, args.model_dir / f"{model_name}.joblib")

    write_csv(args.metrics_output, metrics, METRIC_FIELDS)
    prediction_fields = list(df.columns) + ["prediction", "confidence", "gop_score", "model", "threshold"]
    write_csv(args.predictions_output, all_prediction_rows, prediction_fields)

    print(f"Wrote metrics to {args.metrics_output}")
    print(f"Wrote predictions to {args.predictions_output}")
    print(f"Wrote models to {args.model_dir}")
    for row in metrics:
        print(
            f"{row['model']} {row['split']}: "
            f"balanced_accuracy={row['balanced_accuracy']} "
            f"macro_f1={row['macro_f1']} auc={row['auc']} threshold={row['threshold']}"
        )


if __name__ == "__main__":
    main()
