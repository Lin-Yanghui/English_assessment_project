"""Train enhanced tabular models for phone-level correctness."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
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
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


CATEGORICAL_FEATURES = [
    "target_phone",
    "phone_group",
    "speaker_gender",
    "native_language",
]
NUMERIC_FEATURES = [
    "duration_ms",
    "phone_index",
    "speaker_age",
]

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


def filter_alignment_quality(df: pd.DataFrame, alignment_quality: str) -> pd.DataFrame:
    if alignment_quality.lower() == "all":
        return df.copy()
    allowed = {
        "" if item.strip().lower() in {"blank", "empty"} else item.strip().lower()
        for item in alignment_quality.split(",")
    }
    return df[df["alignment_quality"].str.lower().isin(allowed)].copy()


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
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
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
        "enhanced_extra_trees": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        min_samples_leaf=8,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "enhanced_hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.06,
                        max_iter=250,
                        min_samples_leaf=30,
                        l2_regularization=0.02,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def best_threshold(y_true: pd.Series, scores: list[float], objective: str) -> tuple[float, float]:
    best_t = 0.5
    best_score = -1.0
    for step in range(5, 96):
        threshold = step / 100
        y_pred = [1 if score >= threshold else 0 for score in scores]
        if objective == "macro_f1":
            score = f1_score(y_true, y_pred, average="macro", zero_division=0)
        else:
            score = balanced_accuracy_score(y_true, y_pred)
        if score > best_score:
            best_score = float(score)
            best_t = threshold
    return best_t, best_score


def safe_auc(y_true: pd.Series, scores: list[float]) -> float:
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return 0.5


def evaluate(
    split: str,
    model_name: str,
    y_true: pd.Series,
    scores: list[float],
    threshold: float,
) -> dict[str, object]:
    y_pred = [1 if score >= threshold else 0 for score in scores]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
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
        "error_precision": round(precision_score(y_true, y_pred, pos_label=0, zero_division=0), 6),
        "error_recall": round(recall_score(y_true, y_pred, pos_label=0, zero_division=0), 6),
        "error_f1": round(f1_score(y_true, y_pred, pos_label=0, zero_division=0), 6),
        "auc": round(safe_auc(y_true, scores), 6),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def positive_scores(model: Pipeline, frame: pd.DataFrame) -> list[float]:
    estimator = model.named_steps["model"]
    classes = list(estimator.classes_)
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
    scores: list[float],
    threshold: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (_, row), score in zip(frame.iterrows(), scores):
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
    parser.add_argument("--metrics-output", type=Path, default=Path("reports/enhanced_model_metrics.csv"))
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("reports/enhanced_model_predictions.csv"),
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--objective", choices=["macro_f1", "balanced_accuracy"], default="macro_f1")
    parser.add_argument("--alignment-quality", default="pass")
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    df = filter_alignment_quality(df, args.alignment_quality)
    if df.empty:
        raise SystemExit("No rows available after filtering.")

    df["gold_binary"] = df["gold_binary"].astype(int)
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce")

    train = df[df["split"] == "train"].copy()
    dev = df[df["split"] == "dev"].copy()
    test = df[df["split"] == "test"].copy()
    if train.empty or dev.empty or test.empty:
        raise SystemExit("Input must contain train, dev, and test splits.")

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    metrics: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    args.model_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in candidate_models(args.random_state).items():
        model.fit(train[feature_cols], train["gold_binary"])
        dev_scores = positive_scores(model, dev[feature_cols])
        threshold, _ = best_threshold(dev["gold_binary"], dev_scores, args.objective)

        for split_name, split_frame in [("dev", dev), ("test", test)]:
            split_scores = positive_scores(model, split_frame[feature_cols])
            metrics.append(
                evaluate(split_name, model_name, split_frame["gold_binary"], split_scores, threshold)
            )
            predictions.extend(prediction_rows(split_frame, model_name, split_scores, threshold))

        joblib.dump(model, args.model_dir / f"{model_name}.joblib")

    write_csv(args.metrics_output, metrics, METRIC_FIELDS)
    prediction_fields = list(df.columns) + ["prediction", "confidence", "gop_score", "model", "threshold"]
    write_csv(args.predictions_output, predictions, prediction_fields)

    print(f"Wrote enhanced metrics to {args.metrics_output}")
    print(f"Wrote enhanced predictions to {args.predictions_output}")
    for row in metrics:
        print(
            f"{row['model']} {row['split']}: "
            f"balanced_accuracy={row['balanced_accuracy']} "
            f"macro_f1={row['macro_f1']} auc={row['auc']}"
        )


if __name__ == "__main__":
    main()
