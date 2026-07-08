"""Train fusion models from baseline scores and phone-level metadata."""

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


BASE_KEY_COLUMNS = ["utterance_id", "target_phone", "phone_index", "split"]
CATEGORICAL_FEATURES = ["target_phone", "phone_group"]
BASE_NUMERIC_FEATURES = [
    "duration_ms",
    "phone_index",
    "proxy_gop_score",
    "group_threshold",
    "score_margin",
    "feature_logreg_score",
    "feature_random_forest_score",
]
SSL_PREFIX = "ssl_"

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
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_training_frame(predictions_path: Path, ssl_embeddings_path: Path | None = None) -> pd.DataFrame:
    predictions = pd.read_csv(predictions_path, encoding="utf-8-sig", keep_default_na=False)
    if "calibration" in predictions.columns:
        preferred = predictions[predictions["calibration"] == "phone_group_threshold"].copy()
        if not preferred.empty:
            predictions = preferred
    predictions["gold_binary"] = predictions["gold_binary"].astype(int)
    predictions["proxy_gop_score"] = predictions["proxy_gop_score"].astype(float)
    predictions["group_threshold"] = predictions["group_threshold"].astype(float)

    base = predictions[predictions["model"] == "feature_random_forest"].copy()
    if base.empty:
        raise SystemExit("Fusion input must include feature_random_forest rows.")
    base["score_margin"] = base["proxy_gop_score"] - base["group_threshold"]

    score_table = predictions.pivot_table(
        index=BASE_KEY_COLUMNS,
        columns="model",
        values="proxy_gop_score",
        aggfunc="first",
    ).reset_index()
    score_table = score_table.rename(
        columns={
            "feature_logreg": "feature_logreg_score",
            "feature_random_forest": "feature_random_forest_score",
        }
    )

    frame = base.merge(score_table, on=BASE_KEY_COLUMNS, how="left")
    for col in ["feature_logreg_score", "feature_random_forest_score"]:
        if col not in frame.columns:
            frame[col] = frame["proxy_gop_score"]
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(frame["proxy_gop_score"])

    if ssl_embeddings_path and ssl_embeddings_path.exists():
        ssl_header = pd.read_csv(ssl_embeddings_path, encoding="utf-8-sig", nrows=0).columns
        ssl_cols = [col for col in ssl_header if col.startswith(SSL_PREFIX)]
        if ssl_cols:
            ssl_df = pd.read_csv(
                ssl_embeddings_path,
                encoding="utf-8-sig",
                usecols=BASE_KEY_COLUMNS + ssl_cols,
                keep_default_na=False,
            )
            frame = frame.merge(ssl_df, on=BASE_KEY_COLUMNS, how="left")
    return frame


def build_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
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
            ("num", numeric, numeric_features),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def candidate_models(random_state: int, numeric_features: list[str]) -> dict[str, Pipeline]:
    return {
        "fusion_logreg": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(numeric_features)),
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
        "fusion_random_forest": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(numeric_features)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=8,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
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


def positive_scores(model: Pipeline, frame: pd.DataFrame) -> list[float]:
    estimator = model.named_steps["model"]
    classes = list(estimator.classes_)
    positive_index = classes.index(1)
    return [float(score[positive_index]) for score in model.predict_proba(frame)]


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
    parser.add_argument("--input", type=Path, default=Path("reports/proxy_gop_predictions.csv"))
    parser.add_argument("--ssl-embeddings", type=Path, default=Path("reports/ssl_phone_embeddings.csv"))
    parser.add_argument("--metrics-output", type=Path, default=Path("reports/fusion_model_metrics.csv"))
    parser.add_argument("--predictions-output", type=Path, default=Path("reports/fusion_model_predictions.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--objective", choices=["macro_f1", "balanced_accuracy"], default="macro_f1")
    args = parser.parse_args()

    frame = build_training_frame(args.input, args.ssl_embeddings)
    ssl_features = [col for col in frame.columns if col.startswith(SSL_PREFIX)]
    numeric_features = BASE_NUMERIC_FEATURES + ssl_features
    for col in CATEGORICAL_FEATURES:
        frame[col] = frame[col].fillna("").astype(str)
    for col in numeric_features:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    train = frame[frame["split"] == "train"].copy()
    dev = frame[frame["split"] == "dev"].copy()
    test = frame[frame["split"] == "test"].copy()
    if train.empty:
        train = frame[frame["split"].isin(["dev"])].copy()
    if train.empty or dev.empty or test.empty:
        raise SystemExit("Fusion input must contain dev and test splits, and preferably train.")

    feature_cols = numeric_features + CATEGORICAL_FEATURES
    metrics: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    args.model_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in candidate_models(args.random_state, numeric_features).items():
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
    prediction_base_fields = [col for col in frame.columns if not col.startswith(SSL_PREFIX)]
    prediction_fields = prediction_base_fields + ["prediction", "confidence", "gop_score", "model", "threshold"]
    write_csv(args.predictions_output, predictions, prediction_fields)

    print(f"Wrote fusion metrics to {args.metrics_output}")
    print(f"Wrote fusion predictions to {args.predictions_output}")
    print(f"SSL embedding features used: {len(ssl_features)}")
    for row in metrics:
        print(
            f"{row['model']} {row['split']}: "
            f"balanced_accuracy={row['balanced_accuracy']} "
            f"macro_f1={row['macro_f1']} auc={row['auc']}"
        )


if __name__ == "__main__":
    main()
