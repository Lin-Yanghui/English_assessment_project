"""Train error-focused fusion models for the phase-plan acceptance target.

This model predicts the error class directly, then converts it back to the
project's gold_binary convention where 1 means correct/acceptable and 0 means
incorrect. Thresholds are selected on dev with the phase-plan constraint:
error precision >= 0.40, maximizing error recall.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
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


KEY_COLUMNS = ["utterance_id", "target_phone", "phone_index", "split"]
SSL_PREFIX = "ssl_"
MIN_ERROR_PRECISION = 0.40
MIN_ERROR_RECALL = 0.50

BASE_CATEGORICAL = [
    "target_phone",
    "phone_group",
    "dataset_source",
    "word",
    "native_language",
]

BASE_NUMERIC = [
    "duration_ms",
    "phone_index",
    "word_index",
    "speaker_age",
    "phone_train_error_rate",
    "phone_group_train_error_rate",
    "word_phone_train_error_rate",
    "dataset_phone_train_error_rate",
    "feature_logreg_score",
    "feature_random_forest_score",
]

METRIC_FIELDS = [
    "split",
    "model",
    "calibration",
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
    "meets_plan_minimum",
]


def filter_alignment_quality(df: pd.DataFrame, alignment_quality: str) -> pd.DataFrame:
    if alignment_quality.lower() == "all":
        return df.copy()
    allowed = {
        "" if item.strip().lower() in {"blank", "empty"} else item.strip().lower()
        for item in alignment_quality.split(",")
    }
    return df[df["alignment_quality"].str.lower().isin(allowed)].copy()


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def add_smoothed_prior(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    group_cols: list[str],
    out_col: str,
    smoothing: float = 20.0,
) -> pd.DataFrame:
    global_error_rate = float((1 - train["gold_binary"].astype(int)).mean())
    prior = (
        train.assign(error_label=1 - train["gold_binary"].astype(int))
        .groupby(group_cols, dropna=False)["error_label"]
        .agg(["sum", "count"])
        .reset_index()
    )
    prior[out_col] = (prior["sum"] + smoothing * global_error_rate) / (prior["count"] + smoothing)
    merged = frame.merge(prior[group_cols + [out_col]], on=group_cols, how="left")
    merged[out_col] = pd.to_numeric(merged[out_col], errors="coerce").fillna(global_error_rate)
    return merged


def load_feature_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=KEY_COLUMNS)
    df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    for col in KEY_COLUMNS:
        df[col] = df[col].astype(str)
    score_table = df.pivot_table(
        index=KEY_COLUMNS,
        columns="model",
        values="gop_score",
        aggfunc="first",
    ).reset_index()
    score_table = score_table.rename(
        columns={
            "feature_logreg": "feature_logreg_score",
            "feature_random_forest": "feature_random_forest_score",
        }
    )
    return score_table


def load_frame(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.input, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    df = filter_alignment_quality(df, args.alignment_quality)
    if df.empty:
        raise SystemExit("No rows available after filtering.")

    for col in KEY_COLUMNS:
        df[col] = df[col].astype(str)
    df["gold_binary"] = pd.to_numeric(df["gold_binary"], errors="coerce").astype(int)

    train = df[df["split"] == "train"].copy()
    if train.empty:
        raise SystemExit("Train split is required for acceptance error fusion.")

    for cols, out_col in [
        (["target_phone"], "phone_train_error_rate"),
        (["phone_group"], "phone_group_train_error_rate"),
        (["word", "target_phone"], "word_phone_train_error_rate"),
        (["dataset_source", "target_phone"], "dataset_phone_train_error_rate"),
    ]:
        df = add_smoothed_prior(df, train, cols, out_col)

    score_table = load_feature_scores(args.feature_predictions)
    if not score_table.empty:
        df = df.merge(score_table, on=KEY_COLUMNS, how="left")

    if args.ssl_embeddings.exists():
        ssl_header = pd.read_csv(args.ssl_embeddings, encoding="utf-8-sig", nrows=0).columns
        ssl_cols = [col for col in ssl_header if col.startswith(SSL_PREFIX)]
        usecols = KEY_COLUMNS + ssl_cols
        ssl_df = pd.read_csv(args.ssl_embeddings, encoding="utf-8-sig", usecols=usecols, keep_default_na=False)
        for col in KEY_COLUMNS:
            ssl_df[col] = ssl_df[col].astype(str)
        df = df.merge(ssl_df, on=KEY_COLUMNS, how="left")
    return df


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_features),
            ("cat", categorical, categorical_features),
        ]
    )


def candidate_models(
    random_state: int,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Pipeline]:
    return {
        "acceptance_error_sgd": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(numeric_features, categorical_features)),
                (
                    "model",
                    SGDClassifier(
                        loss="log_loss",
                        alpha=0.00003,
                        penalty="elasticnet",
                        l1_ratio=0.05,
                        class_weight={0: 1.0, 1: 14.0},
                        max_iter=1000,
                        tol=1e-4,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "acceptance_error_extra_trees": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(numeric_features, categorical_features)),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=180,
                        min_samples_leaf=6,
                        max_features="sqrt",
                        class_weight={0: 1.0, 1: 18.0},
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def error_scores(model: Pipeline, frame: pd.DataFrame) -> list[float]:
    estimator = model.named_steps["model"]
    classes = list(estimator.classes_)
    error_index = classes.index(1)
    return [float(score[error_index]) for score in model.predict_proba(frame)]


def safe_auc(y_true_correct: pd.Series, correct_scores: list[float]) -> float:
    try:
        return float(roc_auc_score(y_true_correct, correct_scores))
    except ValueError:
        return 0.5


def evaluate(
    split: str,
    model_name: str,
    calibration: str,
    y_true_correct: pd.Series,
    error_prob: list[float],
    threshold: float,
) -> dict[str, object]:
    y_pred_correct = [0 if score >= threshold else 1 for score in error_prob]
    correct_scores = [1 - score for score in error_prob]
    tn, fp, fn, tp = confusion_matrix(y_true_correct, y_pred_correct, labels=[0, 1]).ravel()
    error_precision = precision_score(y_true_correct, y_pred_correct, pos_label=0, zero_division=0)
    error_recall = recall_score(y_true_correct, y_pred_correct, pos_label=0, zero_division=0)
    return {
        "split": split,
        "model": model_name,
        "calibration": calibration,
        "n_samples": len(y_true_correct),
        "positive_rate": round(float(y_true_correct.mean()), 6),
        "threshold": round(threshold, 6),
        "accuracy": round(accuracy_score(y_true_correct, y_pred_correct), 6),
        "balanced_accuracy": round(balanced_accuracy_score(y_true_correct, y_pred_correct), 6),
        "precision": round(precision_score(y_true_correct, y_pred_correct, zero_division=0), 6),
        "recall": round(recall_score(y_true_correct, y_pred_correct, zero_division=0), 6),
        "macro_f1": round(f1_score(y_true_correct, y_pred_correct, average="macro", zero_division=0), 6),
        "error_precision": round(error_precision, 6),
        "error_recall": round(error_recall, 6),
        "error_f1": round(f1_score(y_true_correct, y_pred_correct, pos_label=0, zero_division=0), 6),
        "auc": round(safe_auc(y_true_correct, correct_scores), 6),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "meets_plan_minimum": int(error_precision >= MIN_ERROR_PRECISION and error_recall >= MIN_ERROR_RECALL),
    }


def choose_threshold(y_true_correct: pd.Series, error_prob: list[float]) -> float:
    best_threshold = 0.5
    best_key = (-1, -1.0, -1.0, -1.0)
    for step in range(1, 100):
        threshold = step / 100
        y_pred_correct = [0 if score >= threshold else 1 for score in error_prob]
        error_precision = precision_score(y_true_correct, y_pred_correct, pos_label=0, zero_division=0)
        error_recall = recall_score(y_true_correct, y_pred_correct, pos_label=0, zero_division=0)
        error_f1 = f1_score(y_true_correct, y_pred_correct, pos_label=0, zero_division=0)
        passes_precision = error_precision >= MIN_ERROR_PRECISION
        key = (
            int(passes_precision),
            error_recall if passes_precision else error_precision,
            error_f1,
            balanced_accuracy_score(y_true_correct, y_pred_correct),
        )
        if key > best_key:
            best_key = key
            best_threshold = threshold
    return best_threshold


def choose_group_thresholds(dev: pd.DataFrame, error_prob: list[float]) -> dict[str, float]:
    work = dev[["phone_group", "gold_binary"]].copy()
    work["error_prob"] = error_prob
    thresholds: dict[str, float] = {}
    global_threshold = choose_threshold(work["gold_binary"], work["error_prob"].tolist())
    for phone_group, group in work.groupby("phone_group"):
        if len(group) < 200 or int((group["gold_binary"] == 0).sum()) < 20:
            thresholds[str(phone_group)] = global_threshold
        else:
            thresholds[str(phone_group)] = choose_threshold(group["gold_binary"], group["error_prob"].tolist())
    return thresholds


def prediction_rows(frame: pd.DataFrame, model_name: str, error_prob: list[float], thresholds: list[float], calibration: str) -> list[dict[str, object]]:
    rows = []
    base_cols = [
        "utterance_id",
        "speaker_id",
        "target_phone",
        "phone_group",
        "phone_index",
        "split",
        "dataset_source",
        "audio_path",
        "gold_binary",
    ]
    for (_, row), score, threshold in zip(frame.iterrows(), error_prob, thresholds):
        out = {col: row.get(col, "") for col in base_cols}
        out["error_probability"] = round(score, 6)
        out["prediction"] = 0 if score >= threshold else 1
        out["confidence"] = round(max(score, 1 - score), 6)
        out["model"] = model_name
        out["calibration"] = calibration
        out["threshold"] = round(threshold, 6)
        rows.append(out)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("phones.csv"))
    parser.add_argument("--ssl-embeddings", type=Path, default=Path("reports/combined_ssl_phone_embeddings.csv"))
    parser.add_argument("--feature-predictions", type=Path, default=Path("reports/combined_feature_baseline_predictions.csv"))
    parser.add_argument("--metrics-output", type=Path, default=Path("reports/combined_acceptance_error_fusion_metrics.csv"))
    parser.add_argument("--predictions-output", type=Path, default=Path("reports/combined_acceptance_error_fusion_predictions.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/combined"))
    parser.add_argument("--alignment-quality", default="pass,blank")
    parser.add_argument("--max-ssl-features", type=int, default=96)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    frame = load_frame(args)
    ssl_features = [col for col in frame.columns if col.startswith(SSL_PREFIX)]
    if args.max_ssl_features > 0:
        ssl_features = ssl_features[: args.max_ssl_features]
    numeric_features = [col for col in BASE_NUMERIC + ssl_features if col in frame.columns]
    categorical_features = [col for col in BASE_CATEGORICAL if col in frame.columns]
    for col in numeric_features:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in categorical_features:
        frame[col] = frame[col].fillna("").astype(str)

    train = frame[frame["split"] == "train"].copy()
    dev = frame[frame["split"] == "dev"].copy()
    test = frame[frame["split"] == "test"].copy()
    if train.empty or dev.empty or test.empty:
        raise SystemExit("Train/dev/test splits are required.")

    feature_cols = numeric_features + categorical_features
    metrics: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    args.model_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in candidate_models(args.random_state, numeric_features, categorical_features).items():
        model.fit(train[feature_cols], 1 - train["gold_binary"].astype(int))
        dev_error_prob = error_scores(model, dev[feature_cols])
        global_threshold = choose_threshold(dev["gold_binary"], dev_error_prob)
        group_thresholds = choose_group_thresholds(dev, dev_error_prob)

        for split_name, split_frame in [("dev", dev), ("test", test)]:
            split_error_prob = error_scores(model, split_frame[feature_cols])
            metrics.append(
                evaluate(
                    split_name,
                    model_name,
                    "plan_global_threshold",
                    split_frame["gold_binary"],
                    split_error_prob,
                    global_threshold,
                )
            )
            global_thresholds = [global_threshold] * len(split_frame)
            predictions.extend(
                prediction_rows(
                    split_frame,
                    model_name,
                    split_error_prob,
                    global_thresholds,
                    "plan_global_threshold",
                )
            )

            split_group_thresholds = [
                group_thresholds.get(str(phone_group), global_threshold)
                for phone_group in split_frame["phone_group"]
            ]
            y_pred_correct = [
                0 if score >= threshold else 1
                for score, threshold in zip(split_error_prob, split_group_thresholds)
            ]
            group_threshold_reference = max(set(split_group_thresholds), key=split_group_thresholds.count)
            row = evaluate(
                split_name,
                model_name,
                "plan_phone_group_threshold",
                split_frame["gold_binary"],
                split_error_prob,
                group_threshold_reference,
            )
            row["threshold"] = "per_phone_group"
            tn, fp, fn, tp = confusion_matrix(split_frame["gold_binary"], y_pred_correct, labels=[0, 1]).ravel()
            error_precision = precision_score(split_frame["gold_binary"], y_pred_correct, pos_label=0, zero_division=0)
            error_recall = recall_score(split_frame["gold_binary"], y_pred_correct, pos_label=0, zero_division=0)
            row.update(
                {
                    "accuracy": round(accuracy_score(split_frame["gold_binary"], y_pred_correct), 6),
                    "balanced_accuracy": round(balanced_accuracy_score(split_frame["gold_binary"], y_pred_correct), 6),
                    "precision": round(precision_score(split_frame["gold_binary"], y_pred_correct, zero_division=0), 6),
                    "recall": round(recall_score(split_frame["gold_binary"], y_pred_correct, zero_division=0), 6),
                    "macro_f1": round(f1_score(split_frame["gold_binary"], y_pred_correct, average="macro", zero_division=0), 6),
                    "error_precision": round(error_precision, 6),
                    "error_recall": round(error_recall, 6),
                    "error_f1": round(f1_score(split_frame["gold_binary"], y_pred_correct, pos_label=0, zero_division=0), 6),
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                    "meets_plan_minimum": int(error_precision >= MIN_ERROR_PRECISION and error_recall >= MIN_ERROR_RECALL),
                }
            )
            metrics.append(row)
            predictions.extend(
                prediction_rows(
                    split_frame,
                    model_name,
                    split_error_prob,
                    split_group_thresholds,
                    "plan_phone_group_threshold",
                )
            )

        joblib.dump(model, args.model_dir / f"{model_name}.joblib")

    write_csv(args.metrics_output, metrics, METRIC_FIELDS)
    write_csv(
        args.predictions_output,
        predictions,
        [
            "utterance_id",
            "speaker_id",
            "target_phone",
            "phone_group",
            "phone_index",
            "split",
            "dataset_source",
            "audio_path",
            "gold_binary",
            "error_probability",
            "prediction",
            "confidence",
            "model",
            "calibration",
            "threshold",
        ],
    )
    print(f"Wrote acceptance error fusion metrics to {args.metrics_output}")
    print(f"Wrote acceptance error fusion predictions to {args.predictions_output}")
    for row in metrics:
        if row["split"] == "test":
            print(
                f"{row['model']} {row['calibration']}: "
                f"error_precision={row['error_precision']} "
                f"error_recall={row['error_recall']} "
                f"pass={row['meets_plan_minimum']}"
            )


if __name__ == "__main__":
    main()
