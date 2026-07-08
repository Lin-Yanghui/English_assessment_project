"""Calibrate phone-group and target-phone thresholds from proxy GOP scores."""

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
    "calibration",
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

THRESHOLD_FIELDS = [
    "model",
    "threshold_level",
    "threshold_key",
    "phone_group",
    "target_phone",
    "threshold",
    "dev_n_samples",
    "dev_positive_rate",
    "dev_balanced_accuracy",
    "fallback_used",
]


def threshold_score(y_true: pd.Series, y_pred: pd.Series, objective: str) -> float:
    if objective == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if objective == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    raise ValueError(f"Unsupported objective: {objective}")


def best_threshold(y_true: pd.Series, scores: pd.Series, objective: str) -> tuple[float, float]:
    best_t = 0.5
    best_score = -1.0
    for step in range(5, 96):
        threshold = step / 100
        y_pred = (scores >= threshold).astype(int)
        score = threshold_score(y_true, y_pred, objective)
        if score > best_score:
            best_score = score
            best_t = threshold
    return best_t, best_score


def evaluate(
    frame: pd.DataFrame,
    model_name: str,
    calibration: str,
    score_col: str,
    pred_col: str,
) -> dict[str, object]:
    y_true = frame["gold_binary"].astype(int)
    y_pred = frame[pred_col].astype(int)
    scores = frame[score_col].astype(float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = 0.5
    return {
        "split": str(frame["split"].iloc[0]),
        "model": model_name,
        "calibration": calibration,
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
        "auc": round(auc, 6),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def calibrate_model(
    df: pd.DataFrame,
    model_name: str,
    score_col: str,
    min_group_samples: int,
    min_phone_samples: int,
    objective: str,
) -> tuple[list[dict[str, object]], pd.DataFrame, list[dict[str, object]]]:
    model_df = df[df["model"] == model_name].copy()
    dev = model_df[model_df["split"] == "dev"].copy()
    eval_df = model_df[model_df["split"].isin(["train", "dev", "test"])].copy()
    if dev.empty:
        raise ValueError(f"No dev rows found for model {model_name}")

    global_threshold, global_score = best_threshold(
        dev["gold_binary"].astype(int),
        dev[score_col],
        objective,
    )
    group_thresholds: dict[str, float] = {}
    phone_thresholds: dict[str, float] = {}
    threshold_rows: list[dict[str, object]] = []

    for phone_group, group_dev in sorted(dev.groupby("phone_group")):
        y_true = group_dev["gold_binary"].astype(int)
        scores = group_dev[score_col].astype(float)
        fallback_used = len(group_dev) < min_group_samples or y_true.nunique() < 2
        if fallback_used:
            threshold = global_threshold
            score = global_score
        else:
            threshold, score = best_threshold(y_true, scores, objective)
        group_thresholds[str(phone_group)] = threshold
        threshold_rows.append(
            {
                "model": model_name,
                "threshold_level": "phone_group",
                "threshold_key": phone_group,
                "phone_group": phone_group,
                "target_phone": "",
                "threshold": round(threshold, 6),
                "dev_n_samples": len(group_dev),
                "dev_positive_rate": round(float(y_true.mean()), 6),
                "dev_balanced_accuracy": round(score, 6),
                "fallback_used": int(fallback_used),
            }
        )

    for target_phone, phone_dev in sorted(dev.groupby("target_phone")):
        y_true = phone_dev["gold_binary"].astype(int)
        scores = phone_dev[score_col].astype(float)
        phone_group = str(phone_dev["phone_group"].iloc[0])
        group_fallback = group_thresholds.get(phone_group, global_threshold)
        fallback_used = len(phone_dev) < min_phone_samples or y_true.nunique() < 2
        if fallback_used:
            threshold = group_fallback
            y_pred = (scores >= threshold).astype(int)
            score = threshold_score(y_true, y_pred, objective)
        else:
            threshold, score = best_threshold(y_true, scores, objective)
        phone_thresholds[str(target_phone)] = threshold
        threshold_rows.append(
            {
                "model": model_name,
                "threshold_level": "target_phone",
                "threshold_key": target_phone,
                "phone_group": phone_group,
                "target_phone": target_phone,
                "threshold": round(threshold, 6),
                "dev_n_samples": len(phone_dev),
                "dev_positive_rate": round(float(y_true.mean()), 6),
                "dev_balanced_accuracy": round(score, 6),
                "fallback_used": int(fallback_used),
            }
        )

    eval_df["proxy_gop_score"] = eval_df[score_col].astype(float)
    eval_df["group_threshold"] = eval_df["phone_group"].map(group_thresholds).fillna(global_threshold)
    eval_df["target_phone_threshold"] = eval_df["target_phone"].map(phone_thresholds)
    eval_df["target_phone_threshold"] = eval_df["target_phone_threshold"].fillna(eval_df["group_threshold"])

    group_df = eval_df.copy()
    group_df["decision_threshold"] = group_df["group_threshold"]
    group_df["threshold_level"] = "phone_group"
    group_df["prediction"] = (group_df["proxy_gop_score"] >= group_df["decision_threshold"]).astype(int)
    group_df["confidence"] = group_df.apply(
        lambda row: round(
            max(row["proxy_gop_score"], 1 - row["proxy_gop_score"]),
            6,
        ),
        axis=1,
    )
    group_df["calibration"] = "phone_group_threshold"

    phone_df = eval_df.copy()
    phone_df["decision_threshold"] = phone_df["target_phone_threshold"]
    phone_df["threshold_level"] = "target_phone"
    phone_df["prediction"] = (phone_df["proxy_gop_score"] >= phone_df["decision_threshold"]).astype(int)
    phone_df["confidence"] = phone_df.apply(
        lambda row: round(
            max(row["proxy_gop_score"], 1 - row["proxy_gop_score"]),
            6,
        ),
        axis=1,
    )
    phone_df["calibration"] = "target_phone_threshold"

    predictions = pd.concat([group_df, phone_df], ignore_index=True)
    metrics: list[dict[str, object]] = []
    for calibration in ["phone_group_threshold", "target_phone_threshold"]:
        calibration_df = predictions[predictions["calibration"] == calibration]
        metrics.extend(
            [
                evaluate(
                    calibration_df[calibration_df["split"] == split],
                    model_name,
                    calibration,
                    "proxy_gop_score",
                    "prediction",
                )
                for split in ["dev", "test"]
            ]
        )

    return threshold_rows, predictions, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("reports/feature_baseline_predictions.csv"))
    parser.add_argument(
        "--thresholds-output", type=Path, default=Path("reports/proxy_gop_group_thresholds.csv")
    )
    parser.add_argument(
        "--target-thresholds-output",
        type=Path,
        default=Path("reports/proxy_gop_target_phone_thresholds.csv"),
    )
    parser.add_argument(
        "--all-thresholds-output",
        type=Path,
        default=Path("reports/proxy_gop_thresholds.csv"),
    )
    parser.add_argument("--metrics-output", type=Path, default=Path("reports/proxy_gop_metrics.csv"))
    parser.add_argument(
        "--predictions-output", type=Path, default=Path("reports/proxy_gop_predictions.csv")
    )
    parser.add_argument("--models", nargs="+", default=["feature_logreg", "feature_random_forest"])
    parser.add_argument("--min-group-samples", type=int, default=50)
    parser.add_argument("--min-phone-samples", type=int, default=30)
    parser.add_argument(
        "--objective",
        choices=["macro_f1", "balanced_accuracy"],
        default="macro_f1",
        help="Metric optimized when selecting thresholds on the dev split.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig", keep_default_na=False)
    df["gold_binary"] = df["gold_binary"].astype(int)
    df["gop_score"] = df["gop_score"].astype(float)
    if "observed_phone" not in df.columns:
        df["observed_phone"] = df.get("perceived_phone", "")
    if "error_type_hint" not in df.columns:
        df["error_type_hint"] = df.get("error_type", "")
    if "annotation_text" not in df.columns:
        df["annotation_text"] = df.get("target_phone_raw", df.get("target_phone", ""))

    all_thresholds: list[dict[str, object]] = []
    all_metrics: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for model_name in args.models:
        thresholds, predictions, metrics = calibrate_model(
            df,
            model_name,
            "gop_score",
            args.min_group_samples,
            args.min_phone_samples,
            args.objective,
        )
        all_thresholds.extend(thresholds)
        all_metrics.extend(metrics)
        prediction_frames.append(predictions)

    group_thresholds = [row for row in all_thresholds if row["threshold_level"] == "phone_group"]
    target_thresholds = [row for row in all_thresholds if row["threshold_level"] == "target_phone"]
    write_csv(args.thresholds_output, group_thresholds, THRESHOLD_FIELDS)
    write_csv(args.target_thresholds_output, target_thresholds, THRESHOLD_FIELDS)
    write_csv(args.all_thresholds_output, all_thresholds, THRESHOLD_FIELDS)
    write_csv(args.metrics_output, all_metrics, METRIC_FIELDS)

    out_df = pd.concat(prediction_frames, ignore_index=True)
    keep_cols = [
        "utterance_id",
        "speaker_id",
        "target_phone",
        "phone_index",
        "start_ms",
        "end_ms",
        "duration_ms",
        "gold_binary",
        "observed_phone",
        "error_type_hint",
        "phone_group",
        "split",
        "audio_path",
        "annotation_path",
        "annotation_text",
        "dataset_source",
        "proxy_gop_score",
        "group_threshold",
        "target_phone_threshold",
        "decision_threshold",
        "threshold_level",
        "prediction",
        "confidence",
        "model",
        "calibration",
    ]
    for col in keep_cols:
        if col not in out_df.columns:
            out_df[col] = ""
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    out_df[keep_cols].to_csv(args.predictions_output, index=False, encoding="utf-8-sig")

    print(f"Wrote thresholds to {args.thresholds_output}")
    print(f"Wrote target-phone thresholds to {args.target_thresholds_output}")
    print(f"Wrote all thresholds to {args.all_thresholds_output}")
    print(f"Wrote metrics to {args.metrics_output}")
    print(f"Wrote predictions to {args.predictions_output}")
    for row in all_metrics:
        print(
            f"{row['model']} {row['split']}: "
            f"balanced_accuracy={row['balanced_accuracy']} "
            f"macro_f1={row['macro_f1']} auc={row['auc']}"
        )


if __name__ == "__main__":
    main()
