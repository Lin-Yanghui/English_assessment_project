"""Calibrate deployable model thresholds against the phase-plan error target.

The phase-1 plan defines the minimum operating point as:
error precision >= 0.40 while maximizing error recall, with recall >= 0.50
as the pass line. Scores are interpreted as "higher means more likely correct".
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


CORE_GROUPS = {
    "core_vowels": {"vowel"},
    "core_liquids": {"liquid"},
    "core_fricatives": {"fricative"},
    "core_stops": {"stop"},
    "core_nasals": {"nasal"},
}

METRIC_FIELDS = [
    "split",
    "model",
    "calibration",
    "evaluation_scope",
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
    "meets_min_precision",
    "meets_min_recall",
    "meets_plan_minimum",
]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)


def normalize_predictions(path: Path, score_col: str, model_prefix: str = "") -> pd.DataFrame:
    df = read_csv(path)
    if score_col not in df.columns:
        raise ValueError(f"{path} does not contain score column {score_col}")
    if "model" not in df.columns:
        df["model"] = model_prefix or path.stem
    if model_prefix:
        df["model"] = model_prefix + df["model"].astype(str)
    if "calibration" not in df.columns:
        df["calibration"] = "score_threshold"
    keep = [
        "utterance_id",
        "speaker_id",
        "target_phone",
        "phone_group",
        "phone_index",
        "split",
        "dataset_source",
        "gold_binary",
        "model",
        "calibration",
        score_col,
    ]
    present = [col for col in keep if col in df.columns]
    out = df[present].copy()
    out["score"] = pd.to_numeric(out[score_col], errors="coerce")
    out["gold_binary"] = pd.to_numeric(out["gold_binary"], errors="coerce").astype(int)
    out["target_phone"] = out.get("target_phone", "").astype(str)
    out["phone_group"] = out.get("phone_group", "").astype(str)
    out["split"] = out["split"].astype(str)
    out["model"] = out["model"].astype(str)
    out["calibration"] = out["calibration"].astype(str)
    return out.dropna(subset=["score", "gold_binary"])


def load_all(args: argparse.Namespace) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if args.feature_predictions.exists():
        frames.append(normalize_predictions(args.feature_predictions, "gop_score"))
    if args.proxy_predictions.exists():
        frames.append(normalize_predictions(args.proxy_predictions, "proxy_gop_score"))
    if args.ssl_predictions.exists():
        frames.append(normalize_predictions(args.ssl_predictions, "ssl_score"))
    if args.fusion_predictions.exists():
        fusion = normalize_predictions(args.fusion_predictions, "gop_score", model_prefix="plan_")
        frames.append(fusion)
    if not frames:
        raise SystemExit("No prediction files found.")
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(
        subset=[
            "utterance_id",
            "target_phone",
            "phone_index",
            "split",
            "model",
            "calibration",
        ],
        keep="last",
    )


def error_metrics(y_true: pd.Series, y_pred: pd.Series, scores: pd.Series) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = 0.5
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "error_precision": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "error_recall": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "error_f1": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def candidate_thresholds(scores: pd.Series) -> list[float]:
    del scores
    return [i / 100 for i in range(1, 100)]


def select_threshold(
    dev: pd.DataFrame,
    min_error_precision: float,
) -> tuple[float, dict[str, float | int], bool]:
    best_threshold = 0.5
    best_stats: dict[str, float | int] | None = None
    feasible = False
    for threshold in candidate_thresholds(dev["score"]):
        y_pred = (dev["score"] >= threshold).astype(int)
        stats = error_metrics(dev["gold_binary"], y_pred, dev["score"])
        passes_precision = stats["error_precision"] >= min_error_precision
        key = (
            int(passes_precision),
            stats["error_recall"] if passes_precision else stats["error_precision"],
            stats["error_f1"],
            stats["balanced_accuracy"],
        )
        if best_stats is None:
            best_key = (-1, -1.0, -1.0, -1.0)
        else:
            best_passes = best_stats["error_precision"] >= min_error_precision
            best_key = (
                int(best_passes),
                best_stats["error_recall"] if best_passes else best_stats["error_precision"],
                best_stats["error_f1"],
                best_stats["balanced_accuracy"],
            )
        if key > best_key:
            best_threshold = threshold
            best_stats = stats
            feasible = bool(passes_precision)
    if best_stats is None:
        raise ValueError("Could not select a threshold from empty dev data.")
    return best_threshold, best_stats, feasible


def make_metric_row(
    frame: pd.DataFrame,
    model: str,
    calibration: str,
    scope: str,
    threshold: float,
    min_error_precision: float,
    min_error_recall: float,
) -> dict[str, object]:
    y_pred = (frame["score"] >= threshold).astype(int)
    stats = error_metrics(frame["gold_binary"], y_pred, frame["score"])
    row: dict[str, object] = {
        "split": str(frame["split"].iloc[0]),
        "model": model,
        "calibration": calibration,
        "evaluation_scope": scope,
        "n_samples": len(frame),
        "positive_rate": round(float(frame["gold_binary"].mean()), 6),
        "threshold": round(threshold, 6),
    }
    for key, value in stats.items():
        row[key] = round(value, 6) if isinstance(value, float) else value
    row["meets_min_precision"] = int(stats["error_precision"] >= min_error_precision)
    row["meets_min_recall"] = int(stats["error_recall"] >= min_error_recall)
    row["meets_plan_minimum"] = int(
        stats["error_precision"] >= min_error_precision
        and stats["error_recall"] >= min_error_recall
    )
    return row


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]], min_precision: float, min_recall: float) -> None:
    test_rows = [row for row in rows if row["split"] == "test"]
    passed = [row for row in test_rows if row["meets_plan_minimum"]]
    lines = [
        "# Plan Acceptance Threshold Calibration",
        "",
        f"Minimum line: error precision >= {min_precision:.2f}, error recall >= {min_recall:.2f}.",
        "",
        "| scope | model | calibration | n | threshold | error_precision | error_recall | balanced_accuracy | macro_f1 | pass |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = sorted(
        test_rows,
        key=lambda row: (
            int(row["meets_plan_minimum"]),
            float(row["error_recall"]),
            float(row["error_precision"]),
            float(row["balanced_accuracy"]),
        ),
        reverse=True,
    )
    for row in ordered:
        lines.append(
            f"| {row['evaluation_scope']} | {row['model']} | {row['calibration']} | "
            f"{row['n_samples']} | {row['threshold']:.6f} | "
            f"{row['error_precision']:.6f} | {row['error_recall']:.6f} | "
            f"{row['balanced_accuracy']:.6f} | {row['macro_f1']:.6f} | "
            f"{row['meets_plan_minimum']} |"
        )
    lines.extend(["", "## Summary", ""])
    if passed:
        lines.append(f"- {len(passed)} deployable scoped operating points meet the minimum line.")
    else:
        lines.append("- No deployable operating point meets the minimum line on the current combined test set.")
    lines.append("- Thresholds are selected on dev and reported on test; no source labels or teacher scores are used.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def scoped_frames(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    scopes = [("overall", df)]
    for scope, groups in CORE_GROUPS.items():
        scopes.append((scope, df[df["phone_group"].isin(groups)].copy()))
    return scopes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-predictions", type=Path, default=Path("reports/combined_feature_baseline_predictions.csv"))
    parser.add_argument("--proxy-predictions", type=Path, default=Path("reports/combined_proxy_gop_predictions.csv"))
    parser.add_argument("--ssl-predictions", type=Path, default=Path("reports/combined_ssl_phone_classifier_predictions.csv"))
    parser.add_argument("--fusion-predictions", type=Path, default=Path("reports/combined_fusion_model_predictions.csv"))
    parser.add_argument("--metrics-output", type=Path, default=Path("reports/combined_plan_acceptance_threshold_metrics.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("reports/combined_plan_acceptance_thresholds.md"))
    parser.add_argument("--min-error-precision", type=float, default=0.40)
    parser.add_argument("--min-error-recall", type=float, default=0.50)
    parser.add_argument("--min-dev-errors", type=int, default=20)
    args = parser.parse_args()

    df = load_all(args)
    rows: list[dict[str, object]] = []
    for (model, calibration), model_df in sorted(df.groupby(["model", "calibration"])):
        for scope, scope_df in scoped_frames(model_df):
            dev = scope_df[scope_df["split"] == "dev"].copy()
            test = scope_df[scope_df["split"] == "test"].copy()
            if dev.empty or test.empty or int((dev["gold_binary"] == 0).sum()) < args.min_dev_errors:
                continue
            threshold, _, _ = select_threshold(dev, args.min_error_precision)
            for split_df in [dev, test]:
                rows.append(
                    make_metric_row(
                        split_df,
                        model,
                        f"{calibration}_plan_threshold",
                        scope,
                        threshold,
                        args.min_error_precision,
                        args.min_error_recall,
                    )
                )

    write_csv(args.metrics_output, rows, METRIC_FIELDS)
    write_markdown(args.summary_output, rows, args.min_error_precision, args.min_error_recall)
    test_passes = [row for row in rows if row["split"] == "test" and row["meets_plan_minimum"]]
    print(f"Wrote plan-threshold metrics to {args.metrics_output}")
    print(f"Wrote summary to {args.summary_output}")
    print(f"Passing test operating points: {len(test_passes)}")


if __name__ == "__main__":
    main()
