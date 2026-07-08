"""Train an acoustic phone-posterior model and export GOP-style scores.

This script computes an acoustic GOP evidence score without using the
pronunciation-correctness label as the prediction target. It trains a phone
identity classifier on SSL segment embeddings from correctly realized training
phones, then scores every dev/test segment by the posterior assigned to its
expected target phone.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import pandas as pd
import joblib
from sklearn.naive_bayes import GaussianNB


DEFAULT_METADATA_COLUMNS = [
    "utterance_id",
    "speaker_id",
    "target_phone",
    "phone_group",
    "phone_index",
    "start_ms",
    "end_ms",
    "duration_ms",
    "gold_binary",
    "gold_three_class",
    "split",
    "dataset_source",
    "audio_path",
]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def posterior_margin(probs: list[float], target_index: int) -> float:
    eps = 1e-8
    target = max(float(probs[target_index]), eps)
    competitor = max(float(value) for idx, value in enumerate(probs) if idx != target_index)
    competitor = max(competitor, eps)
    return math.log(target) - math.log(competitor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, default=Path("reports/ssl_phone_embeddings.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/acoustic_gop_scores.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--model-name", default="acoustic_gop_phone_sgd")
    parser.add_argument(
        "--train-positive-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train the phone posterior model only from correct/acceptable training segments.",
    )
    parser.add_argument("--var-smoothing", type=float, default=1e-9)
    args = parser.parse_args()

    df = pd.read_csv(args.embeddings, encoding="utf-8-sig", keep_default_na=False)
    embedding_cols = [col for col in df.columns if col.startswith("ssl_")]
    if not embedding_cols:
        raise SystemExit("No ssl_* embedding columns found.")
    required = {"split", "target_phone", "gold_binary"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    df["gold_binary"] = df["gold_binary"].astype(int)
    train = df[df["split"] == "train"].copy()
    if args.train_positive_only:
        train = train[train["gold_binary"] == 1].copy()
    if train.empty:
        raise SystemExit("No rows available for acoustic GOP phone-posterior training.")
    if train["target_phone"].nunique() < 2:
        raise SystemExit("Need at least two target phones to train an acoustic GOP model.")

    model = GaussianNB(var_smoothing=args.var_smoothing)
    model.fit(train[embedding_cols], train["target_phone"])
    classes = list(model.classes_)
    class_to_index = {str(label): idx for idx, label in enumerate(classes)}

    eval_df = df[df["split"].isin(["dev", "test"])].copy()
    if eval_df.empty:
        raise SystemExit("Embeddings must contain dev and/or test splits.")

    probabilities = model.predict_proba(eval_df[embedding_cols])
    rows: list[dict[str, object]] = []
    fields = [col for col in DEFAULT_METADATA_COLUMNS if col in eval_df.columns]
    for row_idx, (_, row) in enumerate(eval_df.iterrows()):
        target_phone = str(row["target_phone"])
        probs = [float(value) for value in probabilities[row_idx]]
        target_index = class_to_index.get(target_phone)
        if target_index is None:
            target_posterior = 0.0
            margin = -99.0
            predicted_phone = str(classes[int(max(range(len(probs)), key=lambda idx: probs[idx]))])
        else:
            target_posterior = probs[target_index]
            margin = posterior_margin(probs, target_index)
            predicted_phone = str(classes[int(max(range(len(probs)), key=lambda idx: probs[idx]))])
        out = {col: row[col] for col in fields}
        out["observed_phone"] = row.get("perceived_phone", "")
        out["error_type_hint"] = row.get("error_type", "")
        out["annotation_text"] = row.get("target_phone_raw", row.get("target_phone", ""))
        out["gop_score"] = round(target_posterior, 6)
        out["acoustic_gop_margin"] = round(margin, 6)
        out["predicted_phone"] = predicted_phone
        out["model"] = args.model_name
        rows.append(out)

    fieldnames = fields + [
        "observed_phone",
        "error_type_hint",
        "annotation_text",
        "gop_score",
        "acoustic_gop_margin",
        "predicted_phone",
        "model",
    ]
    write_csv(args.output, rows, fieldnames)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_dir / f"{args.model_name}.joblib")

    print(f"Training rows: {len(train)}")
    print(f"Phone classes: {len(classes)}")
    print(f"Wrote acoustic GOP scores to {args.output}")
    print(f"Wrote acoustic GOP model to {args.model_dir / f'{args.model_name}.joblib'}")


if __name__ == "__main__":
    main()
