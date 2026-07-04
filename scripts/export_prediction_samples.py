"""Export prediction samples for a fixed number of test utterances."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SAMPLE_COLUMNS = [
    "utterance_id",
    "speaker_id",
    "target_phone",
    "phone_index",
    "start_ms",
    "end_ms",
    "duration_ms",
    "gold_binary",
    "prediction",
    "confidence",
    "proxy_gop_score",
    "group_threshold",
    "phone_group",
    "observed_phone",
    "error_type_hint",
    "audio_path",
    "annotation_text",
    "model",
    "calibration",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("reports/proxy_gop_predictions.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/prediction_samples_100_utterances.csv"))
    parser.add_argument(
        "--utterance-list-output",
        type=Path,
        default=Path("reports/prediction_sample_utterances.csv"),
    )
    parser.add_argument("--model", default="feature_random_forest")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-utterances", type=int, default=100)
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig", keep_default_na=False)
    subset = df[(df["model"] == args.model) & (df["split"] == args.split)].copy()
    if subset.empty:
        raise SystemExit(f"No rows found for model={args.model!r}, split={args.split!r}.")

    utterances = sorted(subset["utterance_id"].unique())[: args.num_utterances]
    if len(utterances) < args.num_utterances:
        raise SystemExit(
            f"Only found {len(utterances)} utterances, fewer than requested {args.num_utterances}."
        )

    sample = subset[subset["utterance_id"].isin(utterances)].copy()
    sample = sample.sort_values(["utterance_id", "phone_index"])
    missing_columns = [col for col in SAMPLE_COLUMNS if col not in sample.columns]
    if missing_columns:
        raise SystemExit(f"Missing expected columns: {missing_columns}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample[SAMPLE_COLUMNS].to_csv(args.output, index=False, encoding="utf-8-sig")

    utterance_summary = (
        sample.groupby(["utterance_id", "speaker_id", "audio_path"], as_index=False)
        .agg(
            phone_rows=("target_phone", "count"),
            gold_errors=("gold_binary", lambda values: int((values.astype(int) == 0).sum())),
            predicted_errors=("prediction", lambda values: int((values.astype(int) == 0).sum())),
        )
        .sort_values("utterance_id")
    )
    utterance_summary.to_csv(args.utterance_list_output, index=False, encoding="utf-8-sig")

    print(f"Wrote {len(sample)} phone rows from {len(utterances)} utterances to {args.output}")
    print(f"Wrote utterance list to {args.utterance_list_output}")
    print(
        "Label counts:",
        sample["gold_binary"].astype(int).value_counts().sort_index().to_dict(),
    )
    print(
        "Prediction counts:",
        sample["prediction"].astype(int).value_counts().sort_index().to_dict(),
    )


if __name__ == "__main__":
    main()
