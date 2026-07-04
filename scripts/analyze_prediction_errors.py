"""Analyze false positives, false negatives, and weak phone groups."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CASE_COLUMNS = [
    "error_case",
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
]


def add_case_type(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["error_case"] = "correct"
    frame.loc[(frame["gold_binary"] == 1) & (frame["prediction"] == 0), "error_case"] = "false_positive"
    frame.loc[(frame["gold_binary"] == 0) & (frame["prediction"] == 1), "error_case"] = "false_negative"
    frame.loc[(frame["gold_binary"] == 0) & (frame["prediction"] == 0), "error_case"] = "true_negative"
    frame.loc[(frame["gold_binary"] == 1) & (frame["prediction"] == 1), "error_case"] = "true_positive"
    return frame


def aggregate(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(group_col)
        .agg(
            n_samples=("target_phone", "count"),
            gold_errors=("gold_binary", lambda values: int((values == 0).sum())),
            predicted_errors=("prediction", lambda values: int((values == 0).sum())),
            false_positives=("error_case", lambda values: int((values == "false_positive").sum())),
            false_negatives=("error_case", lambda values: int((values == "false_negative").sum())),
            true_negatives=("error_case", lambda values: int((values == "true_negative").sum())),
            true_positives=("error_case", lambda values: int((values == "true_positive").sum())),
            mean_proxy_gop_score=("proxy_gop_score", "mean"),
        )
        .reset_index()
    )
    grouped["gold_error_rate"] = grouped["gold_errors"] / grouped["n_samples"]
    grouped["predicted_error_rate"] = grouped["predicted_errors"] / grouped["n_samples"]
    grouped["false_positive_rate"] = grouped["false_positives"] / grouped["n_samples"]
    grouped["false_negative_rate"] = grouped["false_negatives"] / grouped["n_samples"]
    return grouped.sort_values(
        ["false_negatives", "false_positives", "n_samples"],
        ascending=[False, False, False],
    )


def representative_cases(frame: pd.DataFrame, max_cases_per_type: int) -> pd.DataFrame:
    fp = frame[frame["error_case"] == "false_positive"].copy()
    fn = frame[frame["error_case"] == "false_negative"].copy()

    fp["rank_distance"] = fp["group_threshold"] - fp["proxy_gop_score"]
    fn["rank_distance"] = fn["proxy_gop_score"] - fn["group_threshold"]

    fp = fp.sort_values(["rank_distance", "confidence"], ascending=[False, False]).head(max_cases_per_type)
    fn = fn.sort_values(["rank_distance", "confidence"], ascending=[False, False]).head(max_cases_per_type)
    return pd.concat([fp, fn], ignore_index=True)[CASE_COLUMNS]


def write_summary(
    path: Path,
    frame: pd.DataFrame,
    group_summary: pd.DataFrame,
    phone_summary: pd.DataFrame,
) -> None:
    counts = frame["error_case"].value_counts().to_dict()
    lines = [
        "# Error Analysis",
        "",
        "Model: feature_random_forest + phone_group_threshold on test split.",
        "",
        "## Case Counts",
        "",
        f"- True positives: {counts.get('true_positive', 0)}",
        f"- True negatives: {counts.get('true_negative', 0)}",
        f"- False positives: {counts.get('false_positive', 0)}",
        f"- False negatives: {counts.get('false_negative', 0)}",
        "",
        "## Weakest Phone Groups",
        "",
        "| phone_group | n | gold_errors | false_positives | false_negatives | gold_error_rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    weakest_groups = group_summary.sort_values(
        ["false_negatives", "false_positives"], ascending=False
    ).head(8)
    for _, row in weakest_groups.iterrows():
        lines.append(
            f"| {row['phone_group']} | {int(row['n_samples'])} | {int(row['gold_errors'])} | "
            f"{int(row['false_positives'])} | {int(row['false_negatives'])} | "
            f"{row['gold_error_rate']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Weakest Target Phones",
            "",
            "| target_phone | n | gold_errors | false_positives | false_negatives | gold_error_rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    weakest_phones = phone_summary.sort_values(
        ["false_negatives", "false_positives"], ascending=False
    ).head(15)
    for _, row in weakest_phones.iterrows():
        lines.append(
            f"| {row['target_phone']} | {int(row['n_samples'])} | {int(row['gold_errors'])} | "
            f"{int(row['false_positives'])} | {int(row['false_negatives'])} | "
            f"{row['gold_error_rate']:.4f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("reports/proxy_gop_predictions.csv"))
    parser.add_argument("--model", default="feature_random_forest")
    parser.add_argument("--split", default="test")
    parser.add_argument("--cases-output", type=Path, default=Path("reports/error_cases.csv"))
    parser.add_argument("--group-output", type=Path, default=Path("reports/error_analysis_by_phone_group.csv"))
    parser.add_argument("--phone-output", type=Path, default=Path("reports/error_analysis_by_target_phone.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("reports/error_analysis_summary.md"))
    parser.add_argument("--max-cases-per-type", type=int, default=50)
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig", keep_default_na=False)
    df = df[(df["model"] == args.model) & (df["split"] == args.split)].copy()
    if df.empty:
        raise SystemExit(f"No rows found for model={args.model!r}, split={args.split!r}.")

    numeric_cols = [
        "gold_binary",
        "prediction",
        "confidence",
        "proxy_gop_score",
        "group_threshold",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    df = add_case_type(df)
    group_summary = aggregate(df, "phone_group")
    phone_summary = aggregate(df, "target_phone")
    cases = representative_cases(df, args.max_cases_per_type)

    args.cases_output.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(args.cases_output, index=False, encoding="utf-8-sig")
    group_summary.to_csv(args.group_output, index=False, encoding="utf-8-sig")
    phone_summary.to_csv(args.phone_output, index=False, encoding="utf-8-sig")
    write_summary(args.summary_output, df, group_summary, phone_summary)

    print(f"Wrote representative cases to {args.cases_output}")
    print(f"Wrote phone-group summary to {args.group_output}")
    print(f"Wrote target-phone summary to {args.phone_output}")
    print(f"Wrote summary to {args.summary_output}")
    print(df["error_case"].value_counts().to_dict())


if __name__ == "__main__":
    main()
