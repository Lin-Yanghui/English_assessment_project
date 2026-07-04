"""Generate split acceptance artifacts and speaker-leakage checks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("phones_aligned.csv"))
    parser.add_argument("--alignment-quality", default="pass")
    parser.add_argument("--split-output", type=Path, default=Path("reports/split_manifest.csv"))
    parser.add_argument(
        "--speaker-check-output",
        type=Path,
        default=Path("reports/speaker_split_check.csv"),
    )
    args = parser.parse_args()

    df = pd.read_csv(
        args.input,
        usecols=[
            "utterance_id",
            "speaker_id",
            "split",
            "official_split",
            "gold_binary",
            "alignment_quality",
            "target_phone",
        ],
        encoding="utf-8-sig",
        keep_default_na=False,
        low_memory=False,
    )
    if args.alignment_quality.lower() != "all":
        df = df[df["alignment_quality"].str.lower() == args.alignment_quality.lower()].copy()
    if df.empty:
        raise SystemExit("No rows available after filtering.")

    split_rows: list[dict[str, object]] = []
    for split, group in sorted(df.groupby("split"), key=lambda item: item[0]):
        split_rows.append(
            {
                "split": split,
                "phone_rows": len(group),
                "utterances": group["utterance_id"].nunique(),
                "speakers": group["speaker_id"].nunique(),
                "gold_binary_0": int((group["gold_binary"] == 0).sum()),
                "gold_binary_1": int((group["gold_binary"] == 1).sum()),
                "target_phones": group["target_phone"].nunique(),
                "alignment_quality": args.alignment_quality,
            }
        )

    speaker_rows: list[dict[str, object]] = []
    for speaker_id, group in sorted(df.groupby("speaker_id"), key=lambda item: str(item[0])):
        splits = sorted(str(value) for value in group["split"].unique())
        official_splits = sorted(str(value) for value in group["official_split"].unique())
        speaker_rows.append(
            {
                "speaker_id": speaker_id,
                "splits": ";".join(splits),
                "official_splits": ";".join(official_splits),
                "num_splits": len(splits),
                "leakage": int(len(splits) > 1),
                "phone_rows": len(group),
                "utterances": group["utterance_id"].nunique(),
                "gold_binary_0": int((group["gold_binary"] == 0).sum()),
                "gold_binary_1": int((group["gold_binary"] == 1).sum()),
            }
        )

    write_csv(
        args.split_output,
        split_rows,
        [
            "split",
            "phone_rows",
            "utterances",
            "speakers",
            "gold_binary_0",
            "gold_binary_1",
            "target_phones",
            "alignment_quality",
        ],
    )
    write_csv(
        args.speaker_check_output,
        speaker_rows,
        [
            "speaker_id",
            "splits",
            "official_splits",
            "num_splits",
            "leakage",
            "phone_rows",
            "utterances",
            "gold_binary_0",
            "gold_binary_1",
        ],
    )

    leakage_count = sum(int(row["leakage"]) for row in speaker_rows)
    print(f"Wrote {args.split_output}")
    print(f"Wrote {args.speaker_check_output}")
    print(f"Speakers: {len(speaker_rows)}; leakage speakers: {leakage_count}")
    if leakage_count:
        print("WARNING: speaker leakage detected across splits.")


if __name__ == "__main__":
    main()
