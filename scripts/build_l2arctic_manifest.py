"""Build a small L2-ARCTIC phone-level CSV from annotation TextGrid files."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DEFAULT_SPLITS = {
    "BWC": "train",
    "LXC": "train",
    "NCC": "dev",
    "TXHC": "test",
}

SILENCE_LABELS = {"", "sp", "sil"}

PHONE_GROUPS = {
    "liquid": {"R", "L", "ER", "ER0", "ER1", "ER2"},
    "th": {"TH", "DH"},
    "vw": {"V", "W"},
    "nasal": {"M", "N", "NG"},
    "stop": {"P", "B", "T", "D", "K", "G"},
    "fricative": {"F", "V", "S", "Z", "SH", "ZH", "TH", "DH", "HH"},
    "affricate": {"CH", "JH"},
    "vowel": {
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AY",
        "EH",
        "ER",
        "EY",
        "IH",
        "IY",
        "OW",
        "OY",
        "UH",
        "UW",
    },
}


def strip_stress(phone: str) -> str:
    return re.sub(r"\d+$", "", phone.strip().upper())


def phone_group(phone: str) -> str:
    base = strip_stress(phone)
    for group, phones in PHONE_GROUPS.items():
        if base in phones:
            return group
    return "other"


def parse_textgrid_intervals(path: Path, tier_name: str) -> list[dict[str, str | float]]:
    intervals: list[dict[str, str | float]] = []
    current_tier: str | None = None
    in_target_tier = False
    current: dict[str, str | float] | None = None

    tier_re = re.compile(r'name = "([^"]*)"')
    interval_re = re.compile(r"intervals \[\d+\]:")
    xmin_re = re.compile(r"xmin = ([0-9.]+)")
    xmax_re = re.compile(r"xmax = ([0-9.]+)")
    text_re = re.compile(r'text = "(.*)"')

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        tier_match = tier_re.fullmatch(line)
        if tier_match:
            current_tier = tier_match.group(1)
            in_target_tier = current_tier == tier_name
            current = None
            continue

        if not in_target_tier:
            continue

        if interval_re.fullmatch(line):
            if current and {"xmin", "xmax", "text"} <= current.keys():
                intervals.append(current)
            current = {}
            continue

        if current is None:
            continue

        xmin_match = xmin_re.fullmatch(line)
        xmax_match = xmax_re.fullmatch(line)
        text_match = text_re.fullmatch(line)
        if xmin_match:
            current["xmin"] = float(xmin_match.group(1))
        elif xmax_match:
            current["xmax"] = float(xmax_match.group(1))
        elif text_match:
            current["text"] = text_match.group(1).strip()

    if current and {"xmin", "xmax", "text"} <= current.keys():
        intervals.append(current)
    return intervals


def parse_annotation_text(text: str) -> tuple[str, str, str, int]:
    parts = [part.strip() for part in text.split(",")]
    target_phone = parts[0] if parts else ""
    observed_phone = ""
    error_type = ""

    if len(parts) >= 3:
        observed_phone = parts[1]
        error_type = parts[2].lower()
        gold_binary = 0
    elif len(parts) == 2:
        observed_phone = parts[1]
        error_type = "unknown"
        gold_binary = 0
    else:
        gold_binary = 1

    return target_phone, observed_phone, error_type, gold_binary


def build_rows(root: Path, max_files_per_speaker: int | None) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    speakers = [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "annotation").exists()]

    for speaker_dir in speakers:
        speaker_id = speaker_dir.name
        split = DEFAULT_SPLITS.get(speaker_id, "train")
        annotation_files = sorted((speaker_dir / "annotation").glob("*.TextGrid"))
        if max_files_per_speaker is not None:
            annotation_files = annotation_files[:max_files_per_speaker]

        for textgrid_path in annotation_files:
            utterance_id = textgrid_path.stem
            audio_path = speaker_dir / "wav" / f"{utterance_id}.wav"
            intervals = parse_textgrid_intervals(textgrid_path, "phones")
            phone_index = 0

            for interval in intervals:
                annotation_text = str(interval["text"]).strip()
                target_phone, observed_phone, error_type, gold_binary = parse_annotation_text(
                    annotation_text
                )
                if strip_stress(target_phone).lower() in SILENCE_LABELS:
                    continue

                phone_index += 1
                start_ms = round(float(interval["xmin"]) * 1000)
                end_ms = round(float(interval["xmax"]) * 1000)
                rows.append(
                    {
                        "utterance_id": utterance_id,
                        "speaker_id": speaker_id,
                        "target_phone": target_phone,
                        "phone_index": phone_index,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "duration_ms": end_ms - start_ms,
                        "gold_binary": gold_binary,
                        "observed_phone": observed_phone,
                        "error_type_hint": error_type,
                        "phone_group": phone_group(target_phone),
                        "split": split,
                        "audio_path": str(audio_path),
                        "annotation_path": str(textgrid_path),
                        "annotation_text": annotation_text,
                        "dataset_source": "L2-ARCTIC",
                    }
                )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("data/l2arctic_small_manifest.csv"))
    parser.add_argument("--max-files-per-speaker", type=int, default=150)
    args = parser.parse_args()

    rows = build_rows(args.root, args.max_files_per_speaker)
    if not rows:
        raise SystemExit("No rows generated. Check --root and annotation directories.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    split_counts: dict[str, int] = {}
    error_counts: dict[int, int] = {}
    for row in rows:
        split_counts[str(row["split"])] = split_counts.get(str(row["split"]), 0) + 1
        label = int(row["gold_binary"])
        error_counts[label] = error_counts.get(label, 0) + 1

    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Split counts: {split_counts}")
    print(f"gold_binary counts: {error_counts}")


if __name__ == "__main__":
    main()
