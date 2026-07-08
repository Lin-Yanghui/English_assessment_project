"""Generate dataset acceptance artifacts for SpeechOcean762 and L2-ARCTIC."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


FIELD_DESCRIPTIONS = {
    "utterance_id": "Unique utterance identifier.",
    "speaker_id": "Speaker identifier. Used for speaker-level train/dev/test isolation.",
    "speaker_gender": "Speaker gender metadata when available.",
    "speaker_age": "Speaker age metadata when available.",
    "native_language": "Speaker native language metadata.",
    "transcript": "Sentence transcript.",
    "sentence_accuracy": "Source sentence-level accuracy score when available.",
    "sentence_fluency": "Source sentence-level fluency score when available.",
    "sentence_completeness": "Source sentence-level completeness score when available.",
    "sentence_prosodic": "Source sentence-level prosodic score when available.",
    "word": "Word token containing the target phone.",
    "word_index": "Word index within utterance.",
    "word_accuracy": "Source word-level accuracy score when available.",
    "word_stress": "Source word stress score when available.",
    "target_phone_raw": "Original target phone label from the source parser.",
    "target_phone": "Normalized target phone label used by models.",
    "perceived_phone_raw": "Original perceived or annotated phone label when available.",
    "perceived_phone": "Normalized perceived phone label when available.",
    "phone_index": "Phone index within utterance.",
    "start_ms": "Phone segment start time in milliseconds.",
    "end_ms": "Phone segment end time in milliseconds.",
    "duration_ms": "Phone segment duration in milliseconds.",
    "source_score": "Original source phone-level score when available.",
    "gold_binary": "Binary target label: 1 acceptable/correct, 0 error/unacceptable.",
    "attention_binary": "Broader attention or review flag.",
    "gold_three_class": "Reusable three-class label: correct, acceptable, or incorrect.",
    "error_type": "Source or derived error type.",
    "phone_group": "Coarse phone group used for grouped analysis and thresholds.",
    "dataset_source": "Dataset source name.",
    "split": "Project train/dev/test split.",
    "official_split": "Official or inherited split label where available.",
    "audio_path": "Relative path to source audio.",
    "annotation_path": "Relative path to source annotation.",
    "alignment_method": "Alignment or parsing method identifier.",
    "alignment_score": "Alignment quality score when available.",
    "alignment_quality": "Alignment quality status, e.g. pass/review.",
}


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def value_counts_text(series: pd.Series) -> str:
    counts = series.fillna("").astype(str).value_counts(dropna=False)
    return "; ".join(f"{key or 'blank'}={value}" for key, value in counts.items())


def candidate_paths(root: Path, raw_path: str) -> list[Path]:
    normalized = raw_path.replace("\\", "/")
    candidates = [root / raw_path, Path(raw_path)]
    l2_prefix = "data/raw/L2-ARCTIC-v5.0/Mandarin/"
    if normalized.startswith(l2_prefix):
        candidates.append(root / normalized[len(l2_prefix) :])
    speechocean_prefix = "data/raw/speechocean762/"
    if normalized.startswith(speechocean_prefix):
        candidates.append(root / normalized[len(speechocean_prefix) :])
        candidates.append(root / "speechocean762" / normalized[len(speechocean_prefix) :])
    return candidates


def existing_unique_paths(root: Path, paths: pd.Series) -> tuple[int, int]:
    unique_paths = sorted({p for p in paths.fillna("").astype(str) if p})
    exists = 0
    for rel in unique_paths:
        if any(path.exists() for path in candidate_paths(root, rel)):
            exists += 1
    return exists, len(unique_paths)


def dataset_manifest(root: Path, source: pd.DataFrame, aligned: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset_name, group in source.groupby("dataset_source", dropna=False):
        audio_exists, audio_total = existing_unique_paths(root, group["audio_path"])
        annotation_exists, annotation_total = existing_unique_paths(root, group["annotation_path"])
        pass_rows = int((group["alignment_quality"].fillna("").astype(str).str.lower() == "pass").sum())
        rows.append(
            {
                "dataset_source": dataset_name,
                "table": "phones.csv",
                "rows": len(group),
                "utterances": group["utterance_id"].nunique(),
                "speakers": group["speaker_id"].nunique(),
                "splits": value_counts_text(group["split"]),
                "alignment_quality": value_counts_text(group["alignment_quality"]),
                "pass_rows": pass_rows,
                "gold_binary_0": int((group["gold_binary"].astype(int) == 0).sum()),
                "gold_binary_1": int((group["gold_binary"].astype(int) == 1).sum()),
                "target_phones": group["target_phone"].nunique(),
                "audio_paths_existing": audio_exists,
                "audio_paths_total": audio_total,
                "annotation_paths_existing": annotation_exists,
                "annotation_paths_total": annotation_total,
            }
        )

    for dataset_name, group in aligned.groupby("dataset_source", dropna=False):
        rows.append(
            {
                "dataset_source": dataset_name,
                "table": "phones_aligned.csv",
                "rows": len(group),
                "utterances": group["utterance_id"].nunique(),
                "speakers": group["speaker_id"].nunique(),
                "splits": value_counts_text(group["split"]),
                "alignment_quality": value_counts_text(group["alignment_quality"]),
                "pass_rows": int((group["alignment_quality"].astype(str).str.lower() == "pass").sum()),
                "gold_binary_0": int((group["gold_binary"].astype(int) == 0).sum()),
                "gold_binary_1": int((group["gold_binary"].astype(int) == 1).sum()),
                "target_phones": group["target_phone"].nunique(),
                "audio_paths_existing": "",
                "audio_paths_total": "",
                "annotation_paths_existing": "",
                "annotation_paths_total": "",
            }
        )
    return rows


def split_rows(df: pd.DataFrame, dataset_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split, group in sorted(df.groupby("split"), key=lambda item: str(item[0])):
        rows.append(
            {
                "dataset_source": dataset_name,
                "split": split,
                "phone_rows": len(group),
                "utterances": group["utterance_id"].nunique(),
                "speakers": group["speaker_id"].nunique(),
                "gold_binary_0": int((group["gold_binary"].astype(int) == 0).sum()),
                "gold_binary_1": int((group["gold_binary"].astype(int) == 1).sum()),
                "target_phones": group["target_phone"].nunique(),
                "alignment_quality": value_counts_text(group["alignment_quality"]),
            }
        )
    return rows


def speaker_rows(df: pd.DataFrame, dataset_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for speaker_id, group in sorted(df.groupby("speaker_id"), key=lambda item: str(item[0])):
        splits = sorted(str(value) for value in group["split"].unique())
        rows.append(
            {
                "dataset_source": dataset_name,
                "speaker_id": speaker_id,
                "splits": ";".join(splits),
                "num_splits": len(splits),
                "leakage": int(len(splits) > 1),
                "phone_rows": len(group),
                "utterances": group["utterance_id"].nunique(),
                "gold_binary_0": int((group["gold_binary"].astype(int) == 0).sum()),
                "gold_binary_1": int((group["gold_binary"].astype(int) == 1).sum()),
            }
        )
    return rows


def write_data_dictionary(path: Path, df: pd.DataFrame) -> None:
    lines = [
        "# Data Dictionary",
        "",
        "This dictionary covers the project-level phone table schema used by `phones.csv` and `phones_aligned.csv`.",
        "",
        "| field | dtype | description |",
        "|---|---|---|",
    ]
    for column in df.columns:
        dtype = str(df[column].dtype)
        description = FIELD_DESCRIPTIONS.get(column, "")
        lines.append(f"| `{column}` | `{dtype}` | {description} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_label_mapping(path: Path, df: pd.DataFrame) -> None:
    lines = [
        "# Label Mapping by Dataset",
        "",
        "## Shared Binary Label",
        "",
        "`gold_binary=1` means acceptable/correct target-phone realization.",
        "`gold_binary=0` means phone error, unacceptable realization, or deletion/missing phone.",
        "",
        "## Shared Three-Class Label",
        "",
        "| `gold_three_class` | Chinese label | binary mapping |",
        "|---|---|---:|",
        "| `correct` | 正确 | 1 |",
        "| `acceptable` | 可接受 | 1 |",
        "| `incorrect` | 错误或漏发 | 0 |",
        "",
        "## Dataset Mapping",
        "",
        "| dataset | rows | gold_binary=0 | gold_binary=1 | source fields | mapping note |",
        "|---|---:|---:|---:|---|---|",
    ]
    for dataset_name, group in df.groupby("dataset_source", dropna=False):
        zero = int((group["gold_binary"].astype(int) == 0).sum())
        one = int((group["gold_binary"].astype(int) == 1).sum())
        if str(dataset_name).startswith("SpeechOcean"):
            source_fields = "`source_score`, `target_phone`, `perceived_phone`, `error_type`"
            note = "SpeechOcean phone-level scores are materialized into binary and auxiliary labels in `phones_aligned.csv`/`phones.csv`."
        else:
            source_fields = "`annotation_text`, `observed_phone`, `error_type`"
            note = "L2-ARCTIC TextGrid phone annotations are parsed; plain target-phone intervals map to correct, annotated alternatives/errors map to error."
        lines.append(
            f"| {dataset_name} | {len(group)} | {zero} | {one} | {source_fields} | {note} |"
        )

    lines.extend(
        [
            "",
            "## Auxiliary Fields",
            "",
            "- `attention_binary`: broader review/attention flag.",
            "- `gold_three_class`: correct / acceptable / incorrect label.",
            "- `error_type`: source or derived error type; retained for analysis, not used as the first-round hard target.",
            "- `phone_group`: coarse phone group used by thresholding and grouped error analysis.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset_status(path: Path, manifest: list[dict[str, object]]) -> None:
    lines = [
        "# Dataset Download and Parse Status",
        "",
        "| dataset | table | rows | speakers | utterances | audio paths existing/total | annotation paths existing/total | status |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in manifest:
        audio_total = row["audio_paths_total"]
        annotation_total = row["annotation_paths_total"]
        if audio_total == "":
            audio_status = "not checked"
            annotation_status = "not checked"
        else:
            audio_status = f"{row['audio_paths_existing']}/{audio_total}"
            annotation_status = f"{row['annotation_paths_existing']}/{annotation_total}"
        status = "parsed"
        if audio_total not in ("", 0) and row["audio_paths_existing"] != audio_total:
            status = "parsed table available; raw audio path incomplete"
        lines.append(
            f"| {row['dataset_source']} | {row['table']} | {row['rows']} | {row['speakers']} | "
            f"{row['utterances']} | {audio_status} | {annotation_status} | {status} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phones", type=Path, default=Path("phones.csv"))
    parser.add_argument("--aligned", type=Path, default=Path("phones_aligned.csv"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    root = Path.cwd()
    source = pd.read_csv(args.phones, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    aligned = pd.read_csv(args.aligned, encoding="utf-8-sig", keep_default_na=False, low_memory=False)

    manifest = dataset_manifest(root, source, aligned)
    write_csv(
        args.reports_dir / "dataset_manifest_by_source.csv",
        manifest,
        [
            "dataset_source",
            "table",
            "rows",
            "utterances",
            "speakers",
            "splits",
            "alignment_quality",
            "pass_rows",
            "gold_binary_0",
            "gold_binary_1",
            "target_phones",
            "audio_paths_existing",
            "audio_paths_total",
            "annotation_paths_existing",
            "annotation_paths_total",
        ],
    )

    all_split_rows: list[dict[str, object]] = []
    all_speaker_rows: list[dict[str, object]] = []
    for dataset_name, group in source.groupby("dataset_source", dropna=False):
        dataset_key = str(dataset_name).lower().replace(".", "_").replace("-", "_")
        split = split_rows(group, str(dataset_name))
        speakers = speaker_rows(group, str(dataset_name))
        all_split_rows.extend(split)
        all_speaker_rows.extend(speakers)
        write_csv(
            args.reports_dir / f"{dataset_key}_split_manifest.csv",
            split,
            [
                "dataset_source",
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
            args.reports_dir / f"{dataset_key}_speaker_split_check.csv",
            speakers,
            [
                "dataset_source",
                "speaker_id",
                "splits",
                "num_splits",
                "leakage",
                "phone_rows",
                "utterances",
                "gold_binary_0",
                "gold_binary_1",
            ],
        )

    write_csv(
        args.reports_dir / "combined_dataset_split_manifest.csv",
        all_split_rows,
        [
            "dataset_source",
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
        args.reports_dir / "combined_speaker_split_check.csv",
        all_speaker_rows,
        [
            "dataset_source",
            "speaker_id",
            "splits",
            "num_splits",
            "leakage",
            "phone_rows",
            "utterances",
            "gold_binary_0",
            "gold_binary_1",
        ],
    )
    write_data_dictionary(args.reports_dir / "data_dictionary.md", source)
    write_label_mapping(args.reports_dir / "label_mapping_by_dataset.md", source)
    write_dataset_status(args.reports_dir / "dataset_download_parse_status.md", manifest)

    leakage = sum(int(row["leakage"]) for row in all_speaker_rows)
    print(f"Wrote dataset acceptance artifacts to {args.reports_dir}")
    print(f"Datasets: {source['dataset_source'].nunique()}; speaker leakage rows: {leakage}")


if __name__ == "__main__":
    main()
