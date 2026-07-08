"""Run a minimal phone-level pronunciation diagnosis demo.

Input: an audio file and the expected text.
Output: a phone-level diagnosis CSV.

This is a G6 minimum demo interface. It uses a lexicon-based text-to-phone
step and a lightweight uniform phone segmentation fallback, then scores each
target phone with the trained acceptance-error fusion model when available.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import wave
from pathlib import Path

import joblib
import pandas as pd


PHONE_GROUPS = {
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
    "stop": {"P", "B", "T", "D", "K", "G"},
    "fricative": {"F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"},
    "affricate": {"CH", "JH"},
    "nasal": {"M", "N", "NG"},
    "liquid": {"L", "R"},
    "glide": {"W", "Y"},
}

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

OUTPUT_FIELDS = [
    "utterance_id",
    "speaker_id",
    "word",
    "word_index",
    "target_phone",
    "phone_index",
    "start_ms",
    "end_ms",
    "duration_ms",
    "prediction",
    "predicted_label",
    "confidence",
    "error_probability",
    "gop_score",
    "phone_group",
    "error_type_hint",
    "dataset_source",
    "audio_path",
    "model",
    "calibration",
    "threshold",
    "diagnosis_note",
]


def strip_stress(phone: str) -> str:
    return re.sub(r"\d+$", "", phone.strip().upper())


def phone_group(phone: str) -> str:
    base = strip_stress(phone)
    for group, phones in PHONE_GROUPS.items():
        if base in phones:
            return group
    return "other"


def load_lexicon(path: Path) -> dict[str, list[str]]:
    lexicon: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = re.sub(r"\(\d+\)$", "", parts[0].upper())
            phones = [strip_stress(phone) for phone in parts[1:]]
            lexicon.setdefault(word, phones)
    return lexicon


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.upper())


def text_to_phone_rows(text: str, lexicon: dict[str, list[str]], oov_phone: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    phone_index = 0
    for word_index, word in enumerate(tokenize(text)):
        normalized = word.replace("'", "")
        phones = lexicon.get(normalized)
        if phones is None:
            phones = []
            for char in normalized:
                phones.extend(lexicon.get(char, []))
            if not phones:
                phones = [oov_phone]
        for phone in phones:
            target_phone = strip_stress(phone)
            rows.append(
                {
                    "word": normalized,
                    "word_index": word_index,
                    "target_phone": target_phone,
                    "phone_group": phone_group(target_phone),
                    "phone_index": phone_index,
                }
            )
            phone_index += 1
    return rows


def wav_duration_ms(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
        if rate <= 0:
            raise ValueError(f"Invalid WAV sample rate for {path}")
        return frames / rate * 1000.0


def add_uniform_segments(rows: list[dict[str, object]], duration_ms: float) -> None:
    if not rows:
        return
    segment_ms = duration_ms / len(rows)
    for i, row in enumerate(rows):
        start_ms = round(i * segment_ms, 3)
        end_ms = round((i + 1) * segment_ms, 3)
        row["start_ms"] = start_ms
        row["end_ms"] = end_ms
        row["duration_ms"] = round(end_ms - start_ms, 3)


def smoothed_prior(
    train: pd.DataFrame,
    group_cols: list[str],
    smoothing: float = 20.0,
) -> tuple[dict[tuple[str, ...], float], float]:
    global_error_rate = float((1 - train["gold_binary"].astype(int)).mean())
    work = train.copy()
    work["error_label"] = 1 - work["gold_binary"].astype(int)
    grouped = work.groupby(group_cols, dropna=False)["error_label"].agg(["sum", "count"]).reset_index()
    values: dict[tuple[str, ...], float] = {}
    for _, row in grouped.iterrows():
        key = tuple(str(row[col]) for col in group_cols)
        values[key] = float((row["sum"] + smoothing * global_error_rate) / (row["count"] + smoothing))
    return values, global_error_rate


def add_train_priors(frame: pd.DataFrame, phones_path: Path) -> pd.DataFrame:
    if not phones_path.exists():
        for col in [
            "phone_train_error_rate",
            "phone_group_train_error_rate",
            "word_phone_train_error_rate",
            "dataset_phone_train_error_rate",
        ]:
            frame[col] = 0.05
        return frame

    source = pd.read_csv(phones_path, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    train = source[source["split"] == "train"].copy()
    if train.empty:
        train = source.copy()
    train["gold_binary"] = pd.to_numeric(train["gold_binary"], errors="coerce").fillna(1).astype(int)
    for col in ["target_phone", "phone_group", "word", "dataset_source"]:
        if col in train.columns:
            train[col] = train[col].astype(str)

    prior_specs = [
        (["target_phone"], "phone_train_error_rate"),
        (["phone_group"], "phone_group_train_error_rate"),
        (["word", "target_phone"], "word_phone_train_error_rate"),
        (["dataset_source", "target_phone"], "dataset_phone_train_error_rate"),
    ]
    for group_cols, out_col in prior_specs:
        values, fallback = smoothed_prior(train, group_cols)
        rates = []
        for _, row in frame.iterrows():
            key = tuple(str(row.get(col, "")) for col in group_cols)
            rates.append(values.get(key, fallback))
        frame[out_col] = rates
    return frame


def load_group_thresholds(predictions_path: Path, model_name: str, default_threshold: float) -> dict[str, float]:
    if not predictions_path.exists():
        return {}
    usecols = ["model", "calibration", "phone_group", "threshold"]
    try:
        df = pd.read_csv(predictions_path, encoding="utf-8-sig", usecols=usecols, keep_default_na=False)
    except ValueError:
        return {}
    df = df[
        (df["model"] == model_name)
        & (df["calibration"] == "plan_phone_group_threshold")
        & (df["threshold"] != "")
    ].copy()
    if df.empty:
        return {}
    df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")
    df = df.dropna(subset=["threshold"])
    thresholds: dict[str, float] = {}
    for group, group_df in df.groupby("phone_group"):
        if not group_df.empty:
            thresholds[str(group)] = float(group_df["threshold"].mode().iloc[0])
    thresholds.setdefault("__default__", default_threshold)
    return thresholds


def model_error_probabilities(model_path: Path, frame: pd.DataFrame) -> tuple[list[float], str]:
    if not model_path.exists():
        prior_score = frame[["phone_train_error_rate", "phone_group_train_error_rate"]].mean(axis=1)
        return [float(score) for score in prior_score], "prior_fallback"

    model = joblib.load(model_path)
    for col in BASE_CATEGORICAL:
        if col not in frame.columns:
            frame[col] = ""
        frame[col] = frame[col].fillna("").astype(str)
    for col in BASE_NUMERIC:
        if col not in frame.columns:
            frame[col] = math.nan
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    missing_ssl = {f"ssl_{i}": math.nan for i in range(96) if f"ssl_{i}" not in frame.columns}
    if missing_ssl:
        frame = pd.concat([frame, pd.DataFrame(missing_ssl, index=frame.index)], axis=1)
    feature_cols = BASE_NUMERIC + [f"ssl_{i}" for i in range(96)] + BASE_CATEGORICAL
    estimator = model.named_steps["model"]
    classes = list(estimator.classes_)
    error_index = classes.index(1)
    scores = [float(score[error_index]) for score in model.predict_proba(frame[feature_cols])]
    return scores, model_path.stem


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_demo(args: argparse.Namespace) -> list[dict[str, object]]:
    audio_path = args.audio.resolve()
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")
    if not args.lexicon.exists():
        raise SystemExit(f"Lexicon not found: {args.lexicon}")

    lexicon = load_lexicon(args.lexicon)
    rows = text_to_phone_rows(args.text, lexicon, args.oov_phone)
    if not rows:
        raise SystemExit("No target phones were produced from the input text.")

    duration_ms = args.duration_ms if args.duration_ms is not None else wav_duration_ms(audio_path)
    add_uniform_segments(rows, duration_ms)
    frame = pd.DataFrame(rows)
    frame["utterance_id"] = args.utterance_id
    frame["speaker_id"] = args.speaker_id
    frame["dataset_source"] = "demo"
    frame["native_language"] = args.native_language
    frame["speaker_age"] = args.speaker_age
    frame["feature_logreg_score"] = pd.NA
    frame["feature_random_forest_score"] = pd.NA
    frame = add_train_priors(frame, args.phones)

    scores, model_name = model_error_probabilities(args.model, frame)
    global_threshold = args.threshold
    group_thresholds = load_group_thresholds(args.threshold_predictions, model_name, global_threshold)

    output_rows: list[dict[str, object]] = []
    for (_, row), error_probability in zip(frame.iterrows(), scores):
        threshold = group_thresholds.get(str(row["phone_group"]), group_thresholds.get("__default__", global_threshold))
        prediction = 0 if error_probability >= threshold else 1
        predicted_label = "incorrect" if prediction == 0 else "acceptable"
        output_rows.append(
            {
                "utterance_id": args.utterance_id,
                "speaker_id": args.speaker_id,
                "word": row["word"],
                "word_index": int(row["word_index"]),
                "target_phone": row["target_phone"],
                "phone_index": int(row["phone_index"]),
                "start_ms": row["start_ms"],
                "end_ms": row["end_ms"],
                "duration_ms": row["duration_ms"],
                "prediction": prediction,
                "predicted_label": predicted_label,
                "confidence": round(max(error_probability, 1 - error_probability), 6),
                "error_probability": round(error_probability, 6),
                "gop_score": round(1 - error_probability, 6),
                "phone_group": row["phone_group"],
                "error_type_hint": "needs_review" if prediction == 0 else "",
                "dataset_source": "demo",
                "audio_path": str(audio_path),
                "model": model_name,
                "calibration": "plan_phone_group_threshold" if group_thresholds else "plan_global_threshold",
                "threshold": round(threshold, 6),
                "diagnosis_note": "uniform_segmentation_minimal_demo",
            }
        )
    return output_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True, help="Input WAV audio path.")
    parser.add_argument("--text", required=True, help="Expected transcript text.")
    parser.add_argument("--output", type=Path, default=Path("reports/minimal_phone_demo_diagnosis.csv"))
    parser.add_argument("--utterance-id", default="demo_utt")
    parser.add_argument("--speaker-id", default="demo_speaker")
    parser.add_argument("--native-language", default="Mandarin")
    parser.add_argument("--speaker-age", type=float, default=0.0)
    parser.add_argument("--duration-ms", type=float, default=None, help="Optional duration override for non-WAV input.")
    parser.add_argument("--threshold", type=float, default=0.77)
    parser.add_argument("--oov-phone", default="SPN")
    parser.add_argument("--lexicon", type=Path, default=Path("speechocean762/resource/lexicon.txt"))
    parser.add_argument("--phones", type=Path, default=Path("phones.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/combined/acceptance_error_extra_trees.joblib"))
    parser.add_argument(
        "--threshold-predictions",
        type=Path,
        default=Path("reports/combined_acceptance_error_fusion_predictions.csv"),
    )
    args = parser.parse_args()

    rows = run_demo(args)
    write_csv(args.output, rows)
    print(f"Wrote {len(rows)} phone-level diagnosis rows to {args.output}")
    print("Prediction convention: prediction=1 acceptable/correct, prediction=0 incorrect/needs review")


if __name__ == "__main__":
    main()
