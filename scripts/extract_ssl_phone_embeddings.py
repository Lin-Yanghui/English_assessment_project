"""Extract self-supervised speech embeddings for phone segments.

This script requires optional runtime dependencies:

- torch
- transformers
- soundfile
- numpy

It intentionally keeps those dependencies optional so the rest of the project can
run in lightweight environments.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import pandas as pd


METADATA_COLUMNS = [
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


DEFAULT_PATH_REMAP = [
    ("data/raw/speechocean762/", "speechocean762/"),
    ("data/raw/L2-ARCTIC-v5.0/Mandarin/", "./"),
]


def require_optional_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        import soundfile as sf
        import torch
        from transformers import AutoFeatureExtractor, AutoModel
    except ImportError as exc:
        raise SystemExit(
            "Missing optional SSL dependencies. Install them with:\n"
            "pip install torch torchaudio transformers soundfile\n\n"
            f"Original import error: {exc}"
        ) from exc
    return np, sf, torch, (AutoFeatureExtractor, AutoModel)


def resolve_audio_path(root: Path, raw_path: str, remap: list[tuple[str, str]]) -> Path:
    normalized = raw_path.replace("\\", "/")
    candidates = [root / raw_path, Path(raw_path)]
    for src, dst in remap:
        if normalized.startswith(src):
            candidates.append(root / dst / normalized[len(src) :])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return root / raw_path


def parse_remap(items: list[str]) -> list[tuple[str, str]]:
    remap: list[tuple[str, str]] = list(DEFAULT_PATH_REMAP)
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid --path-remap item {item!r}; expected FROM=TO.")
        src, dst = item.split("=", 1)
        remap.append((src.replace("\\", "/"), dst))
    return remap


def resample_if_needed(np: Any, audio: Any, source_rate: int, target_rate: int) -> Any:
    if source_rate == target_rate:
        return audio
    old_x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    new_len = max(1, int(round(len(audio) * target_rate / source_rate)))
    new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
    return np.interp(new_x, old_x, audio).astype("float32")


def segment_audio(
    np: Any,
    audio: Any,
    sample_rate: int,
    start_ms: float,
    end_ms: float,
    min_segment_ms: float,
) -> Any:
    start = max(0, int(round(start_ms * sample_rate / 1000)))
    end = min(len(audio), int(round(end_ms * sample_rate / 1000)))
    if end <= start:
        end = min(len(audio), start + 1)
    segment = audio[start:end]
    min_samples = max(1, int(round(min_segment_ms * sample_rate / 1000)))
    if len(segment) < min_samples:
        segment = np.pad(segment, (0, min_samples - len(segment)), mode="constant")
    return segment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("phones_aligned.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/ssl_phone_embeddings.csv"))
    parser.add_argument("--model-name-or-path", default="facebook/wav2vec2-base")
    parser.add_argument("--dataset-source", default="")
    parser.add_argument("--alignment-quality", default="pass")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--min-segment-ms", type=float, default=40.0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--cache-size", type=int, default=32)
    parser.add_argument("--path-remap", nargs="*", default=[])
    args = parser.parse_args()

    np, sf, torch, hf = require_optional_dependencies()
    AutoFeatureExtractor, AutoModel = hf

    root = Path.cwd()
    remap = parse_remap(args.path_remap)
    df = pd.read_csv(args.input, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    if args.dataset_source:
        df = df[df["dataset_source"] == args.dataset_source].copy()
    if args.alignment_quality.lower() != "all" and "alignment_quality" in df.columns:
        df = df[df["alignment_quality"].str.lower() == args.alignment_quality.lower()].copy()
    if args.max_rows:
        df = df.head(args.max_rows).copy()
    if df.empty:
        raise SystemExit("No rows available for SSL embedding extraction.")

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [col for col in METADATA_COLUMNS if col in df.columns]
    embedding_fields: list[str] | None = None

    audio_cache: dict[str, tuple[Any, int]] = {}
    hidden_cache: dict[str, Any] = {}
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer: csv.DictWriter[str] | None = None
        written = 0
        missing_audio = 0
        for index, row in df.iterrows():
            audio_path = resolve_audio_path(root, str(row["audio_path"]), remap)
            if not audio_path.exists():
                missing_audio += 1
                continue
            cache_key = str(audio_path)
            if cache_key not in audio_cache:
                audio, sr = sf.read(audio_path)
                if getattr(audio, "ndim", 1) > 1:
                    audio = audio.mean(axis=1)
                audio = audio.astype("float32")
                audio = resample_if_needed(np, audio, sr, args.sample_rate)
                audio_cache[cache_key] = (audio, args.sample_rate)
                while args.cache_size and len(audio_cache) > args.cache_size:
                    audio_cache.pop(next(iter(audio_cache)))
            audio, sr = audio_cache[cache_key]
            if cache_key not in hidden_cache:
                inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
                with torch.no_grad():
                    hidden_cache[cache_key] = model(**inputs).last_hidden_state.squeeze(0).cpu().numpy()
                while args.cache_size and len(hidden_cache) > args.cache_size:
                    hidden_cache.pop(next(iter(hidden_cache)))
            hidden = hidden_cache[cache_key]
            start_sample = max(0, int(round(float(row["start_ms"]) * sr / 1000)))
            end_sample = min(len(audio), int(round(float(row["end_ms"]) * sr / 1000)))
            if end_sample <= start_sample:
                end_sample = min(len(audio), start_sample + max(1, int(args.min_segment_ms * sr / 1000)))
            start_frame = int(np.floor(start_sample * len(hidden) / max(1, len(audio))))
            end_frame = int(np.ceil(end_sample * len(hidden) / max(1, len(audio))))
            start_frame = min(max(0, start_frame), max(0, len(hidden) - 1))
            end_frame = min(max(start_frame + 1, end_frame), len(hidden))
            vector = hidden[start_frame:end_frame].mean(axis=0)
            if embedding_fields is None:
                embedding_fields = [f"ssl_{i}" for i in range(len(vector))]
                writer = csv.DictWriter(f, fieldnames=fields + embedding_fields)
                writer.writeheader()
            assert writer is not None
            out = {col: row[col] for col in fields}
            out.update({name: float(value) for name, value in zip(embedding_fields, vector)})
            writer.writerow(out)
            written += 1
            if args.progress_every and written % args.progress_every == 0:
                print(f"Processed {written} rows; cached_audio={len(audio_cache)}; missing_audio={missing_audio}")

    if embedding_fields is None:
        raise SystemExit("No embeddings were written. Check audio paths and path remapping.")
    print(f"Wrote SSL embeddings to {args.output}")


if __name__ == "__main__":
    main()
