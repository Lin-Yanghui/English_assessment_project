# SSL Phone-Segment Model Status

## Status

The project now includes a reproducible SSL phone-segment classifier pipeline:

- `scripts/extract_ssl_phone_embeddings.py`
- `scripts/run_ssl_phone_classifier.py`
- `scripts/build_ssl_gop_comparison.py`
- `configs/ssl_phone_classifier.yaml`

## Current Environment Check

The current local Python environment is missing the optional SSL runtime dependencies:

```text
torch
torchaudio
transformers
soundfile
```

SpeechOcean762 parsed phone tables are available, but the raw SpeechOcean762 audio paths are not currently present in the workspace. L2-ARCTIC Mandarin audio and annotations are available locally.

## Interpretation

The SSL model implementation is present and reproducible, but SSL metrics cannot be generated in the current environment until:

1. Optional SSL dependencies are installed.
2. The selected SSL encoder is available locally or downloadable.
3. Raw audio files for the chosen dataset are available at the paths described in `docs/data_setup.md`.

## Comparison Target

Once embeddings are extracted and the classifier is trained, `reports/ssl_vs_gop_comparison.csv` and `reports/ssl_vs_gop_comparison.md` compare:

- proxy GOP phone-group threshold baseline
- proxy GOP target-phone threshold baseline
- SSL phone-segment classifier
