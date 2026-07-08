# SSL Phone Classifier Reference Data Sources

## SpeechOcean762

Primary first-stage pronunciation assessment dataset.

- Source: OpenSLR SLR101 SpeechOcean762
- Project role: main phone-level pronunciation correctness table and first-round model input.
- Current parsed files:
  - `phones.csv`
  - `phones_aligned.csv`
- Current local raw-audio status: parsed table available; raw SpeechOcean762 audio paths are not currently present in the workspace.
- Official reference: https://www.openslr.org/101/

OpenSLR describes SpeechOcean762 as a pronunciation scoring dataset independently labeled by five experts, with 5000 English sentences from non-native Mandarin speakers.

## L2-ARCTIC Mandarin Speakers

Auxiliary mispronunciation annotation source.

- Local speakers used in this project:
  - `BWC`
  - `LXC`
  - `NCC`
  - `TXHC`
- Project role: auxiliary Mandarin L1 phone-level error evidence and error-type mapping.
- Current local raw-audio status: 600/600 audio paths and 600/600 annotation paths are available after project path remapping.
- Official reference: https://psi.engr.tamu.edu/l2-arctic-corpus-docs/

The L2-ARCTIC documentation describes manual phone annotations for substitution, deletion, and addition errors and gives the speaker directory structure with `wav`, `transcript`, `textgrid`, and `annotation` folders.

## Self-Supervised Speech Models

The SSL phone-segment pipeline is designed for HuggingFace-compatible encoders such as:

- wav2vec 2.0: `facebook/wav2vec2-base`
- HuBERT: `facebook/hubert-base-ls960`

References:

- wav2vec 2.0: https://arxiv.org/abs/2006.11477
- HuBERT: https://arxiv.org/abs/2106.07447

## Reproducible Commands

Install optional SSL dependencies:

```powershell
pip install torch torchaudio transformers soundfile
```

Extract SSL phone-segment embeddings:

```powershell
python scripts\extract_ssl_phone_embeddings.py --input phones_aligned.csv --output reports\ssl_phone_embeddings.csv --model-name-or-path facebook/wav2vec2-base --dataset-source SpeechOcean762 --alignment-quality pass --path-remap data/raw/speechocean762/=data/raw/speechocean762/
```

Train the SSL phone-segment classifier:

```powershell
python scripts\run_ssl_phone_classifier.py --embeddings reports\ssl_phone_embeddings.csv --metrics-output reports\ssl_phone_classifier_metrics.csv --predictions-output reports\ssl_phone_classifier_predictions.csv --model-dir models
```

Compare SSL against GOP:

```powershell
python scripts\build_ssl_gop_comparison.py --gop-metrics reports\proxy_gop_metrics.csv --ssl-metrics reports\ssl_phone_classifier_metrics.csv --output reports\ssl_vs_gop_comparison.csv --summary-output reports\ssl_vs_gop_comparison.md
```
