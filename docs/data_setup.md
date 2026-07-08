# Data Setup

This project uses two public pronunciation datasets:

- SpeechOcean762
- L2-ARCTIC Mandarin speakers

## Expected Parsed Tables

The current project-level parsed tables are:

- `phones.csv`: combined source table containing SpeechOcean762 and L2-ARCTIC Mandarin rows.
- `phones_aligned.csv`: primary first-round SpeechOcean762 model input.

## SpeechOcean762

1. Download SpeechOcean762 from OpenSLR SLR101 according to its license.
2. Place the raw files under the current local project directory:

```text
speechocean762/
```

3. The parsed tables still keep the original portable audio paths:

```text
data/raw/speechocean762/WAVE/SPEAKERxxxx/xxxxxxxxx.WAV
```

The project scripts remap those paths to the local raw-data directory:

```text
data/raw/speechocean762/WAVE/... -> speechocean762/WAVE/...
```

4. The parsed SpeechOcean762 phone-level table is materialized in:

```text
phones_aligned.csv
```

## L2-ARCTIC Mandarin

The current local Mandarin speaker folders are:

```text
BWC/
LXC/
NCC/
TXHC/
```

Each speaker folder is expected to contain:

```text
annotation/
textgrid/
transcript/
wav/
```

To rebuild a standalone L2-ARCTIC manifest:

```powershell
python scripts\build_l2arctic_manifest.py --root . --output data\l2arctic_small_manifest.csv
```

## Dataset Acceptance Artifacts

To regenerate dataset-level acceptance materials:

```powershell
python scripts\generate_dataset_acceptance_artifacts.py --phones phones.csv --aligned phones_aligned.csv --reports-dir reports
```

This produces:

```text
reports/dataset_manifest_by_source.csv
reports/speechocean762_split_manifest.csv
reports/speechocean762_speaker_split_check.csv
reports/l2_arctic_v5_0_mandarin_split_manifest.csv
reports/l2_arctic_v5_0_mandarin_speaker_split_check.csv
reports/combined_dataset_split_manifest.csv
reports/combined_speaker_split_check.csv
reports/data_dictionary.md
reports/label_mapping_by_dataset.md
reports/dataset_download_parse_status.md
```

## Split Rule

All model evaluation splits must be speaker-isolated. A speaker is valid only if it appears in exactly one of:

```text
train
dev
test
```
