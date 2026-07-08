# G6 Minimal Demo: Phone-Level Diagnosis CLI

## Goal

This interface satisfies the G6 minimum demo requirement: given an input audio file and expected text, it outputs a phone-level diagnosis table.

## Command

```powershell
python scripts/run_minimal_phone_demo.py `
  --audio speechocean762\WAVE\SPEAKER0001\000010011.WAV `
  --text "ability" `
  --utterance-id g6_demo_000010011 `
  --speaker-id SPEAKER0001 `
  --output reports\g6_minimal_demo_diagnosis.csv
```

## Output

Example output:

```text
reports/g6_minimal_demo_diagnosis.csv
```

The output table contains:

- `utterance_id`
- `speaker_id`
- `word`
- `word_index`
- `target_phone`
- `phone_index`
- `start_ms`
- `end_ms`
- `duration_ms`
- `prediction`
- `predicted_label`
- `confidence`
- `error_probability`
- `gop_score`
- `phone_group`
- `error_type_hint`
- `dataset_source`
- `audio_path`
- `model`
- `calibration`
- `threshold`
- `diagnosis_note`

## Prediction Convention

- `prediction=1`: acceptable/correct
- `prediction=0`: incorrect or needs review

## Implementation Notes

- Text-to-phone conversion uses `speechocean762/resource/lexicon.txt`.
- The minimum demo uses uniform phone segmentation from the input WAV duration.
- The scorer uses `models/combined/acceptance_error_extra_trees.joblib` when available.
- If the model is missing, the script falls back to train-set phone and phone-group error priors.

## Current Limitation

This is a minimum demo pipeline, not a production forced-alignment system. The phone boundaries are uniform fallback boundaries. For a stronger second-stage demo, replace the uniform segmentation step with MFA/Kaldi/CTC forced alignment and feed real acoustic posterior GOP scores into the same output schema.
