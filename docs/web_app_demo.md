# Web App Demo

## Start

```powershell
python webapp/app.py --host 127.0.0.1 --port 7860
```

Then open:

```text
http://127.0.0.1:7860
```

## API

```text
POST /api/diagnose
```

Multipart fields:

- `audio`: WAV audio file
- `text`: expected English text
- `utterance_id`: optional utterance id
- `speaker_id`: optional speaker id

The API returns phone-level rows with:

- target phone
- phone group
- start/end time
- prediction
- confidence
- error probability
- GOP-style score

## Current Model

The app calls:

```text
models/combined/acceptance_error_extra_trees.joblib
```

The current version performs binary diagnosis:

- `prediction=1`: acceptable/correct
- `prediction=0`: incorrect/needs review

Attribution is intentionally left for the next stage.

## Limitation

This is a minimum demonstrator. The current boundary assignment uses the G6 uniform segmentation fallback. A later version should replace that step with forced alignment or CTC phone posterior GOP.
