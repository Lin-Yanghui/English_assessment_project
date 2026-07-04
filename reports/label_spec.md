# Label Specification

## Binary Label

`gold_binary` is the first-round training target.

| value | meaning | use |
|---:|---|---|
| 1 | acceptable/correct target phone realization | positive class |
| 0 | phone error or unacceptable realization | error class |

Current SpeechOcean mapping is already materialized in `phones_aligned.csv`.

## Auxiliary Labels

- `source_score`: original phone-level score where available.
- `attention_binary`: broader attention/needs-review flag.
- `gold_three_class`: correct / acceptable-but-accented / error-style label where available.
- `error_type`: source or derived error type. It is not treated as a first-round hard target.

## First-Round Training Filter

Use only rows with:

```text
alignment_quality == "pass"
```

Rows with `alignment_quality == "review"` are retained for analysis and later model improvement, but are excluded from the first-round training baseline.

## Split Rule

Use the existing `split` column:

```text
train / dev / test
```

Thresholds must be selected on `dev`; final metrics must be reported on `test`.
