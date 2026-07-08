# Label Specification

## Binary Label

`gold_binary` is the first-round training target.

| value | meaning | use |
|---:|---|---|
| 1 | correct or acceptable target-phone realization | positive class |
| 0 | phone error, unacceptable realization, or deletion/missing phone | error class |

Current SpeechOcean mapping is already materialized in `phones_aligned.csv`.

## Three-Class Label

`gold_three_class` is the reusable three-class target.

| value | Chinese label | meaning | binary mapping |
|---|---|---|---:|
| `correct` | 正确 | target phone is realized correctly | 1 |
| `acceptable` | 可接受 | target phone is identifiable/acceptable but may be accented | 1 |
| `incorrect` | 错误或漏发 | target phone is wrong, unacceptable, substituted, deleted, inserted around, or missing | 0 |

## Auxiliary Labels

- `source_score`: original phone-level score where available.
- `attention_binary`: broader attention/needs-review flag.
- `gold_three_class`: correct / acceptable / incorrect label.
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
