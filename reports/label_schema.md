# Label Schema

This project keeps two reusable label views for phone-level pronunciation correctness:

- binary label: `gold_binary`
- three-class label: `gold_three_class`

## Three-Class Label

| `gold_three_class` | Chinese label | Meaning | Typical source evidence |
|---|---|---|---|
| `correct` | 正确 | Target phone is realized correctly and is acceptable without a pronunciation-error flag. | SpeechOcean high phone score; L2-ARCTIC plain target-phone interval without error annotation. |
| `acceptable` | 可接受 | Target phone remains identifiable/acceptable, but may contain accent or non-critical quality issues. | SpeechOcean middle phone score or equivalent acceptable label. |
| `incorrect` | 错误或漏发 | Target phone is wrong, substituted, deleted, inserted around, or otherwise unacceptable. | SpeechOcean low phone score/error flag; L2-ARCTIC annotated substitution/deletion/insertion/unknown error. |

## Binary Label

| `gold_binary` | Meaning | Derived from `gold_three_class` |
|---:|---|---|
| `1` | 正确或可接受 | `correct` + `acceptable` |
| `0` | 错误或漏发 | `incorrect` |

The first-round model uses `gold_binary` as the main training target.
`gold_three_class` is retained for analysis, reporting, and later expansion to three-class modeling.

## Dataset-Specific Mapping

### SpeechOcean762

SpeechOcean762 provides phone-level scoring and error-related fields. The parsed tables materialize:

- `source_score`
- `gold_three_class`
- `gold_binary`
- `attention_binary`
- `error_type`

Mapping:

| SpeechOcean-derived class | `gold_three_class` | `gold_binary` |
|---|---|---:|
| correct/high-quality phone realization | `correct` | `1` |
| acceptable/accented but identifiable phone realization | `acceptable` | `1` |
| wrong, unacceptable, or missing phone realization | `incorrect` | `0` |

Current SpeechOcean762 counts in `phones_aligned.csv`:

| label | count |
|---|---:|
| `correct` | 82,343 |
| `acceptable` | 7,560 |
| `incorrect` | 4,542 |

### L2-ARCTIC Mandarin

L2-ARCTIC phone annotations are parsed from TextGrid intervals.

Mapping:

| TextGrid annotation pattern | `gold_three_class` | `gold_binary` |
|---|---|---:|
| Plain target-phone interval, no error annotation | `correct` | `1` |
| Annotated substitution / deletion / insertion / unknown error | `incorrect` | `0` |

L2-ARCTIC does not provide a stable middle "acceptable" score in the same way as SpeechOcean762, so current L2-ARCTIC rows use only:

- `correct`
- `incorrect`

Current L2-ARCTIC Mandarin counts in `phones.csv`:

| label | count |
|---|---:|
| `correct` | 16,740 |
| `acceptable` | 0 |
| `incorrect` | 3,244 |

## Reusable Fields

Any downstream model, evaluation script, or diagnostic output should use:

```text
gold_binary
gold_three_class
error_type
attention_binary
```

Recommended use:

- `gold_binary`: first-round binary training and evaluation.
- `gold_three_class`: three-class analysis and future three-class model training.
- `error_type`: error subtype analysis, not the first-round hard target.
- `attention_binary`: review/attention flag for later data curation.
