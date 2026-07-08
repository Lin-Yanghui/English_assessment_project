# Label Mapping by Dataset

## Shared Binary Label

`gold_binary=1` means acceptable/correct target-phone realization.
`gold_binary=0` means phone error, unacceptable realization, or deletion/missing phone.

## Shared Three-Class Label

| `gold_three_class` | Chinese label | binary mapping |
|---|---|---:|
| `correct` | 正确 | 1 |
| `acceptable` | 可接受 | 1 |
| `incorrect` | 错误或漏发 | 0 |

## Dataset Mapping

| dataset | rows | gold_binary=0 | gold_binary=1 | source fields | mapping note |
|---|---:|---:|---:|---|---|
| L2-ARCTIC-v5.0-Mandarin | 19984 | 3244 | 16740 | `annotation_text`, `observed_phone`, `error_type` | L2-ARCTIC TextGrid phone annotations are parsed; plain target-phone intervals map to correct, annotated alternatives/errors map to error. |
| SpeechOcean762 | 94445 | 4542 | 89903 | `source_score`, `target_phone`, `perceived_phone`, `error_type` | SpeechOcean phone-level scores are materialized into binary and auxiliary labels in `phones_aligned.csv`/`phones.csv`. |

## Auxiliary Fields

- `attention_binary`: broader review/attention flag.
- `gold_three_class`: correct / acceptable / incorrect label.
- `error_type`: source or derived error type; retained for analysis, not used as the first-round hard target.
- `phone_group`: coarse phone group used by thresholding and grouped error analysis.
