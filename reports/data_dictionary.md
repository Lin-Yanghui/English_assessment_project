# Data Dictionary

This dictionary covers the project-level phone table schema used by `phones.csv` and `phones_aligned.csv`.

| field | dtype | description |
|---|---|---|
| `utterance_id` | `object` | Unique utterance identifier. |
| `speaker_id` | `object` | Speaker identifier. Used for speaker-level train/dev/test isolation. |
| `speaker_gender` | `object` | Speaker gender metadata when available. |
| `speaker_age` | `object` | Speaker age metadata when available. |
| `native_language` | `object` | Speaker native language metadata. |
| `transcript` | `object` | Sentence transcript. |
| `sentence_accuracy` | `object` | Source sentence-level accuracy score when available. |
| `sentence_fluency` | `object` | Source sentence-level fluency score when available. |
| `sentence_completeness` | `object` | Source sentence-level completeness score when available. |
| `sentence_prosodic` | `object` | Source sentence-level prosodic score when available. |
| `word` | `object` | Word token containing the target phone. |
| `word_index` | `object` | Word index within utterance. |
| `word_accuracy` | `object` | Source word-level accuracy score when available. |
| `word_stress` | `object` | Source word stress score when available. |
| `target_phone_raw` | `object` | Original target phone label from the source parser. |
| `target_phone` | `object` | Normalized target phone label used by models. |
| `perceived_phone_raw` | `object` | Original perceived or annotated phone label when available. |
| `perceived_phone` | `object` | Normalized perceived phone label when available. |
| `phone_index` | `int64` | Phone index within utterance. |
| `start_ms` | `float64` | Phone segment start time in milliseconds. |
| `end_ms` | `float64` | Phone segment end time in milliseconds. |
| `duration_ms` | `float64` | Phone segment duration in milliseconds. |
| `source_score` | `object` | Original source phone-level score when available. |
| `gold_binary` | `int64` | Binary target label: 1 acceptable/correct, 0 error/unacceptable. |
| `attention_binary` | `object` | Broader attention or review flag. |
| `gold_three_class` | `object` | Reusable three-class label: correct, acceptable, or incorrect. |
| `error_type` | `object` | Source or derived error type. |
| `phone_group` | `object` | Coarse phone group used for grouped analysis and thresholds. |
| `dataset_source` | `object` | Dataset source name. |
| `split` | `object` | Project train/dev/test split. |
| `official_split` | `object` | Official or inherited split label where available. |
| `audio_path` | `object` | Relative path to source audio. |
| `annotation_path` | `object` | Relative path to source annotation. |
| `alignment_method` | `object` | Alignment or parsing method identifier. |
| `alignment_score` | `object` | Alignment quality score when available. |
| `alignment_quality` | `object` | Alignment quality status, e.g. pass/review. |
