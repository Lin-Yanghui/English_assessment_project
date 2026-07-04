# Phase 1 Model Design Status

## Summary

The project now contains a reproducible first-round baseline for phone-level pronunciation correctness. It satisfies the runnable baseline/evaluation part of the phase-1 plan, but it does not yet satisfy the full model-design target because true acoustic GOP, self-supervised speech embeddings, and fusion modeling are still missing.

## Current Best Model

Best test model by Balanced Accuracy: feature_random_forest (global_threshold), Balanced Accuracy=0.604845, Macro-F1=0.384813, AUC=0.645927.

## Completed

- SpeechOcean phone-level aligned input is available in `phones_aligned.csv`.
- First-round model training is restricted to `alignment_quality == "pass"`.
- Majority-class, logistic regression, and random forest baselines are implemented.
- Proxy GOP-style score and phone-group threshold calibration are implemented.
- Formal test metrics, confusion matrices, 100-utterance prediction samples, model comparison, and error analysis are generated.
- Reproducibility instructions are documented in `README.md`.

## Partial Or Missing

- True acoustic GOP is not implemented. Current `proxy_gop_score` is a model probability, not a likelihood/posterior GOP score.
- Self-supervised representation models such as wav2vec2, HuBERT, WavLM, or XLS-R are not implemented.
- Fusion of GOP, duration, phone identity, phone group, and self-supervised embeddings is not implemented.
- Current metrics do not meet the target line from the plan.
- `review` alignment rows are retained but excluded from first-round training.

## Recommended Next Iteration

1. Add real acoustic features or GOP scores from an acoustic model.
2. Add per-target-phone thresholds with fallback to phone-group thresholds.
3. Add a frozen self-supervised embedding extractor once model files and audio paths are available.
4. Re-run comparison with majority baseline, GOP baseline, tabular model, SSL model, and fusion model.
5. Focus error reduction on vowels, fricatives, stops, nasals, and liquids based on `reports/error_analysis_summary.md`.
