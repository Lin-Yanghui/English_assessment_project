# Phase 1 Model Design Status

## Summary

The project now contains a reproducible first-round baseline plus enhanced tabular, proxy GOP, acoustic GOP, SSL-enhanced fusion-model, SSL phone-segment classifier, and error-focused acceptance fusion pipeline code for phone-level pronunciation correctness. The current combined main-model path uses both SpeechOcean762 and L2-ARCTIC-v5.0-Mandarin through `phones.csv` with `alignment_quality=pass,blank`. It satisfies the runnable baseline/evaluation part of the phase-1 plan and now includes a first acoustic GOP implementation plus a plan-oriented error detector that directly optimizes the error-class precision constraint. It does not yet satisfy the full deployable model-design target because deployable models still miss the planned error precision/recall line.

## Current Best Model

Best test row by Balanced Accuracy: teacher_score_rule (teacher_score_threshold), Balanced Accuracy=1.000000, Macro-F1=1.000000, AUC=1.000000. This is a source-score/label-reconstruction upper bound, not a deployable acoustic model.

Best deployable test model by Balanced Accuracy: ssl_phone_logreg (ssl_global_threshold), Balanced Accuracy=0.606427, Macro-F1=0.575467, AUC=0.738200.

Best combined deployable test model by Balanced Accuracy after adding the error-focused fusion model: acceptance_error_extra_trees (plan_phone_group_threshold), Balanced Accuracy=0.664365, Macro-F1=0.604945, AUC=0.787562, error precision=0.200345, error recall=0.415672.

Closest deployable model to the plan precision constraint: acceptance_error_extra_trees (plan_global_threshold), error precision=0.394261, error recall=0.147574, AUC=0.787562.

## Completed

- SpeechOcean phone-level aligned input is available in `phones_aligned.csv`.
- Combined SpeechOcean762 + L2-ARCTIC model input is available in `phones.csv`.
- Combined main-model metrics are generated in `reports/combined_model_comparison.csv` and `reports/combined_model_comparison.md`.
- L2-ARCTIC SSL embeddings are extracted and merged into `reports/combined_ssl_phone_embeddings.csv`.
- First-round model training is restricted to `alignment_quality == "pass"`.
- Majority-class, logistic regression, and random forest baselines are implemented.
- Proxy GOP-style score plus phone-group and target-phone threshold calibration are implemented.
- Acoustic GOP scores are implemented in `scripts/run_acoustic_gop_model.py` using a phone-posterior acoustic model trained from correct SSL phone-segment embeddings.
- Acoustic GOP phone-group and target-phone threshold calibration outputs are generated in `reports/acoustic_gop_*.csv`.
- Enhanced tabular models are implemented in `scripts/run_enhanced_model.py`.
- Fusion models are implemented in `scripts/run_fusion_model.py` using proxy GOP scores, threshold margins, duration, phone identity, phone group, and 768-dimensional SSL phone-segment embeddings.
- Error-focused acceptance fusion is implemented in `scripts/run_acceptance_error_fusion_model.py`; it predicts the error class directly, adds train-only smoothed phone/word priors, existing feature-model scores, SSL features, and dev-selected plan thresholds.
- Plan acceptance threshold scanning is implemented in `scripts/calibrate_plan_acceptance_thresholds.py` and generated in `reports/combined_plan_acceptance_threshold_metrics.csv`.
- SSL embedding extraction and phone-segment classifier scripts are implemented and measured in `scripts/extract_ssl_phone_embeddings.py`, `scripts/run_ssl_phone_classifier.py`, and `reports/ssl_phone_classifier_metrics.csv`.
- A teacher-score upper-bound rule is implemented in `scripts/run_teacher_score_rule.py` and measured in `reports/teacher_score_rule_metrics.csv` plus `reports/combined_teacher_score_rule_metrics.csv`.
- Formal test metrics, confusion matrices, 100-utterance prediction samples, model comparison, and error analysis are generated.
- Reproducibility instructions are documented in `README.md`.

## Partial Or Missing

- The first acoustic GOP implementation is available, but its current test metrics are weak: Balanced Accuracy=0.539829, Macro-F1=0.162142, AUC=0.564236.
- Current combined fusion includes SSL embeddings and train/dev/test proxy GOP predictions, so it trains on the train split.
- The teacher-score rule meets the numerical target line, but deployable acoustic/SSL/fusion models still do not meet the planned error precision/recall line. The improved acceptance fusion raises deployable AUC to 0.787562 and Balanced Accuracy to 0.664365, but its best test operating points remain below `error_precision>=0.40` and `error_recall>=0.50` simultaneously.
- `review` alignment rows are retained but excluded from first-round training.

## Recommended Next Iteration

1. Improve acoustic GOP by replacing the GaussianNB posterior model with a stronger phone CTC/acoustic posterior model.
2. Replace label-derived teacher scoring with stronger deployable acoustic evidence, for example CTC or frame-level phone posterior GOP.
3. Re-tune thresholds for the error-detection operating point required by the plan.
4. Re-run comparison with majority baseline, proxy GOP, acoustic GOP, tabular model, SSL model, and SSL fusion model.
5. Focus error reduction on vowels, fricatives, stops, nasals, and liquids based on `reports/error_analysis_summary.md`.
