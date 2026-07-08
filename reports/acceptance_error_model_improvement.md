# Error-Precision/Recall Model Improvement

## What Changed

- Added `scripts/calibrate_plan_acceptance_thresholds.py` to scan existing deployable model scores under the phase-plan target: error precision >= 0.40 and error recall >= 0.50.
- Added `scripts/run_acceptance_error_fusion_model.py` to train a direct error detector instead of a generic correctness classifier.
- The new model uses train-only smoothed error priors by phone, phone group, word-phone pair, and dataset-phone pair.
- The new model also uses feature-model scores and SSL phone-segment embeddings.
- Thresholds are selected on dev using the plan precision constraint and reported on test.

## Current Combined Test Results

| model | calibration | BA | Macro-F1 | AUC | error_precision | error_recall | pass |
|---|---|---:|---:|---:|---:|---:|---:|
| acceptance_error_extra_trees | plan_phone_group_threshold | 0.664365 | 0.604945 | 0.787562 | 0.200345 | 0.415672 | 0 |
| acceptance_error_extra_trees | plan_global_threshold | 0.567846 | 0.593471 | 0.787562 | 0.394261 | 0.147574 | 0 |
| acceptance_error_sgd | plan_phone_group_threshold | 0.593766 | 0.584892 | 0.704907 | 0.196528 | 0.238663 | 0 |
| acceptance_error_sgd | plan_global_threshold | 0.580809 | 0.588057 | 0.704907 | 0.236575 | 0.194511 | 0 |

## Interpretation

The improved deployable model raises the best combined deployable Balanced Accuracy from 0.620598 to 0.664365 and raises the best deployable AUC to 0.787562. However, it still does not meet the plan's minimum error operating point. The closest precision-constrained point is `acceptance_error_extra_trees` with global plan threshold: error precision 0.394261, just below 0.40, but error recall is only 0.147574.

The remaining gap is therefore not only threshold calibration. The model's ranking has improved, but the acoustic evidence still does not separate enough true pronunciation errors from low-confidence correct pronunciations.

## Next Required Upgrade

To reach the plan line with a deployable model, the next iteration should replace the current proxy/acoustic evidence with a stronger phone-posterior or CTC-based GOP score, then feed that score into the acceptance fusion model.
