# Enhanced and Fusion Model Report

## Scope

This report documents the added phase-1 enhanced-model and fusion-model experiments.

## Enhanced Models

Implemented in `scripts/run_enhanced_model.py`.

Models:

- `enhanced_extra_trees`
- `enhanced_hist_gradient_boosting`

Input features avoid human-score leakage and use deployable metadata:

- `target_phone`
- `phone_group`
- `duration_ms`
- `phone_index`
- `speaker_gender`
- `speaker_age`
- `native_language`

Test results:

| model | accuracy | balanced_accuracy | precision | recall | macro_f1 | error_precision | error_recall | error_f1 | auc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| enhanced_extra_trees | 0.886411 | 0.583783 | 0.967062 | 0.912761 | 0.545717 | 0.108614 | 0.254805 | 0.152306 | 0.673511 |
| enhanced_hist_gradient_boosting | 0.837875 | 0.572711 | 0.966490 | 0.860962 | 0.516948 | 0.078640 | 0.284459 | 0.123216 | 0.678104 |

## Fusion Models

Implemented in `scripts/run_fusion_model.py`.

Models:

- `fusion_logreg`
- `fusion_random_forest`

Fusion evidence:

- proxy GOP score
- phone-group threshold
- score margin
- baseline model scores
- duration
- phone identity
- phone group

Test results:

| model | accuracy | balanced_accuracy | precision | recall | macro_f1 | error_precision | error_recall | error_f1 | auc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fusion_logreg | 0.917860 | 0.533857 | 0.962697 | 0.951294 | 0.529455 | 0.090676 | 0.116420 | 0.101948 | 0.632950 |
| fusion_random_forest | 0.921027 | 0.523929 | 0.961882 | 0.955601 | 0.522146 | 0.079772 | 0.092257 | 0.085561 | 0.617124 |

## Current Interpretation

The enhanced-model module is complete as a runnable first-stage improvement over the baseline stack.
`enhanced_extra_trees` improves Macro-F1 and AUC over the original calibrated proxy-GOP baseline, but it does not beat the original global-threshold random forest on Balanced Accuracy.

The fusion-model module is implemented and produces evaluation artifacts, but the current fusion setup is a partial fusion baseline. It uses proxy GOP and tabular evidence only; true acoustic GOP and self-supervised speech embeddings are not yet included.

## Artifacts

- `scripts/run_enhanced_model.py`
- `scripts/run_fusion_model.py`
- `configs/phase1_model_stack.yaml`
- `reports/enhanced_model_metrics.csv`
- `reports/fusion_model_metrics.csv`
- `reports/model_comparison.csv`
- `reports/model_comparison.md`
