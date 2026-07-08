# Model Comparison

Sorted by test Balanced Accuracy, then Macro-F1 and AUC.

| rank | model | calibration | accuracy | balanced_accuracy | precision | recall | macro_f1 | error_precision | error_recall | error_f1 | auc | tn | fp | fn | tp |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | teacher_score_rule | teacher_score_threshold | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1821 | 0 | 0 | 43650 |
| 2 | ssl_phone_logreg | ssl_global_threshold | 0.910690 | 0.606427 | 0.968764 | 0.937182 | 0.575467 | 0.154747 | 0.275673 | 0.198223 | 0.738200 | 502 | 1319 | 2742 | 40908 |
| 3 | feature_random_forest | global_threshold | 0.513118 | 0.604845 | 0.976181 | 0.505132 | 0.384813 | 0.056065 | 0.704558 | 0.103866 | 0.645927 | 1283 | 538 | 21601 | 22049 |
| 4 | enhanced_extra_trees | enhanced_global_threshold | 0.886411 | 0.583783 | 0.967062 | 0.912761 | 0.545717 | 0.108614 | 0.254805 | 0.152306 | 0.673511 | 464 | 1357 | 3808 | 39842 |
| 5 | feature_logreg | global_threshold | 0.643839 | 0.582945 | 0.969878 | 0.649141 | 0.440925 | 0.057886 | 0.516749 | 0.104110 | 0.623729 | 941 | 880 | 15315 | 28335 |
| 6 | enhanced_hist_gradient_boosting | enhanced_global_threshold | 0.837875 | 0.572711 | 0.966490 | 0.860962 | 0.516948 | 0.078640 | 0.284459 | 0.123216 | 0.678104 | 518 | 1303 | 6069 | 37581 |
| 7 | feature_random_forest | phone_group_threshold | 0.877988 | 0.557820 | 0.964885 | 0.905865 | 0.527742 | 0.085059 | 0.209775 | 0.121039 | 0.645927 | 382 | 1439 | 4109 | 39541 |
| 8 | fusion_logreg | fusion_global_threshold | 0.896747 | 0.544436 | 0.963651 | 0.927423 | 0.528245 | 0.084922 | 0.161450 | 0.111300 | 0.596452 | 294 | 1527 | 3168 | 40482 |
| 9 | acoustic_gop_phone_nb | phone_group_threshold | 0.169559 | 0.539829 | 0.982784 | 0.137320 | 0.162142 | 0.043584 | 0.942339 | 0.083315 | 0.564236 | 1716 | 105 | 37656 | 5994 |
| 10 | acoustic_gop_phone_nb | target_phone_threshold | 0.169559 | 0.539829 | 0.982784 | 0.137320 | 0.162142 | 0.043584 | 0.942339 | 0.083315 | 0.564236 | 1716 | 105 | 37656 | 5994 |
| 11 | feature_random_forest | target_phone_threshold | 0.903807 | 0.538641 | 0.963139 | 0.935601 | 0.527346 | 0.084066 | 0.141680 | 0.105521 | 0.645927 | 258 | 1563 | 2811 | 40839 |
| 12 | feature_logreg | phone_group_threshold | 0.867102 | 0.538468 | 0.963266 | 0.895716 | 0.513364 | 0.067595 | 0.181219 | 0.098463 | 0.623729 | 330 | 1491 | 4552 | 39098 |
| 13 | feature_logreg | target_phone_threshold | 0.875635 | 0.522389 | 0.961855 | 0.906392 | 0.507566 | 0.058091 | 0.138386 | 0.081831 | 0.623729 | 252 | 1569 | 4086 | 39564 |
| 14 | fusion_random_forest | fusion_global_threshold | 0.959952 | 0.500000 | 0.959952 | 1.000000 | 0.489784 | 0.000000 | 0.000000 | 0.000000 | 0.754886 | 0 | 1821 | 0 | 43650 |
| 15 | majority_class | majority_class | 0.959952 | 0.500000 | 0.959952 | 1.000000 | 0.489784 | 0.000000 | 0.000000 | 0.000000 | 0.500000 | 0 | 1821 | 0 | 43650 |

## Current Best

- Best test model by Balanced Accuracy: teacher_score_rule (teacher_score_threshold), Balanced Accuracy=1.000000, Macro-F1=1.000000, AUC=1.000000.
- Majority-class accuracy is high because the dataset is imbalanced; Balanced Accuracy and Macro-F1 are the primary comparison metrics.

## Teacher-Score Note

- `teacher_score_rule` uses the source phone-level score that the project labels are derived from. It is a label-reconstruction upper bound for acceptance comparison, not a deployable acoustic pronunciation model.
- For deployable model selection, compare models below `teacher_score_rule`; the current best deployable model remains `ssl_phone_logreg`.
