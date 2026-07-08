# Formal Test Evaluation

The table below reports test-split performance for the current calibrated models.

| model | calibration | n | accuracy | balanced_accuracy | precision | recall | macro_f1 | error_precision | error_recall | error_f1 | auc | tn | fp | fn | tp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| feature_random_forest | phone_group_threshold | 45471 | 0.877988 | 0.557820 | 0.964885 | 0.905865 | 0.527742 | 0.085059 | 0.209775 | 0.121039 | 0.645927 | 382 | 1439 | 4109 | 39541 |
| feature_random_forest | target_phone_threshold | 45471 | 0.903807 | 0.538641 | 0.963139 | 0.935601 | 0.527346 | 0.084066 | 0.141680 | 0.105521 | 0.645927 | 258 | 1563 | 2811 | 40839 |
| feature_logreg | phone_group_threshold | 45471 | 0.867102 | 0.538468 | 0.963266 | 0.895716 | 0.513364 | 0.067595 | 0.181219 | 0.098463 | 0.623729 | 330 | 1491 | 4552 | 39098 |
| feature_logreg | target_phone_threshold | 45471 | 0.875635 | 0.522389 | 0.961855 | 0.906392 | 0.507566 | 0.058091 | 0.138386 | 0.081831 | 0.623729 | 252 | 1569 | 4086 | 39564 |
