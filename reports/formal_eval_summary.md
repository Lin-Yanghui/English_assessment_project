# Formal Test Evaluation

The table below reports test-split performance for the current calibrated models.

| model | calibration | n | accuracy | balanced_accuracy | macro_f1 | auc | tn | fp | fn | tp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| feature_random_forest | phone_group_threshold | 45471 | 0.877988 | 0.557820 | 0.527742 | 0.645927 | 382 | 1439 | 4109 | 39541 |
| feature_logreg | phone_group_threshold | 45471 | 0.867102 | 0.538468 | 0.513364 | 0.623729 | 330 | 1491 | 4552 | 39098 |
