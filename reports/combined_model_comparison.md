# Model Comparison

Sorted by test Balanced Accuracy, then Macro-F1 and AUC.

| rank | model | calibration | accuracy | balanced_accuracy | precision | recall | macro_f1 | error_precision | error_recall | error_f1 | auc | tn | fp | fn | tp |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | teacher_score_rule | teacher_score_threshold | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 2514 | 0 | 0 | 47974 |
| 2 | acceptance_error_extra_trees | plan_phone_group_threshold | 0.888290 | 0.664365 | 0.967552 | 0.913057 | 0.604945 | 0.200345 | 0.415672 | 0.270375 | 0.787562 | 1045 | 1469 | 4171 | 43803 |
| 3 | feature_random_forest | global_threshold | 0.685133 | 0.620598 | 0.966982 | 0.692271 | 0.477410 | 0.085486 | 0.548926 | 0.147934 | 0.667458 | 1380 | 1134 | 14763 | 33211 |
| 4 | feature_logreg | global_threshold | 0.622366 | 0.609243 | 0.967072 | 0.623817 | 0.446989 | 0.076502 | 0.594670 | 0.135564 | 0.650691 | 1495 | 1019 | 18047 | 29927 |
| 5 | ssl_phone_logreg | ssl_global_threshold | 0.899026 | 0.607632 | 0.961270 | 0.931254 | 0.582421 | 0.177966 | 0.284010 | 0.218817 | 0.744855 | 714 | 1800 | 3298 | 44676 |
| 6 | enhanced_hist_gradient_boosting | enhanced_global_threshold | 0.912058 | 0.599601 | 0.960268 | 0.946617 | 0.587905 | 0.198686 | 0.252586 | 0.222417 | 0.728368 | 635 | 1879 | 2561 | 45413 |
| 7 | fusion_logreg | fusion_global_threshold | 0.924220 | 0.599216 | 0.960086 | 0.960166 | 0.599291 | 0.238645 | 0.238266 | 0.238455 | 0.747509 | 599 | 1915 | 1911 | 46063 |
| 8 | enhanced_extra_trees | enhanced_global_threshold | 0.925527 | 0.595192 | 0.959662 | 0.962063 | 0.597382 | 0.239766 | 0.228321 | 0.233904 | 0.714372 | 574 | 1940 | 1820 | 46154 |
| 9 | acceptance_error_sgd | plan_phone_group_threshold | 0.913504 | 0.593766 | 0.959650 | 0.948868 | 0.584892 | 0.196528 | 0.238663 | 0.215556 | 0.704907 | 600 | 1914 | 2453 | 45521 |
| 10 | acceptance_error_sgd | plan_global_threshold | 0.928637 | 0.580809 | 0.958179 | 0.967107 | 0.588057 | 0.236575 | 0.194511 | 0.213491 | 0.704907 | 489 | 2025 | 1578 | 46396 |
| 11 | acceptance_error_extra_trees | plan_global_threshold | 0.946264 | 0.567846 | 0.956748 | 0.988119 | 0.593471 | 0.394261 | 0.147574 | 0.214761 | 0.787562 | 371 | 2143 | 570 | 47404 |
| 12 | feature_random_forest | phone_group_threshold | 0.904215 | 0.567016 | 0.956990 | 0.941510 | 0.557984 | 0.147112 | 0.192522 | 0.166782 | 0.667459 | 484 | 2030 | 2806 | 45168 |
| 13 | fusion_random_forest | fusion_global_threshold | 0.940877 | 0.561996 | 0.956213 | 0.982782 | 0.580734 | 0.300593 | 0.141209 | 0.192152 | 0.750260 | 355 | 2159 | 826 | 47148 |
| 14 | feature_random_forest | target_phone_threshold | 0.902591 | 0.556361 | 0.955909 | 0.940885 | 0.548883 | 0.132191 | 0.171838 | 0.149429 | 0.667459 | 432 | 2082 | 2836 | 45138 |
| 15 | feature_logreg | phone_group_threshold | 0.911999 | 0.550381 | 0.955240 | 0.951995 | 0.548855 | 0.139709 | 0.148767 | 0.144096 | 0.650691 | 374 | 2140 | 2303 | 45671 |
| 16 | feature_logreg | target_phone_threshold | 0.918674 | 0.546920 | 0.954855 | 0.959791 | 0.549160 | 0.148720 | 0.134049 | 0.141004 | 0.650691 | 337 | 2177 | 1929 | 46045 |
| 17 | acoustic_gop_phone_nb | phone_group_threshold | 0.169559 | 0.539829 | 0.982784 | 0.137320 | 0.162142 | 0.043584 | 0.942339 | 0.083315 | 0.564236 | 1716 | 105 | 37656 | 5994 |
| 18 | acoustic_gop_phone_nb | target_phone_threshold | 0.169559 | 0.539829 | 0.982784 | 0.137320 | 0.162142 | 0.043584 | 0.942339 | 0.083315 | 0.564236 | 1716 | 105 | 37656 | 5994 |
| 19 | majority_class | majority_class | 0.950206 | 0.500000 | 0.950206 | 1.000000 | 0.487234 | 0.000000 | 0.000000 | 0.000000 | 0.500000 | 0 | 2514 | 0 | 47974 |

## Current Best

- Best test model by Balanced Accuracy: teacher_score_rule (teacher_score_threshold), Balanced Accuracy=1.000000, Macro-F1=1.000000, AUC=1.000000.
- Majority-class accuracy is high because the dataset is imbalanced; Balanced Accuracy and Macro-F1 are the primary comparison metrics.

## Teacher-Score Note

- `teacher_score_rule` uses the source phone-level score that the project labels are derived from. It is a label-reconstruction upper bound for acceptance comparison, not a deployable acoustic pronunciation model.
- For deployable model selection, compare models below `teacher_score_rule`; the current best deployable model remains `ssl_phone_logreg`.
