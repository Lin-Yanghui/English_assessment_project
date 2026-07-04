# Model Comparison

Sorted by test Balanced Accuracy, then Macro-F1 and AUC.

| rank | model | calibration | accuracy | balanced_accuracy | macro_f1 | auc | tn | fp | fn | tp |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | feature_random_forest | global_threshold | 0.513118 | 0.604845 | 0.384813 | 0.645927 | 1283 | 538 | 21601 | 22049 |
| 2 | feature_logreg | global_threshold | 0.643839 | 0.582945 | 0.440925 | 0.623729 | 941 | 880 | 15315 | 28335 |
| 3 | feature_random_forest | phone_group_threshold | 0.877988 | 0.557820 | 0.527742 | 0.645927 | 382 | 1439 | 4109 | 39541 |
| 4 | feature_logreg | phone_group_threshold | 0.867102 | 0.538468 | 0.513364 | 0.623729 | 330 | 1491 | 4552 | 39098 |
| 5 | majority_class | majority_class | 0.959952 | 0.500000 | 0.489784 | 0.500000 | 0 | 1821 | 0 | 43650 |

## Current Best

- Best test model by Balanced Accuracy: feature_random_forest (global_threshold), Balanced Accuracy=0.604845, Macro-F1=0.384813, AUC=0.645927.
- Majority-class accuracy is high because the dataset is imbalanced; Balanced Accuracy and Macro-F1 are the primary comparison metrics.
