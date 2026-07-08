# SSL vs GOP Comparison

This report compares proxy GOP threshold baselines against SSL phone-segment classifiers when SSL metrics are available.

| rank | family | model | calibration | accuracy | balanced_accuracy | macro_f1 | error_f1 | auc |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | ssl_phone_classifier | ssl_phone_logreg | global_threshold | 0.910690 | 0.606427 | 0.575467 | 0.198223 | 0.738200 |
| 2 | proxy_gop | feature_random_forest | phone_group_threshold | 0.877988 | 0.557820 | 0.527742 | 0.121039 | 0.645927 |
| 3 | acoustic_gop | acoustic_gop_phone_nb | phone_group_threshold | 0.169559 | 0.539829 | 0.162142 | 0.083315 | 0.564236 |
| 4 | acoustic_gop | acoustic_gop_phone_nb | target_phone_threshold | 0.169559 | 0.539829 | 0.162142 | 0.083315 | 0.564236 |
| 5 | proxy_gop | feature_random_forest | target_phone_threshold | 0.903807 | 0.538641 | 0.527346 | 0.105521 | 0.645927 |
| 6 | proxy_gop | feature_logreg | phone_group_threshold | 0.867102 | 0.538468 | 0.513364 | 0.098463 | 0.623729 |
| 7 | proxy_gop | feature_logreg | target_phone_threshold | 0.875635 | 0.522389 | 0.507566 | 0.081831 | 0.623729 |
