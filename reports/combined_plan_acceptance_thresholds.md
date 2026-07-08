# Plan Acceptance Threshold Calibration

Minimum line: error precision >= 0.40, error recall >= 0.50.

| scope | model | calibration | n | threshold | error_precision | error_recall | balanced_accuracy | macro_f1 | pass |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| core_fricatives | plan_fusion_random_forest | phone_group_threshold_plan_threshold | 8957 | 0.720000 | 0.213368 | 0.475191 | 0.683166 | 0.610481 | 0 |
| core_fricatives | plan_fusion_logreg | phone_group_threshold_plan_threshold | 8957 | 0.410000 | 0.196062 | 0.456107 | 0.669948 | 0.598007 | 0 |
| core_fricatives | feature_random_forest | phone_group_threshold_plan_threshold | 8957 | 0.350987 | 0.146774 | 0.347328 | 0.610934 | 0.559831 | 0 |
| core_fricatives | feature_random_forest | score_threshold_plan_threshold | 8957 | 0.350987 | 0.146774 | 0.347328 | 0.610934 | 0.559831 | 0 |
| core_fricatives | feature_random_forest | target_phone_threshold_plan_threshold | 8957 | 0.350987 | 0.146774 | 0.347328 | 0.610934 | 0.559831 | 0 |
| core_stops | plan_fusion_random_forest | phone_group_threshold_plan_threshold | 9512 | 0.778435 | 0.104730 | 0.336957 | 0.625440 | 0.552517 | 0 |
| core_fricatives | feature_logreg | phone_group_threshold_plan_threshold | 8957 | 0.251630 | 0.161728 | 0.250000 | 0.584741 | 0.565873 | 0 |
| core_fricatives | feature_logreg | score_threshold_plan_threshold | 8957 | 0.251630 | 0.161728 | 0.250000 | 0.584741 | 0.565873 | 0 |
| core_fricatives | feature_logreg | target_phone_threshold_plan_threshold | 8957 | 0.251630 | 0.161728 | 0.250000 | 0.584741 | 0.565873 | 0 |
| core_liquids | ssl_phone_logreg | score_threshold_plan_threshold | 3407 | 0.190000 | 0.288889 | 0.220339 | 0.589987 | 0.600617 | 0 |
| overall | plan_fusion_logreg | phone_group_threshold_plan_threshold | 50488 | 0.040000 | 0.337149 | 0.128878 | 0.557800 | 0.578740 | 0 |
| core_nasals | plan_fusion_logreg | phone_group_threshold_plan_threshold | 5443 | 0.069861 | 0.174603 | 0.100000 | 0.540044 | 0.549257 | 0 |
| overall | feature_random_forest | phone_group_threshold_plan_threshold | 50488 | 0.234948 | 0.234893 | 0.095863 | 0.539750 | 0.552387 | 0 |
| overall | feature_random_forest | score_threshold_plan_threshold | 50488 | 0.234948 | 0.234893 | 0.095863 | 0.539750 | 0.552387 | 0 |
| overall | feature_random_forest | target_phone_threshold_plan_threshold | 50488 | 0.234948 | 0.234893 | 0.095863 | 0.539750 | 0.552387 | 0 |
| core_stops | plan_fusion_logreg | phone_group_threshold_plan_threshold | 9512 | 0.020000 | 0.328358 | 0.079710 | 0.537419 | 0.556137 | 0 |
| core_liquids | plan_fusion_logreg | phone_group_threshold_plan_threshold | 3407 | 0.020000 | 0.352941 | 0.076271 | 0.532932 | 0.543490 | 0 |
| core_nasals | ssl_phone_logreg | score_threshold_plan_threshold | 5443 | 0.090000 | 0.205128 | 0.072727 | 0.530428 | 0.541130 | 0 |
| overall | feature_logreg | phone_group_threshold_plan_threshold | 50488 | 0.200000 | 0.323944 | 0.064041 | 0.528519 | 0.539746 | 0 |
| overall | feature_logreg | score_threshold_plan_threshold | 50488 | 0.200000 | 0.323944 | 0.064041 | 0.528519 | 0.539746 | 0 |
| overall | feature_logreg | target_phone_threshold_plan_threshold | 50488 | 0.200000 | 0.323944 | 0.064041 | 0.528519 | 0.539746 | 0 |
| overall | plan_fusion_random_forest | phone_group_threshold_plan_threshold | 50488 | 0.460000 | 0.441011 | 0.062450 | 0.529151 | 0.541677 | 0 |
| core_vowels | plan_fusion_logreg | phone_group_threshold_plan_threshold | 20409 | 0.030887 | 0.270161 | 0.059397 | 0.525005 | 0.532947 | 0 |
| core_liquids | feature_logreg | phone_group_threshold_plan_threshold | 3407 | 0.240000 | 0.170732 | 0.059322 | 0.518939 | 0.521704 | 0 |
| core_liquids | feature_logreg | score_threshold_plan_threshold | 3407 | 0.240000 | 0.170732 | 0.059322 | 0.518939 | 0.521704 | 0 |
| core_liquids | feature_logreg | target_phone_threshold_plan_threshold | 3407 | 0.240000 | 0.170732 | 0.059322 | 0.518939 | 0.521704 | 0 |
| core_nasals | feature_logreg | phone_group_threshold_plan_threshold | 5443 | 0.391032 | 0.078788 | 0.059091 | 0.514994 | 0.516673 | 0 |
| core_nasals | feature_logreg | score_threshold_plan_threshold | 5443 | 0.391032 | 0.078788 | 0.059091 | 0.514994 | 0.516673 | 0 |
| core_nasals | feature_logreg | target_phone_threshold_plan_threshold | 5443 | 0.391032 | 0.078788 | 0.059091 | 0.514994 | 0.516673 | 0 |
| core_vowels | plan_fusion_random_forest | phone_group_threshold_plan_threshold | 20409 | 0.553653 | 0.154728 | 0.047872 | 0.516286 | 0.519161 | 0 |
| overall | ssl_phone_logreg | score_threshold_plan_threshold | 50488 | 0.060000 | 0.315217 | 0.046142 | 0.520444 | 0.526742 | 0 |
| core_stops | ssl_phone_logreg | score_threshold_plan_threshold | 9512 | 0.040000 | 0.291667 | 0.025362 | 0.511761 | 0.515696 | 0 |
| core_vowels | ssl_phone_logreg | score_threshold_plan_threshold | 20409 | 0.030000 | 0.619048 | 0.023050 | 0.511110 | 0.508123 | 0 |
| core_fricatives | ssl_phone_logreg | score_threshold_plan_threshold | 8957 | 0.030000 | 0.476190 | 0.019084 | 0.508890 | 0.503235 | 0 |
| core_stops | feature_random_forest | phone_group_threshold_plan_threshold | 9512 | 0.140000 | 0.250000 | 0.010870 | 0.504948 | 0.502891 | 0 |
| core_stops | feature_random_forest | score_threshold_plan_threshold | 9512 | 0.140000 | 0.250000 | 0.010870 | 0.504948 | 0.502891 | 0 |
| core_stops | feature_random_forest | target_phone_threshold_plan_threshold | 9512 | 0.140000 | 0.250000 | 0.010870 | 0.504948 | 0.502891 | 0 |
| core_liquids | feature_random_forest | phone_group_threshold_plan_threshold | 3407 | 0.100000 | 1.000000 | 0.008475 | 0.504237 | 0.490611 | 0 |
| core_liquids | feature_random_forest | score_threshold_plan_threshold | 3407 | 0.100000 | 1.000000 | 0.008475 | 0.504237 | 0.490611 | 0 |
| core_liquids | feature_random_forest | target_phone_threshold_plan_threshold | 3407 | 0.100000 | 1.000000 | 0.008475 | 0.504237 | 0.490611 | 0 |
| core_liquids | plan_fusion_random_forest | phone_group_threshold_plan_threshold | 3407 | 0.346195 | 0.333333 | 0.004237 | 0.501803 | 0.486161 | 0 |
| core_vowels | feature_random_forest | phone_group_threshold_plan_threshold | 20409 | 0.120000 | 0.250000 | 0.001773 | 0.500731 | 0.487497 | 0 |
| core_vowels | feature_random_forest | score_threshold_plan_threshold | 20409 | 0.120000 | 0.250000 | 0.001773 | 0.500731 | 0.487497 | 0 |
| core_vowels | feature_random_forest | target_phone_threshold_plan_threshold | 20409 | 0.120000 | 0.250000 | 0.001773 | 0.500731 | 0.487497 | 0 |
| core_vowels | feature_logreg | phone_group_threshold_plan_threshold | 20409 | 0.100000 | 0.000000 | 0.000000 | 0.500000 | 0.485790 | 0 |
| core_vowels | feature_logreg | score_threshold_plan_threshold | 20409 | 0.100000 | 0.000000 | 0.000000 | 0.500000 | 0.485790 | 0 |
| core_vowels | feature_logreg | target_phone_threshold_plan_threshold | 20409 | 0.100000 | 0.000000 | 0.000000 | 0.500000 | 0.485790 | 0 |
| core_nasals | feature_random_forest | phone_group_threshold_plan_threshold | 5443 | 0.270000 | 0.000000 | 0.000000 | 0.499617 | 0.489495 | 0 |
| core_nasals | feature_random_forest | score_threshold_plan_threshold | 5443 | 0.270000 | 0.000000 | 0.000000 | 0.499617 | 0.489495 | 0 |
| core_nasals | feature_random_forest | target_phone_threshold_plan_threshold | 5443 | 0.270000 | 0.000000 | 0.000000 | 0.499617 | 0.489495 | 0 |
| core_stops | feature_logreg | phone_group_threshold_plan_threshold | 9512 | 0.230000 | 0.000000 | 0.000000 | 0.499459 | 0.492368 | 0 |
| core_stops | feature_logreg | score_threshold_plan_threshold | 9512 | 0.230000 | 0.000000 | 0.000000 | 0.499459 | 0.492368 | 0 |
| core_stops | feature_logreg | target_phone_threshold_plan_threshold | 9512 | 0.230000 | 0.000000 | 0.000000 | 0.499459 | 0.492368 | 0 |
| core_nasals | plan_fusion_random_forest | phone_group_threshold_plan_threshold | 5443 | 0.470000 | 0.000000 | 0.000000 | 0.499426 | 0.489400 | 0 |

## Summary

- No deployable operating point meets the minimum line on the current combined test set.
- Thresholds are selected on dev and reported on test; no source labels or teacher scores are used.
