# Error Analysis

Model: feature_random_forest + phone_group_threshold on test split.

## Case Counts

- True positives: 39541
- True negatives: 382
- False positives: 4109
- False negatives: 1439

## Weakest Phone Groups

| phone_group | n | gold_errors | false_positives | false_negatives | gold_error_rate |
|---|---:|---:|---:|---:|---:|
| vowel | 18462 | 939 | 2943 | 637 | 0.0509 |
| fricative | 7999 | 309 | 71 | 298 | 0.0386 |
| nasal | 4926 | 177 | 289 | 158 | 0.0359 |
| stop | 8626 | 200 | 743 | 154 | 0.0232 |
| liquid | 3011 | 152 | 51 | 149 | 0.0505 |
| glide | 2124 | 34 | 12 | 33 | 0.0160 |
| affricate | 323 | 10 | 0 | 10 | 0.0310 |

## Weakest Target Phones

| target_phone | n | gold_errors | false_positives | false_negatives | gold_error_rate |
|---|---:|---:|---:|---:|---:|
| AH | 4329 | 228 | 874 | 152 | 0.0527 |
| N | 2929 | 100 | 190 | 90 | 0.0341 |
| L | 1836 | 90 | 7 | 89 | 0.0490 |
| S | 2100 | 77 | 0 | 77 | 0.0367 |
| IH | 2889 | 112 | 434 | 72 | 0.0388 |
| T | 3623 | 79 | 337 | 65 | 0.0218 |
| DH | 1368 | 62 | 1 | 62 | 0.0453 |
| R | 1175 | 62 | 44 | 60 | 0.0528 |
| IY | 1830 | 61 | 148 | 49 | 0.0333 |
| Z | 1265 | 52 | 34 | 48 | 0.0411 |
| M | 1441 | 49 | 27 | 46 | 0.0340 |
| EH | 1248 | 95 | 327 | 45 | 0.0761 |
| AY | 1108 | 65 | 109 | 43 | 0.0587 |
| AE | 1414 | 50 | 147 | 42 | 0.0354 |
| EY | 791 | 88 | 256 | 41 | 0.1113 |
