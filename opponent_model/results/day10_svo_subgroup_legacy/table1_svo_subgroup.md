# Day 10 SVO Subgroup Table

Matched records: `opponent_model/results/turn_eval_bayesian_svo_lambda_legacy_m5_f0.50_full150/turn_records.jsonl`
Mismatched records: `opponent_model/results/turn_eval_bayesian_svo_lambda_legacy_mismatch_m5_f0.50_full150/turn_records.jsonl`
Data: `data/casino_test.json`

First-offer integrativeness = first formal/predicted offer reaches the dialogue's max feasible joint points.
Match-vs-mismatch p-values are Welch tests on accept-decision correctness indicators.

| Row | Human self | Agent self | Human joint | Agent joint | Human first-offer int. | Agent first-offer int. | Match accept-F1 | Mismatch accept-F1 | Match - mismatch | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| proself | 18.417 (n=72) | 17.504 (n=466) | 36.653 (n=72) | 38.639 (n=466) | 0.462 (n=39) | 0.750 (n=72) | 0.877 (n=42) | 0.909 (n=42) | -0.032 | 0.584 |
| prosocial | 17.853 (n=75) | 0.000 (n=484) | 36.347 (n=75) | 36.000 (n=484) | 0.429 (n=35) | 0.267 (n=75) | 0.937 (n=42) | 0.892 (n=42) | 0.045 | 0.372 |
| unclassified | 17.333 (n=3) | 12.857 (n=21) | 36.333 (n=3) | 38.857 (n=21) | NA (n=0) | 0.667 (n=3) | 1.000 (n=3) | 1.000 (n=3) | 0.000 | NA |
| classified_all | 18.129 (n=147) | 8.586 (n=950) | 36.497 (n=147) | 37.295 (n=950) | 0.446 (n=74) | 0.503 (n=147) | 0.908 (n=84) | 0.901 (n=84) | 0.007 | 0.839 |
| overall | 18.113 (n=150) | 8.679 (n=971) | 36.493 (n=150) | 37.329 (n=971) | 0.446 (n=74) | 0.507 (n=150) | 0.911 (n=87) | 0.904 (n=87) | 0.007 | 0.840 |
