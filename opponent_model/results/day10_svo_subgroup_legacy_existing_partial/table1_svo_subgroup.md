# Day 10 SVO Subgroup Table

Matched records: `opponent_model/results/turn_eval_bayesian_svo_lambda_m5_f0.50_full150/turn_records.jsonl`
Mismatched records: `not provided`
Data: `data/casino_test.json`

First-offer integrativeness = first formal/predicted offer reaches the dialogue's max feasible joint points.
Match-vs-mismatch p-values are Welch tests on accept-decision correctness indicators.

| Row | Human self | Agent self | Human joint | Agent joint | Human first-offer int. | Agent first-offer int. | Match accept-F1 | Mismatch accept-F1 | Match - mismatch | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| proself | 18.417 (n=72) | 17.315 (n=464) | 36.653 (n=72) | 38.638 (n=464) | 0.462 (n=39) | 0.750 (n=72) | 0.880 (n=42) | NA (n=0) | NA | NA |
| prosocial | 17.853 (n=75) | 0.000 (n=484) | 36.347 (n=75) | 36.000 (n=484) | 0.429 (n=35) | 0.267 (n=75) | 0.937 (n=42) | NA (n=0) | NA | NA |
| unclassified | 17.333 (n=3) | 12.857 (n=21) | 36.333 (n=3) | 38.857 (n=21) | NA (n=0) | 0.667 (n=3) | 1.000 (n=3) | NA (n=0) | NA | NA |
| classified_all | 18.129 (n=147) | 8.475 (n=948) | 36.497 (n=147) | 37.291 (n=948) | 0.446 (n=74) | 0.503 (n=147) | 0.909 (n=84) | NA (n=0) | NA | NA |
| overall | 18.113 (n=150) | 8.570 (n=969) | 36.493 (n=150) | 37.325 (n=969) | 0.446 (n=74) | 0.507 (n=150) | 0.912 (n=87) | NA (n=0) | NA | NA |
