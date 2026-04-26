# Day 10 SVO Subgroup Table

Matched records: `opponent_model/results/turn_eval_bayesian_svo_lambda_rescaled_rerun_m5_f0.50_full150/turn_records.jsonl`
Mismatched records: `opponent_model/results/turn_eval_bayesian_svo_lambda_rescaled_mismatch_m5_f0.50_full150/turn_records.jsonl`
Data: `data/casino_test.json`

First-offer integrativeness = first formal/predicted offer reaches the dialogue's max feasible joint points.
Match-vs-mismatch p-values are Welch tests on accept-decision correctness indicators.

| Row | Human self | Agent self | Human joint | Agent joint | Human first-offer int. | Agent first-offer int. | Match accept-F1 | Mismatch accept-F1 | Match - mismatch | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| proself | 18.417 (n=72) | 36.000 (n=479) | 36.653 (n=72) | 36.000 (n=479) | 0.462 (n=39) | 0.319 (n=72) | 0.800 (n=42) | 0.800 (n=42) | 0.000 | 1.000 |
| prosocial | 17.853 (n=75) | 34.404 (n=502) | 36.347 (n=75) | 37.010 (n=502) | 0.429 (n=35) | 0.307 (n=75) | 0.754 (n=42) | 0.754 (n=42) | 0.000 | 1.000 |
| unclassified | 17.333 (n=3) | 36.000 (n=23) | 36.333 (n=3) | 36.000 (n=23) | NA (n=0) | 0.000 (n=3) | 0.500 (n=3) | 0.500 (n=3) | 0.000 | 1.000 |
| classified_all | 18.129 (n=147) | 35.183 (n=981) | 36.497 (n=147) | 36.517 (n=981) | 0.446 (n=74) | 0.313 (n=147) | 0.777 (n=84) | 0.777 (n=84) | 0.000 | 1.000 |
| overall | 18.113 (n=150) | 35.202 (n=1004) | 36.493 (n=150) | 36.505 (n=1004) | 0.446 (n=74) | 0.307 (n=150) | 0.768 (n=87) | 0.768 (n=87) | 0.000 | 1.000 |
