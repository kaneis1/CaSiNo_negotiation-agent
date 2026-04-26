# Day 10 Table 1: SVO x Integrative Potential

Records: `opponent_model/results/turn_eval_bayesian_svo_lambda_m5_f0.50_full150/turn_records.jsonl`
Data: `data/casino_test.json`

Welch p-values test agent-vs-human mean differences. They are not formal equivalence tests.

| SVO | IP tercile | Human self | Agent self | p | Human joint | Agent joint | p | Human accept-F1 | Agent accept-F1 | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| proself | low | 17.696 (n=23) | 13.510 (n=151) | 0.000 | 34.870 (n=23) | 36.000 (n=151) | 0.328 | 1.000 (n=14) | 0.833 (n=14) | 0.040 |
| proself | mid | 17.654 (n=26) | 17.418 (n=170) | 0.779 | 35.962 (n=26) | 38.365 (n=170) | 0.033 | 1.000 (n=16) | 0.968 (n=16) | 0.333 |
| proself | high | 20.000 (n=23) | 21.210 (n=143) | 0.140 | 39.217 (n=23) | 41.748 (n=143) | 0.000 | 1.000 (n=12) | 0.800 (n=12) | 0.039 |
| prosocial | low | 16.250 (n=20) | 0.000 (n=130) | 0.000 | 33.400 (n=20) | 36.000 (n=130) | 0.163 | 1.000 (n=10) | 0.824 (n=10) | 0.081 |
| prosocial | mid | 18.320 (n=25) | 0.000 (n=160) | 0.000 | 37.400 (n=25) | 36.000 (n=160) | 0.000 | 1.000 (n=12) | 0.909 (n=12) | 0.166 |
| prosocial | high | 18.533 (n=30) | 0.000 (n=194) | 0.000 | 37.433 (n=30) | 36.000 (n=194) | 0.323 | 1.000 (n=20) | 1.000 (n=20) | NA |
