# Evaluation Scripts

Manifest IDs are authoritative; this file indexes evaluation and diagnostic code.

| Paper location | Script/module | Main output | Cost | Manifest IDs |
|---|---|---|---|---|
| Section 3.6/4.5 | `opponent_model/turn_eval_run.py` | `opponent_model/results/turn_eval_*` | Expensive/GPU for live models | `table_03_main_protocol3_metrics` |
| Section 4.1 | `opponent_model/scripts/make_day9_headline_artifacts.py` | `opponent_model/results/day9_headline_artifacts` | Cheap/local | `figure_02_brier_trajectory`, `table_06_brier_by_turn` |
| Section 4.2 | `opponent_model/scripts/make_auditability_artifacts.py` | `opponent_model/results/day11_auditability` | Cheap/local | `table_01_belief_policy_decomposition`, `figure_06_posterior_trajectories`, `table_15_auditability_cases` |
| Section 4.3 | `sft_8b/ablation_aggregate.py` | `opponent_model/results/ablation_neurips2026/aggregate` | Cheap/local after runs exist | `table_02_ablation_results`, `table_13_likelihood_sweep` |
| Section 4.4 | `sft_8b/dnd_aggregate.py` | `opponent_model/results/dnd_transfer/aggregate` | Cheap/local after runs exist | `table_09_dnd_transfer_headline`, `table_17_dnd_transfer_full` |
| Section 4.6 | `opponent_model/scripts/diagnose_svo_accept_match_mismatch.py` | `opponent_model/results/day10_svo_accept_diagnostic_*` | Cheap/local | `table_10_svo_match_mismatch_accept` |
