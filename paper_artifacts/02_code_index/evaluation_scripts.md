# Evaluation Scripts

Manifest IDs are authoritative; this file indexes evaluation and diagnostic code.

| Paper location | Script/module | Main output | Cost | Manifest IDs |
|---|---|---|---|---|
| Section 3.6/4.5 | `src/casino_belief/evaluation/turn_eval_run.py` | `artifacts/results/protocol3/*` | Expensive/GPU for live models | `table_03_main_protocol3_metrics` |
| Section 4.1 | `src/casino_belief/diagnostics/posterior_quality/make_headline_artifacts.py` | `artifacts/results/posterior_quality` | Cheap/local | `figure_02_brier_trajectory`, `table_06_brier_by_turn` |
| Section 4.2 | `src/casino_belief/diagnostics/auditability/make_auditability_artifacts.py` | `artifacts/results/auditability/main` | Cheap/local | `table_01_belief_policy_decomposition`, `figure_06_posterior_trajectories`, `table_15_auditability_cases` |
| Section 4.3 | `src/casino_belief/diagnostics/ablation/ablation_aggregate.py` | `artifacts/results/ablation/main/aggregate` | Cheap/local after runs exist | `table_02_ablation_results`, `table_13_likelihood_sweep` |
| Section 4.4 | `src/casino_belief/transfer/dnd/dnd_aggregate.py` | `artifacts/results/dnd_transfer/main/aggregate` | Cheap/local after runs exist | `table_09_dnd_transfer_headline`, `table_17_dnd_transfer_full` |
| Section 4.6 | `src/casino_belief/diagnostics/svo/diagnose_svo_accept_match_mismatch.py` | `artifacts/results/svo/accept_diagnostic_*` | Cheap/local | `table_10_svo_match_mismatch_accept` |
