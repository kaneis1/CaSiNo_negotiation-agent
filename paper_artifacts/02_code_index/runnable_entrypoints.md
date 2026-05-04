# Runnable Entrypoints

Manifest IDs are authoritative; this file is a navigation summary.

| Goal | Paper item | Command | Cost | Produces manifest IDs |
|---|---|---|---|---|
| Validate overlay | Repository hygiene | `python scripts/check_paper_artifacts_manifest.py` | Cheap/local | all overlay entries |
| Run unit tests | Appendix D/Table 5 | `/sc/arion/work/cuiz02/conda-envs/envs/casino/bin/python -m unittest discover -s tests -v` | Cheap/local | `table_05_structured_output_validity` |
| Make Brier trajectory artifacts | Figure 2/Table 6 | `python -m casino_belief.diagnostics.posterior_quality.make_headline_artifacts` | Cheap/local if source summaries exist | `figure_02_brier_trajectory`, `table_06_brier_by_turn` |
| Make auditability artifacts | Table 1/Figure 6/Table 15 | `python -m casino_belief.diagnostics.auditability.make_auditability_artifacts` | Cheap/local if source records exist | `table_01_belief_policy_decomposition`, `figure_06_posterior_trajectories`, `table_15_auditability_cases` |
| Aggregate ablations | Table 2/Table 13/Figure 5 | `bsub < experiments/lsf/ablation/run_ablation_aggregate.lsf` | Expensive/GPU provenance | `table_02_ablation_results`, `table_13_likelihood_sweep`, `figure_05_likelihood_sensitivity_heatmap` |
| Aggregate DND transfer | Table 9/Table 17 | `bsub < experiments/lsf/dnd_transfer/run_dnd_aggregate.lsf` | Expensive/GPU provenance | `table_09_dnd_transfer_headline`, `table_17_dnd_transfer_full` |
