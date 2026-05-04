# Claim 04: Posterior Correction Changes Action

Paper location:
- Section 4.2
- Figure 6
- Table 14
- Table 15

Evidence:
- `paper_artifacts/shared_results/auditability_artifacts`
- `paper_artifacts/shared_results/ablation_results`
- `paper_artifacts/tables/table_14_posterior_prefix_injection/`
- `paper_artifacts/tables/table_15_auditability_cases/`

Main source:
- `artifacts/results/ablation/main/a2d_correct_prefix`
- `artifacts/results/ablation/main/a2d_adversarial_prefix`
- `artifacts/results/auditability/main/posterior_correction_cases.csv`

What this proves:
- When the posterior is corrected or adversarially replaced, the selected action or bid changes for a measurable subset of evaluated turns.

Limit:
- This is an intervention-style diagnostic, not a full causal proof over all dialogues.

Manifest IDs:
- `shared_ablation_results`
- `shared_auditability_artifacts`
- `table_14_posterior_prefix_injection`
- `table_15_auditability_cases`
