# Claim 05: Overconfidence Diagnostics

Paper location:
- Section 4.2
- Appendix U

Evidence:
- `paper_artifacts/shared_results/auditability_artifacts`

Main source:
- `artifacts/results/auditability/main/entropy_confidence_diagnostic.csv`
- `artifacts/results/auditability/main/auditability_metrics_summary.json`

What this proves:
- Confidence, entropy, and Brier can be inspected alongside action alignment to find overconfident wrong-belief or wrong-policy cases.

Limit:
- This is a diagnostic slice, not a claim that confidence alone fully predicts decision quality.

Manifest IDs:
- `shared_auditability_artifacts`
- `table_01_belief_policy_decomposition`
