# Claim 02: Posterior Quality Over Turns

Paper location:
- Section 4.1
- Figure 2
- Appendix E / Table 6

Evidence:
- `paper_artifacts/figures/figure_02_brier_trajectory.png`
- `paper_artifacts/tables/table_06_brier_by_turn/`
- `paper_artifacts/shared_results/bayesian_teacher_full150`
- `paper_artifacts/shared_results/distilled_student_balanced_full150`

Main source:
- `opponent_model/results/day9_headline_artifacts`

What this proves:
- The Brier trajectory can be audited turn by turn against the teacher, student, and uniform reference.

Limit:
- Supported-turn statements use the paper's support threshold and should not be generalized to sparse late turns.

Manifest IDs:
- `figure_02_brier_trajectory`
- `table_06_brier_by_turn`
- `shared_bayesian_teacher_full150`
- `shared_distilled_student_balanced_full150`
