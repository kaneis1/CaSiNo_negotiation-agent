# Experiment Scripts

Manifest IDs are authoritative; this file groups experiment orchestration scripts by paper location.

| Paper location | Script | Main output | Cost | Manifest IDs |
|---|---|---|---|---|
| Section 4.1/4.5 | `experiments/lsf/evaluation/run_bayesian_protocol3.lsf` | `artifacts/results/protocol3/bayesian_teacher_full150` | Expensive/GPU | `shared_bayesian_teacher_full150` |
| Section 4.1/4.5 | `experiments/lsf/evaluation/run_day9_student_eval.lsf` | `artifacts/results/protocol3/distilled_student_balanced_full150` | Expensive/GPU | `shared_distilled_student_balanced_full150` |
| Section 4.5 | `experiments/lsf/structured_cot/run_p3_baseline_70b.lsf` | `artifacts/results/protocol3/structured_cot_70b_full150` | Expensive/GPU | `shared_structured_cot_70b_full150` |
| Section 4.1/Appendix V | `experiments/lsf/structured_cot/run_structured_opp_inference_70b_pipeline.lsf` | `artifacts/results/posterior_quality/structured_cot_elicited_posterior` | Expensive/GPU | `shared_structured_cot_70b_elicited_posterior` |
| Section 4.3/Appendix S/T | `experiments/lsf/ablation/submit_ablation_study.sh` | `artifacts/results/ablation/main` | Expensive/GPU | `shared_ablation_results` |
| Section 4.4/Appendix W | `experiments/lsf/dnd_transfer/submit_dnd_transfer.sh` | `artifacts/results/dnd_transfer/main` | Expensive/GPU | `shared_dnd_transfer_results` |
