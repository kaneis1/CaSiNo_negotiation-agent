# Experiment Scripts

Manifest IDs are authoritative; this file groups experiment orchestration scripts by paper location.

| Paper location | Script | Main output | Cost | Manifest IDs |
|---|---|---|---|---|
| Section 4.1/4.5 | `sft_8b/scripts/run_bayesian_protocol3.lsf` | `opponent_model/results/turn_eval_bayesian_lambda1.0_m5_f0.50_full150` | Expensive/GPU | `shared_bayesian_teacher_full150` |
| Section 4.1/4.5 | `sft_8b/scripts/run_day9_student_eval.lsf` | `opponent_model/results/turn_eval_student_balanced_full150` | Expensive/GPU | `shared_distilled_student_balanced_full150` |
| Section 4.5 | `structured_cot/scripts/run_p3_baseline_70b.lsf` | `opponent_model/results/turn_eval_structured_cot_p3_live_70b_m1_150` | Expensive/GPU | `shared_structured_cot_70b_full150` |
| Section 4.1/Appendix V | `structured_cot/scripts/run_structured_opp_inference_70b_pipeline.lsf` | `structured_cot/results/structured_opp_inference_70b_pipeline_table12_full150` | Expensive/GPU | `shared_structured_cot_70b_elicited_posterior` |
| Section 4.3/Appendix S/T | `sft_8b/scripts/submit_ablation_study.sh` | `opponent_model/results/ablation_neurips2026` | Expensive/GPU | `shared_ablation_results` |
| Section 4.4/Appendix W | `sft_8b/scripts/submit_dnd_transfer.sh` | `opponent_model/results/dnd_transfer` | Expensive/GPU | `shared_dnd_transfer_results` |
