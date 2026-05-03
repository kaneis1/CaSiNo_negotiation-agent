# Paper Section File Map

Crosswalk for `Distilling_Bayesian_Belief_States_into_Language_Models_for_Auditable_Negotiation.pdf`.

I scanned the repo file inventory, extracted the PDF text with Ghostscript, and matched the paper's main sections, tables, figures, and appendices to local code/data/result artifacts. Paths are relative to the repo root.

## High-level Anchors

| Paper claim/artifact | Primary files |
|---|---|
| Final 150-dialogue held-out split | `data/casino_test.json`, `data/casino_train.json`, `data/casino.json` |
| CaSiNo raw data and annotations | `CaSiNo/data/casino.json`, `CaSiNo/data/casino_ann.json`, `CaSiNo/README.md`, `CaSiNo/data/README.md` |
| Main Protocol 3 harness | `opponent_model/turn_eval_run.py`, `opponent_model/turn_level_metrics.py`, `opponent_model/turn_agents.py` |
| Bayesian teacher | `sft_8b/posterior.py`, `sft_8b/bayesian_agent.py`, `sft_8b/menu.py`, `sft_8b/prompts.py` |
| Distilled student | `sft_8b/student_prompts.py`, `sft_8b/student_parser.py`, `sft_8b/student_model.py`, `sft_8b/build_day8_sft_data.py`, `sft_8b/train.py` |
| Structured-CoT baseline | `structured_cot/prompts.py`, `structured_cot/agent.py`, `structured_cot/live_turn_agent.py`, `structured_cot/run_protocol1.py`, `structured_cot/run_protocol3_baseline.py` |
| Main results | `opponent_model/results/turn_eval_bayesian_lambda1.0_m5_f0.50_full150/`, `opponent_model/results/turn_eval_student_balanced_full150/`, `opponent_model/results/turn_eval_structured_cot_p3_live_70b_m1_150/`, `opponent_model/results/day9_headline_artifacts/` |
| Ablations | `sft_8b/ablation.py`, `sft_8b/ablation_student.py`, `sft_8b/ablation_aggregate.py`, `opponent_model/results/ablation_neurips2026/` |
| Auditability diagnostics | `opponent_model/scripts/make_auditability_artifacts.py`, `opponent_model/results/day11_auditability/` |
| DND transfer | `sft_8b/dnd_data.py`, `sft_8b/dnd_posterior.py`, `sft_8b/dnd_eval.py`, `sft_8b/dnd_menu.py`, `sft_8b/dnd_metrics.py`, `data/dnd/`, `opponent_model/results/dnd_transfer/` |

## Main Paper

| Section | What it covers | Corresponding files |
|---|---|---|
| Abstract | Headline Brier, student/teacher/baseline comparison, auditability, prefix interventions | `README.md`; `opponent_model/results/turn_eval_bayesian_lambda1.0_m5_f0.50_full150/turn_summary.json`; `opponent_model/results/turn_eval_student_balanced_full150/turn_summary.json`; `opponent_model/results/turn_eval_structured_cot_p3_live_70b_m1_150/turn_summary.json`; `structured_cot/results/structured_opp_inference_70b_pipeline_table12_full150/summary.json`; `opponent_model/results/day11_auditability/auditability_metrics_summary.json`; `opponent_model/results/ablation_neurips2026/aggregate/ablation_results.md` |
| 1 Introduction | Motivation, contributions, core numbers, auditability story | Same as Abstract, plus `structured_cot/baseline_weaknesses.md`, `structured_cot/protocol_alignment_audit.md`, `roadmap.md` |
| 2 Related Work | CaSiNo, prior opponent modeling, structured prompting, DND | `CaSiNo/README.md`; `CaSiNo/data/README.md`; `opponent_model_benchmark/README.md`; `opponent_model_benchmark/oppmodeling/`; `structured_cot/prompts.py`; `structured_cot/agent.py`; `data/dnd/raw/`; `sft_8b/dnd_data.py` |
| 3 Methods | Overall modular agent and distillation setup | `sft_8b/posterior.py`; `sft_8b/bayesian_agent.py`; `sft_8b/menu.py`; `sft_8b/build_distill_data.py`; `sft_8b/build_day8_sft_data.py`; `sft_8b/student_prompts.py`; `opponent_model/turn_eval_run.py`; `opponent_model/turn_level_metrics.py` |
| 3.1 Task and Hypothesis Space | CaSiNo six-ordering preference space | `CaSiNo/data/casino.json`; `data/casino.json`; `data/casino_train.json`; `data/casino_test.json`; `sft_8b/posterior.py` (`ORDERINGS`); `opponent_model/hypotheses.py`; `sft_8b/menu.py` |
| 3.2 Architecture | LLM likelihood, Bayesian posterior, menu, decision module, student | `sft_8b/posterior.py`; `sft_8b/bayesian_agent.py`; `sft_8b/menu.py`; `sft_8b/student_model.py`; `opponent_model/turn_agents.py`; `opponent_model/turn_eval_run.py` |
| 3.3 Bayesian Teacher | MC posterior, full-context scoring, temperature/clipping, accept rule | `sft_8b/posterior.py`; `sft_8b/bayesian_agent.py`; `sft_8b/prompts.py`; `sft_8b/results/lora_run/lora_best/`; `opponent_model/results/turn_eval_bayesian_lambda1.0_m5_f0.50_full150/`; `sft_8b/scripts/run_bayesian_protocol3.lsf` |
| 3.4 Distillation to Student | Day 7 teacher corpus, Day 8 SFT corpus, tagged output schema | `sft_8b/build_distill_data.py`; `sft_8b/results/distill/day7/day7_summary.json`; `sft_8b/results/distill/day7/day7_distill.jsonl`; `sft_8b/build_day8_sft_data.py`; `sft_8b/results/day8_sft_data/day8_data_summary.json`; `sft_8b/student_prompts.py`; `sft_8b/student_parser.py`; `sft_8b/train.py`; `sft_8b/results/day8_lora_run/` |
| 3.5 Menu Scoring | Enumerating all 64 allocations, `U_self + lambda * E[U_opp]` | `sft_8b/menu.py`; `sft_8b/bayesian_agent.py`; `sft_8b/svo_to_lambda.py`; `data/validation_notes.md` |
| 3.6 Evaluation Protocol | Protocol 3, metrics, held-out support, model configs | `opponent_model/turn_eval_run.py`; `opponent_model/turn_level_metrics.py`; `opponent_model/metrics.py`; `opponent_model/turn_level_metrics.py`; `opponent_model/results/*/turn_summary.json`; `structured_cot/P1_VS_P3_BASELINE_FRAMING.md`; `structured_cot/protocol_alignment_audit.md` |
| 4 Results | Posterior transfer, auditability, ablations, DND, decision quality, SVO | Result directories under `opponent_model/results/`; scripts under `opponent_model/scripts/`; ablation code under `sft_8b/ablation*.py`; DND code under `sft_8b/dnd*.py` |
| 4.1 Auditable Posterior Quality | Teacher/student Brier, elicited 70B posterior, prior CaSiNo SOTA anchor | `opponent_model/results/day9_headline_artifacts/`; `opponent_model/scripts/make_day9_headline_artifacts.py`; `opponent_model/results/turn_eval_bayesian_lambda1.0_m5_f0.50_full150/`; `opponent_model/results/turn_eval_student_balanced_full150/`; `structured_cot/results/structured_opp_inference_70b_pipeline_table12_full150/`; `sft_8b/results/cv/cv_summary.json` |
| 4.2 Auditability in Use | Belief-policy decomposition, case studies, posterior correction | `opponent_model/results/day11_auditability/`; `opponent_model/scripts/make_auditability_artifacts.py`; `opponent_model/results/ablation_neurips2026/a2d_correct_prefix/`; `opponent_model/results/ablation_neurips2026/a2d_adversarial_prefix/`; `sft_8b/ablation_student.py` |
| 4.3 Ablation Analysis | A1/A2/A3/A4/A5/A8/A9 variants | `sft_8b/ablation.py`; `sft_8b/ablation_student.py`; `sft_8b/ablation_aggregate.py`; `sft_8b/scripts/run_ablation_*.lsf`; `opponent_model/results/ablation_neurips2026/aggregate/ablation_results.md`; `opponent_model/results/ablation_neurips2026/aggregate/a8_a9_brier_heatmap.png` |
| 4.4 Cross-domain sanity check | DND six-ordering transfer headline | `opponent_model/results/dnd_transfer/aggregate/dnd_transfer_results.md`; `opponent_model/results/dnd_transfer/aggregate/artifact_manifest.json`; `data/dnd/dnd_data_stats.json`; `sft_8b/dnd_eval.py`; `sft_8b/dnd_posterior.py`; `sft_8b/dnd_menu.py` |
| 4.5 Decision quality | Table 3 accept/bid/strategy/Brier comparison | `opponent_model/results/turn_eval_structured_cot_p3_live_70b_m1_150/turn_summary.json`; `opponent_model/results/turn_eval_bayesian_lambda1.0_m5_f0.50_full150/turn_summary.json`; `opponent_model/results/turn_eval_student_balanced_full150/turn_summary.json`; `opponent_model/results/day9_extracted_bid_analysis/summary.json` |
| 4.6 SVO diagnostic | SVO-as-lambda null and scale sensitivity | `sft_8b/svo_to_lambda.py`; `opponent_model/results/day10_svo_accept_diagnostic_rescaled/`; `opponent_model/results/day10_svo_accept_diagnostic_legacy/`; `opponent_model/results/day10_svo_subgroup_rescaled/`; `opponent_model/results/day10_svo_subgroup_legacy/`; `opponent_model/scripts/diagnose_svo_accept_match_mismatch.py`; `opponent_model/scripts/make_svo_subgroup_table.py` |
| References | Bibliographic claims | Mostly paper text. Local source anchors: `CaSiNo/README.md`, `opponent_model_benchmark/README.md`, `structured_cot/`, `data/dnd/raw/` |

## Figures and Tables

| Paper item | Local source/result |
|---|---|
| Figure 1, modular architecture | No standalone source file found. The architecture is implemented across `sft_8b/posterior.py`, `sft_8b/menu.py`, `sft_8b/bayesian_agent.py`, `sft_8b/student_prompts.py`, and `opponent_model/turn_eval_run.py`. |
| Figure 2, Brier trajectory | `opponent_model/results/day9_headline_artifacts/brier_trajectory.png`; generated by `opponent_model/scripts/make_day9_headline_artifacts.py`; checks in `opponent_model/results/day9_headline_artifacts/headline_checks.json` and `headline_numbers.csv` |
| Table 1, belief-policy decomposition | `opponent_model/results/day11_auditability/auditability_metrics_summary.json`; `belief_policy_cases.csv`; `belief_policy_cases.md`; generated by `opponent_model/scripts/make_auditability_artifacts.py` |
| Table 2, ablations | `opponent_model/results/ablation_neurips2026/aggregate/ablation_results.md`; source runs under `opponent_model/results/ablation_neurips2026/a*/`; generated by `sft_8b/ablation_aggregate.py` |
| Table 3, main Protocol 3 metrics | `opponent_model/results/turn_eval_structured_cot_p3_live_70b_m1_150/turn_summary.json`; `opponent_model/results/turn_eval_bayesian_lambda1.0_m5_f0.50_full150/turn_summary.json`; `opponent_model/results/turn_eval_student_balanced_full150/turn_summary.json`; 70B elicited posterior in `structured_cot/results/structured_opp_inference_70b_pipeline_table12_full150/summary.json` |
| Figure 3, Big Five x strategy heatmap | `opponent_model/results/day10_human_bigfive_strategy_matrix/bigfive_strategy_pearson_heatmap.png`; source CSVs in same directory; generated by `opponent_model/scripts/make_human_bigfive_strategy_matrix.py` |
| Figure 4, student prompt/output schema | No standalone image source found. Text/schema source is `sft_8b/student_prompts.py`; example rows are in `sft_8b/results/day8_sft_data/day8_eval_rows.jsonl`; parser is `sft_8b/student_parser.py`. |
| Figure 5, likelihood sensitivity heatmap | `opponent_model/results/ablation_neurips2026/aggregate/a8_a9_brier_heatmap.png`; results in `opponent_model/results/ablation_neurips2026/aggregate/ablation_results.md`; source runs under `a8_*` and `a9_*` |
| Figure 6, posterior trajectories | `opponent_model/results/day11_auditability/posterior_trajectories.png`; manifest and trajectory cases in `opponent_model/results/day11_auditability/artifact_manifest.json` |
| Tables 4-7 | Table 4: `sft_8b/results/distill/day7/day7_summary.json` and `sft_8b/results/day8_sft_data/day8_data_summary.json`; Table 5: `opponent_model/results/turn_eval_student_balanced_full150/turn_summary.json` and `student_parse_failures.jsonl`; Table 6: `opponent_model/results/day9_headline_artifacts/headline_numbers.csv` and `headline_checks.json`; Table 7: `sft_8b/results/day8_lora_run/lora_best/sft_run_meta.json` and `sft_8b/results/day8_lora_run/training_args.json` |
| Table 8, compute resources | LSF outputs: `structured_cot/results/lsf/p3_baseline_70b.240396097.out`; `sft_8b/results/lsf/bayes_p3.240396199.out`; `sft_8b/results/lsf/day8_sft.239276962.out`; `sft_8b/results/lsf/day9_student_p3.240396201.out`; cache-only reruns in `sft_8b/results/lsf/day9_student_p3.240522669.out` / `240522670.out` |
| Table 9 and Table 17, DND transfer | `opponent_model/results/dnd_transfer/aggregate/dnd_transfer_results.md`; run summaries in `opponent_model/results/dnd_transfer/eval_*/turn_summary.json` |
| Table 10, SVO match/mismatch accept | `opponent_model/results/day10_svo_accept_diagnostic_rescaled/summary.json`; `opponent_model/results/day10_svo_accept_diagnostic_legacy/summary.json` |
| Table 11, legacy SVO x IP | `opponent_model/results/day10_svo_ip_table/table1_svo_ip.md`; `opponent_model/results/day10_svo_ip_table/table1_svo_ip.csv`; generated by `opponent_model/scripts/make_svo_ip_table.py` |
| Table 12, rescaled SVO subgroup | `opponent_model/results/day10_svo_subgroup_rescaled/table1_svo_subgroup.md`; `table1_svo_subgroup.csv` |
| Table 13, likelihood sweeps | `opponent_model/results/ablation_neurips2026/aggregate/ablation_results.md`; `a8_temp_*`; `a9_clip_*`; generated by `sft_8b/ablation.py` and aggregated by `sft_8b/ablation_aggregate.py` |
| Table 14, posterior prefix injection | `opponent_model/results/ablation_neurips2026/a2d_correct_prefix/turn_summary.json`; `opponent_model/results/ablation_neurips2026/a2d_adversarial_prefix/turn_summary.json`; `opponent_model/results/day11_auditability/auditability_metrics_summary.json` |
| Table 15, auditability cases | `opponent_model/results/day11_auditability/belief_policy_cases.md`; `belief_policy_cases.csv` |
| Table 16, external CaSiNo opponent modeling | `sft_8b/results/cv/cv_summary.json`; fold artifacts under `sft_8b/results/cv/fold_*/`; 70B reference in `structured_cot/results/structured_opp_inference_70b_pipeline_table12_full150/summary.json`; prior-code anchor in `opponent_model_benchmark/` |

## Appendices

| Appendix | Title | Corresponding files |
|---|---|---|
| A | Limitations and Broader Impact | No single dedicated source file. Supporting artifacts: `README.md`, `roadmap.md`, `structured_cot/protocol_alignment_audit.md`, `structured_cot/baseline_weaknesses.md`, `data/validation_notes.md`, `opponent_model/results/day11_auditability/auditability_section_draft.md`, SVO null files under `opponent_model/results/day10_*` |
| B | Dataset Split and Quality Filtering | `data/build_quality_subdataset.py`; `data/quality_audit.csv`; `data/validation_notes.md`; `data/casino_train.json`; `data/casino_test.json`; `data/casino_train_w0.2.json`; `data/casino_train_w0.5.json`; `data/casino_train_w0.8.json`; `CaSiNo/data/casino.json` |
| C | Distillation Data Audit | `sft_8b/build_distill_data.py`; `sft_8b/results/distill/day7/day7_summary.json`; `day7_distill.jsonl`; `day7_failures.jsonl`; `sft_8b/build_day8_sft_data.py`; `sft_8b/results/day8_sft_data/day8_data_summary.json`; `day8_train_rows.jsonl`; `day8_eval_rows.jsonl` |
| D | Structured Output Validity | `sft_8b/student_parser.py`; `sft_8b/student_model.py`; `tests/test_student_parser.py`; `tests/test_auditability_artifacts.py`; `opponent_model/results/turn_eval_student_balanced_full150/turn_summary.json`; `opponent_model/results/turn_eval_student_balanced_full150/student_parse_failures.jsonl` |
| E | Full Metric Breakdown | `opponent_model/results/*/turn_summary.json`; `opponent_model/results/*/turn_records.jsonl`; `opponent_model/results/day9_headline_artifacts/headline_numbers.csv`; `headline_checks.json`; `opponent_model/results/day9_extracted_bid_analysis/summary.json` |
| F | Bid Coverage Diagnostic | `opponent_model/scripts/analyze_extracted_bids.py`; `opponent_model/scripts/day9_student_coverage_spotcheck.py`; `opponent_model/bid_extractor.py`; `opponent_model/results/day9_extracted_bid_analysis/summary.json`; `opponent_model/results/day9_student_coverage_spotcheck/summary.json`; `methods_note.md`; `debug_rows.jsonl` |
| G | Protocol Alignment: Replay vs Turn-Level Evaluation | `structured_cot/P1_VS_P3_BASELINE_FRAMING.md`; `structured_cot/protocol_alignment_audit.md`; `structured_cot/baseline_weaknesses.md`; `structured_cot/results/protocol1_70b_full/`; `opponent_model/results/turn_eval_structured_cot_replay_full150/`; `opponent_model/results/turn_eval_structured_cot_p3_live_70b_m1_150/`; `structured_cot/run_protocol3_baseline.py`; `structured_cot/scripts/run_p3_baseline_70b.lsf` |
| H | Expanded SVO Diagnostics | `sft_8b/svo_to_lambda.py`; `opponent_model/scripts/check_human_svo_baseline.py`; `opponent_model/scripts/diagnose_svo_accept_match_mismatch.py`; `opponent_model/scripts/make_svo_subgroup_table.py`; `opponent_model/scripts/make_svo_ip_table.py`; `opponent_model/results/day10_svo_*`; `opponent_model/results/turn_eval_bayesian_svo_lambda_*` |
| I | Big Five and Style-Token Null Results | `sft_8b/bigfive_to_style.py`; `opponent_model/scripts/check_style_variant_strategy_distributions.py`; `opponent_model/scripts/make_human_bigfive_strategy_matrix.py`; `opponent_model/results/day10_style_variant_strategy_distributions/`; `opponent_model/results/day10_human_bigfive_strategy_matrix/`; `opponent_model/results/day9_bigfive_distribution/`; student style runs under `opponent_model/results/turn_eval_student_*_full150/` |
| J | Prompt and Output Schema | `sft_8b/student_prompts.py`; `sft_8b/student_parser.py`; `sft_8b/build_day8_sft_data.py`; `sft_8b/results/day8_sft_data/day8_eval_rows.jsonl`; `structured_cot/prompts.py`; `sft_8b/prompts.py` |
| K | LoRA Training Details | `sft_8b/train.py`; `sft_8b/scripts/run_day8_sft_train.lsf`; `sft_8b/results/day8_lora_run/lora_best/sft_run_meta.json`; `sft_8b/results/day8_lora_run/training_args.json`; `sft_8b/results/day8_lora_run/lora_best/trainer_state.json`; teacher adapter metadata under `sft_8b/results/lora_run/` |
| L | Reproducibility Statement | `README.md`; `environment.yml`; `setup.py`; all LSF scripts under `sft_8b/scripts/`, `structured_cot/scripts/`, and `opponent_model/scripts/`; run configs in `*/args.json`, `*/run_config.json`, `*/turn_summary.json`, and `*/training_args.json` |
| M | Compute Resources | LSF logs under `structured_cot/results/lsf/`, `sft_8b/results/lsf/`, `opponent_model/results/ablation_neurips2026/lsf/`, `opponent_model/results/dnd_transfer/lsf/`; main compute rows map to `p3_baseline_70b.240396097.out`, `bayes_p3.240396199.out`, `day8_sft.239276962.out`, `day9_student_p3.240396201.out` |
| N | Assets, Licenses, and Data Use | `CaSiNo/LICENSE`; `CaSiNo/README.md`; `CaSiNo/data/README.md`; `opponent_model_benchmark/LICENSE`; `opponent_model_benchmark/README.md`; `data/dnd/raw/`; `data/dnd/dnd_data_stats.json`; model paths recorded in `turn_summary.json`, `sft_run_meta.json`, and `training_args.json` |
| O | LLM Usage | `sft_8b/posterior.py`; `sft_8b/student_model.py`; `sft_8b/train.py`; `structured_cot/llm_client.py`; `structured_cot/live_turn_agent.py`; `opponent_model/turn_agents.py`; all model configs in result summaries and training args |
| P | Cross-Domain Transfer to Deal-or-No-Deal | `sft_8b/dnd_data.py`; `sft_8b/dnd_posterior.py`; `sft_8b/dnd_eval.py`; `sft_8b/dnd_menu.py`; `sft_8b/dnd_metrics.py`; `sft_8b/build_dnd_sft_data.py`; `data/dnd/`; `opponent_model/results/dnd_transfer/aggregate/dnd_transfer_results.md` |
| Q | SVO-as-lambda Scale Confound | `sft_8b/svo_to_lambda.py`; `sft_8b/menu.py`; `opponent_model/results/day10_svo_accept_diagnostic_legacy/`; `opponent_model/results/day10_svo_accept_diagnostic_rescaled/`; `opponent_model/results/day10_svo_subgroup_legacy/`; `opponent_model/results/day10_svo_ip_table/`; scripts listed under H |
| R | SVO Subgroup Analysis Under Rescaled Mapping | `opponent_model/results/day10_svo_subgroup_rescaled/table1_svo_subgroup.md`; `table1_svo_subgroup.csv`; `summary.json`; `opponent_model/results/turn_eval_bayesian_svo_lambda_rescaled_rerun_m5_f0.50_full150/`; `opponent_model/results/turn_eval_bayesian_svo_lambda_rescaled_mismatch_m5_f0.50_full150/` |
| S | Likelihood Sensitivity Sweeps | `sft_8b/ablation.py`; `sft_8b/scripts/run_ablation_sweeps.lsf`; `opponent_model/results/ablation_neurips2026/a8_temp_*/`; `opponent_model/results/ablation_neurips2026/a9_clip_*/`; `opponent_model/results/ablation_neurips2026/aggregate/a8_a9_brier_heatmap.png`; `ablation_results.md` |
| T | Belief-Action Coupling Diagnostic | `sft_8b/ablation_student.py`; `sft_8b/ablation_interventions.py`; `sft_8b/scripts/run_ablation_interventions.lsf`; `opponent_model/results/ablation_neurips2026/a2d_correct_prefix/`; `opponent_model/results/ablation_neurips2026/a2d_adversarial_prefix/`; `opponent_model/results/day11_auditability/posterior_correction_cases.csv`; `auditability_metrics_summary.json` |
| U | Auditability Case Studies | `opponent_model/scripts/make_auditability_artifacts.py`; `opponent_model/results/day11_auditability/artifact_manifest.json`; `belief_policy_cases.md`; `belief_policy_cases.csv`; `posterior_trajectories.png`; `entropy_confidence_diagnostic.csv`; `auditability_section_draft.md` |
| V | External Comparison to Prior CaSiNo Opponent Modeling | `sft_8b/cv_data.py`; `sft_8b/cv_aggregate.py`; `sft_8b/scripts/run_cv_fold.lsf`; `sft_8b/scripts/submit_all_folds.sh`; `sft_8b/results/cv/cv_summary.json`; `sft_8b/results/cv/cv_split_manifest.json`; fold directories under `sft_8b/results/cv/fold_*/`; `opponent_model_benchmark/`; `CaSiNo/strategy_prediction/`; 70B prompt comparison in `structured_cot/results/structured_opp_inference_70b_pipeline_table12_full150/summary.json` |
| W | Deal-or-No-Deal Transfer Details | `opponent_model/results/dnd_transfer/aggregate/dnd_transfer_results.md`; run directories under `opponent_model/results/dnd_transfer/eval_*/`; `opponent_model/results/dnd_transfer/sft_data/fewshot_opponent/`; `opponent_model/results/dnd_transfer/fewshot_lora/`; `sft_8b/scripts/run_dnd_*.lsf` |
| X | DND Transfer Protocol | `sft_8b/dnd_data.py`; `sft_8b/dnd_prompts.py`; `sft_8b/dnd_posterior.py`; `sft_8b/dnd_rules.py`; `sft_8b/dnd_eval.py`; `sft_8b/dnd_metrics.py`; `data/dnd/raw/`; `data/dnd/dnd_data_stats.json` |
| NeurIPS checklist | Claims, limitations, reproducibility, assets, LLM usage | Mostly paper text, supported by `README.md`, `environment.yml`, `setup.py`, `CaSiNo/LICENSE`, `opponent_model_benchmark/LICENSE`, Appendix L/N/O files above |

## Gaps / Things Not Found as Standalone Files

| Missing or weakly sourced item | Status |
|---|---|
| Editable source for Figure 1 architecture | Not found. Only implemented components and paper-rendered figure are present. |
| Editable source for Figure 4 prompt/schema screenshot | Not found. Schema text is in `sft_8b/student_prompts.py`; examples are in Day 8 JSONL rows. |
| Saved pytest log for "15/15 passed" | Not found in the file scan. The tests themselves are under `tests/`, and Table 5 cites this, but I did not find a persisted pytest-output artifact. |
| Standalone limitations/broader-impact source | Not found. Appendix A appears written directly from result/context artifacts rather than a single local draft file. |
| Standalone bibliography/source `.bib` or `.tex` | Not found. Only the PDF is present. |

## Quick Reproduction Entry Points

| Goal | Command/file |
|---|---|
| Build quality subsets | `python data/build_quality_subdataset.py` |
| Build Day 7 distillation data | `bsub < sft_8b/scripts/run_distill_data.lsf` |
| Build Day 8 student rows | `python -m sft_8b.build_day8_sft_data` |
| Train Day 8 student | `bsub < sft_8b/scripts/run_day8_sft_train.lsf` |
| Run Bayesian Protocol 3 | `bsub < sft_8b/scripts/run_bayesian_protocol3.lsf` |
| Run student Protocol 3 | `bsub < sft_8b/scripts/run_day9_student_eval.lsf` |
| Run 70B Structured-CoT Protocol 3 | `bsub < structured_cot/scripts/run_p3_baseline_70b.lsf` |
| Make headline Brier artifacts | `python -m opponent_model.scripts.make_day9_headline_artifacts` |
| Aggregate ablations | `bsub < sft_8b/scripts/run_ablation_aggregate.lsf` |
| Make auditability artifacts | `python -m opponent_model.scripts.make_auditability_artifacts` |
| Aggregate DND transfer | `bsub < sft_8b/scripts/run_dnd_aggregate.lsf` |

