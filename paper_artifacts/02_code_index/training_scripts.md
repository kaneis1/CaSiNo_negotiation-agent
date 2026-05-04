# Training Scripts

Manifest IDs are authoritative; this file indexes training-related code.

| Paper location | Script/module | Main output | Cost | Manifest IDs |
|---|---|---|---|---|
| Appendix C | `src/casino_belief/training/build_distill_data.py` | `artifacts/training_metadata/distill/day7` | Expensive/GPU if posterior generation is uncached | `table_04_distillation_data_audit` |
| Appendix C | `src/casino_belief/training/build_day8_sft_data.py` | `artifacts/training_metadata/day8_sft_data` | Cheap/local | `table_04_distillation_data_audit` |
| Appendix K | `experiments/lsf/training/run_day8_sft_train.lsf` | `artifacts/training_metadata/day8_lora_run` | Expensive/GPU | `table_07_lora_training_details` |
| Appendix V | `experiments/lsf/training/run_cv_fold.lsf` | `artifacts/results/external_comparison/cv/fold_*` | Expensive/GPU | `table_16_external_casino_comparison` |
| Appendix W/X | `experiments/lsf/dnd_transfer/run_dnd_fewshot_train.lsf` | `artifacts/results/dnd_transfer/main/fewshot_lora` | Expensive/GPU | `shared_dnd_transfer_results` |
