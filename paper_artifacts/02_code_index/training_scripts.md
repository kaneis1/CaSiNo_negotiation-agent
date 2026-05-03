# Training Scripts

Manifest IDs are authoritative; this file indexes training-related code.

| Paper location | Script/module | Main output | Cost | Manifest IDs |
|---|---|---|---|---|
| Appendix C | `sft_8b/build_distill_data.py` | `sft_8b/results/distill/day7` | Expensive/GPU if posterior generation is uncached | `table_04_distillation_data_audit` |
| Appendix C | `sft_8b/build_day8_sft_data.py` | `sft_8b/results/day8_sft_data` | Cheap/local | `table_04_distillation_data_audit` |
| Appendix K | `sft_8b/scripts/run_day8_sft_train.lsf` | `sft_8b/results/day8_lora_run` | Expensive/GPU | `table_07_lora_training_details` |
| Appendix V | `sft_8b/scripts/run_cv_fold.lsf` | `sft_8b/results/cv/fold_*` | Expensive/GPU | `table_16_external_casino_comparison` |
| Appendix W/X | `sft_8b/scripts/run_dnd_fewshot_train.lsf` | `opponent_model/results/dnd_transfer/fewshot_lora` | Expensive/GPU | `shared_dnd_transfer_results` |
