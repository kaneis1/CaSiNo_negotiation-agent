# Experiment Log
---
## 04/12/2026

### Benchmark Experiments

**Objective:** Establish a zero-shot baseline for negotiation strategy classification on the CaSiNo dataset using Meta-Llama-3.3-70B-Instruct. Evaluated per-label and overall (macro/micro) precision, recall, and F1 across 396 annotated dialogues. Results saved to `results/benchmark_results.md`.

---
## 04/13/2026

### Experiments change 

**Objective:** Input the whold dialogue into the model, instead of one uttrance with context window. Add the personality trait into prompt...