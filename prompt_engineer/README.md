# `prompt_engineer`

Reorganized workspace for the CaSiNo prompt-engineering baseline.

## Layout

- `core/`: agent logic, bidding, prompting, and opponent modeling
- `evaluation/`: offline evaluation harnesses and LLM-as-judge utilities
- `llm/`: model client code and download/smoke-test helpers
- `preprocessing/`: shared scoring and dataset statistics helpers
- `scripts/`: runnable experiment entrypoints
- `results/`: generated outputs and saved experiment artifacts

## Suggested Commands

Run the baseline entrypoint:

```bash
python -m prompt_engineer.scripts.run_baseline --help
```

Compute dataset stats:

```bash
python -m prompt_engineer.preprocessing.stats --help
```

Inspect the evaluation harness:

```bash
python -m prompt_engineer.evaluation.evaluate_classifier
python -m prompt_engineer.evaluation.judge
```
