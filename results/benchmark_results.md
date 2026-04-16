# Benchmark Results without judging preferences 

## Model
**Meta-Llama-3.3-70B-Instruct**
- Temperature: 0.0
- Max new tokens: 128
- Data: `CaSiNo/data/casino_ann.json`
- Dialogues evaluated: 396

---

## Per-Label Results

| Strategy | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| small-talk | 0.654 | 0.796 | 0.718 | 1054 |
| other-need | 0.570 | 0.768 | 0.654 | 409 |
| self-need | 0.482 | 0.857 | 0.617 | 964 |
| elicit-pref | 0.448 | 0.981 | 0.615 | 377 |
| no-need | 0.520 | 0.673 | 0.587 | 196 |
| vouch-fair | 0.418 | 0.733 | 0.533 | 439 |
| uv-part | 0.540 | 0.511 | 0.525 | 131 |
| non-strategic | 0.676 | 0.421 | 0.519 | 1455 |
| promote-coordination | 0.309 | 0.855 | 0.454 | 579 |
| showing-empathy | 0.363 | 0.480 | 0.414 | 254 |

---

## Overall Results

| Metric | Precision | Recall | F1 |
|---|---|---|---|
| Macro | 0.498 | 0.708 | 0.563 |
| Micro | 0.490 | 0.700 | 0.576 |

---

## Notes

- Recall is generally high but precision is low — model over-predicts most labels
- `elicit-pref`: recall=0.981 but precision=0.448 → model labels almost everything as elicit-pref
- `promote-coordination`: same pattern — recall=0.855, precision=0.309
- `non-strategic`: precision=0.676 but recall=0.421 → model misses many non-strategic utterances (labels them as something else)
- `showing-empathy`: lowest F1 (0.414) — hardest to detect
- `small-talk`: best F1 (0.718) — easiest to detect

---


# Benchmark Results with personality traits

## Model
**Meta-Llama-3.3-70B-Instruct**
- Temperature: 0.0
- Max new tokens: 128
- Data: `CaSiNo/data/casino_ann.json`
- Dialogues evaluated: 396
- Utterances evaluated: 4615

---

## Per-Label Results

| Strategy | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| small-talk | 0.771 | 0.676 | 0.720 | 1054 |
| self-need | 0.458 | 0.851 | 0.595 | 964 |
| other-need | 0.664 | 0.469 | 0.550 | 409 |
| vouch-fair | 0.498 | 0.554 | 0.524 | 439 |
| no-need | 0.468 | 0.531 | 0.498 | 196 |
| elicit-pref | 0.322 | 0.979 | 0.484 | 377 |
| non-strategic | 0.598 | 0.364 | 0.453 | 1455 |
| promote-coordination | 0.268 | 0.834 | 0.406 | 579 |
| uv-part | 0.358 | 0.336 | 0.346 | 131 |
| showing-empathy | 0.243 | 0.543 | 0.336 | 254 |

---

## Overall Results

| Metric | Precision | Recall | F1 |
|---|---|---|---|
| Macro | 0.465 | 0.614 | 0.491 |
| Micro | 0.441 | 0.621 | 0.516 |

---

## Preference Prediction Results

| Metric | Value |
|---|---|
| Item Accuracy | 0.399 |
| Exact Match Rate | 0.222 |
| Dialogues Evaluated | 396 |

---

## Notes

- Recall is generally high but precision is low — model over-predicts most labels
- `elicit-pref`: recall=0.979 but precision=0.322 → model labels almost everything as elicit-pref
- `promote-coordination`: same pattern — recall=0.834, precision=0.268
- `self-need`: recall=0.851 but precision=0.458 → significant over-prediction
- `non-strategic`: precision=0.598 but recall=0.364 → model misses many non-strategic utterances
- `uv-part` and `showing-empathy`: lowest F1s (0.346, 0.336) — hardest to detect
- `small-talk`: best F1 (0.720) — easiest to detect
- Preference prediction: item accuracy=0.399, exact match=0.222 — well above random chance but room for improvement

---

# Benchmark Results with personality traits

## Model
**Meta-Llama-3.3-70B-Instruct**
- Temperature: 0.0
- Max new tokens: 128
- Data: `CaSiNo/data/casino_ann.json`
- Dialogues evaluated: 396
- Utterances evaluated: 4615

---

## Per-Label Results

| Strategy | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| small-talk | 0.788 | 0.723 | 0.754 | 1054 |
| elicit-pref | 0.475 | 0.955 | 0.634 | 377 |
| self-need | 0.471 | 0.890 | 0.616 | 964 |
| other-need | 0.720 | 0.521 | 0.604 | 409 |
| vouch-fair | 0.467 | 0.554 | 0.507 | 439 |
| non-strategic | 0.598 | 0.414 | 0.490 | 1455 |
| no-need | 0.355 | 0.668 | 0.464 | 196 |
| promote-coordination | 0.277 | 0.827 | 0.415 | 579 |
| showing-empathy | 0.315 | 0.579 | 0.408 | 254 |
| uv-part | 0.337 | 0.450 | 0.386 | 131 |

---

## Overall Results

| Metric | Precision | Recall | F1 |
|---|---|---|---|
| Macro | 0.480 | 0.658 | 0.528 |
| Micro | 0.475 | 0.658 | 0.552 |

---

## Preference Prediction Results

| Metric | Value |
|---|---|
| Item Accuracy | 0.405 |
| Exact Match Rate | 0.235 |
| Dialogues Evaluated | 396 |

---

## Notes

- Recall is generally high but precision is low — model over-predicts most labels
- `elicit-pref`: recall=0.955 but precision=0.475 → model labels almost everything as elicit-pref
- `promote-coordination`: same pattern — recall=0.827, precision=0.277
- `self-need`: recall=0.890 but precision=0.471 → significant over-prediction
- `non-strategic`: precision=0.598 but recall=0.414 → model misses many non-strategic utterances
- `uv-part` and `showing-empathy`: lowest F1s (0.386, 0.408) — hardest to detect
- `small-talk`: best F1 (0.754) — easiest to detect
- Preference prediction: item accuracy=0.405, exact match=0.235 — slight improvement over prior run
