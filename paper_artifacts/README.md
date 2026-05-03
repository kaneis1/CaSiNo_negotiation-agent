# Paper Artifacts Overlay

This directory is a paper-facing overlay for reviewers and reproducers. It does not move or rename the runnable project code, raw results, checkpoints, JSONL files, or LSF logs.

`manifest.yml` is the source of truth. The files in `claims/`, `tables/`, `figures/`, `03_methods/`, and `04_results/` are navigation aids that point back to manifest artifact IDs.

## Layout

| Folder | Purpose |
|---|---|
| `00_manuscript/` | Symlinks to the PDF, section-file map, and number audit. |
| `01_data/` | Paper-facing aliases for CaSiNo and DND data artifacts. |
| `02_code_index/` | Entry points, experiment scripts, training scripts, and evaluation scripts. |
| `03_methods/` | Short method-section notes with code anchors. |
| `04_results/` | Result-section notes grouped by paper section. |
| `shared_results/` | Reused result-directory aliases with stable names. |
| `figures/` | Paper-numbered figure aliases. |
| `tables/` | Paper-numbered table-source folders. |
| `claims/` | Claim-centric summaries for revision and auditability discussion. |

## Validation

Run:

```bash
python scripts/check_paper_artifacts_manifest.py
find -L paper_artifacts -type l -print
```

The second command should print nothing.
