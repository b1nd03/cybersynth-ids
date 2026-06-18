# Dataset Card: CyberSynth Synthetic Dataset v1.0

## Dataset Summary

CyberSynth Synthetic Dataset v1.0 is a 500,000-row synthetic cybersecurity dataset generated from the local unified IDS training split. It is designed for experimentation with binary intrusion detection and synthetic-data augmentation.

## Files

- Dataset: `data/synthetic/synthetic_dataset_v1.0.parquet`
- Summary: `data/synthetic/synthetic_dataset_v1.0_summary.json`
- Evaluation JSON: `outputs/reports/evaluation_report.json`
- Evaluation Markdown: `outputs/reports/evaluation_report.md`

The generated synthetic dataset is not committed to the public repository. Regenerate it locally with the command in `README.md` when needed.

## Generation Method

Generator: `conditioned_statistical_bootstrap`

The generator samples from real training rows conditioned by attack category, adds numeric jitter, clips values to observed quantile bounds, rounds integer-like network fields, recomputes derived rates, and reduces exact real-row matches.

The v1.0 dataset uses `label_balanced` mode:

- 250,000 normal rows
- 250,000 attack rows
- Attack rows balanced across available attack categories

## Schema

- Rows: 500,000
- Columns: 47
- Feature columns: 41 listed in `data/synthetic/synthetic_dataset_v1.0_summary.json`
- Metadata and label columns: `dataset_source`, `source_file`, `environment_type`, `attack_category`, `attack_subcategory`, `label`

## Validation Snapshot

- Baseline LightGBM F1: 0.9916
- Real + synthetic F1: 0.9920
- Exact real row matches: 0

## Use Cases

- IDS model augmentation experiments
- Coursework and research prototyping
- Testing preprocessing and evaluation workflows
- Comparing generator backends

## Citation Guidance

If used in a report, cite the source datasets and describe this dataset as a locally generated synthetic augmentation artifact produced by the CyberSynth conditioned statistical generator.
