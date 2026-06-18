# Dataset Card: CyberSynth Synthetic Dataset v1.0

## Dataset Summary

CyberSynth Synthetic Dataset v1.0 is a 500,000-row synthetic cybersecurity dataset generated from the local unified IDS training split. It is designed for experimentation with binary intrusion detection and synthetic-data augmentation.

## Files

- Dataset: `data/synthetic/synthetic_dataset_v1.0.parquet`
- Summary: `data/synthetic/synthetic_dataset_v1.0_summary.json`
- Evaluation JSON: `outputs/reports/evaluation_report.json`
- Evaluation Markdown: `outputs/reports/evaluation_report.md`

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

## Evaluation Results

Held-out real test set utility:

| Protocol | F1 | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Real only | 0.9916 | 0.9909 | 0.9924 | 0.9998 |
| Synthetic only | 0.6789 | 0.5193 | 0.9803 | 0.6710 |
| Real + synthetic | 0.9920 | 0.9914 | 0.9926 | 0.9997 |

Fidelity and privacy:

- Average JS divergence: 0.043237
- KS failure rate: 0.9189
- Correlation preservation: 0.3586
- Exact real row matches: 0
- DCR p05: 0.1319
- DCR median: 2.4252
- NNDR median: 0.9932

## Intended Use

- IDS model augmentation experiments
- Coursework and research prototyping
- Testing preprocessing and evaluation workflows
- Comparing future CTGAN/TVAE/copula generator backends

## Not Intended For

- Public benchmark claims without further validation
- Operational IDS deployment
- Privacy-sensitive release without stronger membership-inference and DCR analysis
- Replacing real IDS benchmark data

## Limitations

The current generator is not a neural CTGAN/TVAE generator. Synthetic-only utility is below the target 85% relative F1 threshold, so the dataset is best used for augmentation experiments rather than as a full real-data replacement.

## Citation Guidance

If used in a report, cite the source datasets and describe this dataset as a locally generated synthetic augmentation artifact produced by the CyberSynth conditioned statistical generator.
