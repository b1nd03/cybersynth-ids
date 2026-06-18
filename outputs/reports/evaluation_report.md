# CyberSynth Evaluation Report

## Executive Summary

- Generated synthetic dataset contains 500,000 rows.
- Real-only F1 is 0.9916; synthetic-only reaches 68.47% of real-only F1.
- Augmented training reaches 100.04% of real-only F1 on the held-out real test set.

## Synthetic Dataset

- Rows: 500,000
- Columns: 47
- Labels: {0: 250000, 1: 250000}

## Fidelity

- Average JS divergence: 0.043237
- KS failure rate (p < 0.05): 0.9189
- Correlation preservation: 0.3586

## Utility on Held-Out Real Test Set

| Protocol | F1 | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Real only | 0.9916 | 0.9909 | 0.9924 | 0.9998 |
| Synthetic only | 0.6789 | 0.5193 | 0.9803 | 0.6710 |
| Real + synthetic | 0.9920 | 0.9914 | 0.9926 | 0.9997 |

Synthetic-only relative F1: 68.47%

Augmented relative F1: 100.04%

## Privacy

- Exact real row matches: 0
- DCR p05: 0.1319
- DCR median: 2.4252
- NNDR median: 0.9932
- k-anonymity minimum: 1

## Recommendation

Synthetic dataset is usable for experimentation, but future work should add CTGAN/TVAE backends and stronger privacy auditing before public release.
