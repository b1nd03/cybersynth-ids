# Model Card: CyberSynth IDS Baseline

## Model Details

- Model: LightGBM baseline IDS classifier
- Task: binary intrusion detection
- Output: normal or attack probability
- Interface: FastAPI dashboard and JSON API

## Inputs

The model uses normalized network-flow features such as protocol, service, ports, bytes, packets, timing fields, packet length statistics, flag counts, and derived traffic ratios.

## Excluded From Training

The following metadata fields are kept for reporting, filtering, and analysis, but are excluded from the training feature matrix:

- `dataset_source`
- `source_file`
- `environment_type`
- `attack_category`
- `attack_subcategory`

## Evaluation

Primary metrics:

- F1
- Precision
- Recall
- ROC-AUC
- Per-category detection reporting
- Confusion matrix review

Current validation snapshot:

- Baseline LightGBM F1: `0.9916`
- Real plus generated-data F1: `0.9920`
- Generated-data exact real row matches: `0`

## Training Data

CyberSynth IDS uses a normalized network-flow schema so multiple supported IDS datasets can be mapped into consistent columns before training. Dataset-source and category metadata remain available for analysis without becoming direct model inputs.

The public repository includes tiny synthetic demo splits under `data/processed/` so the Docker dashboard can run after cloning. Full processed train/test rows are not required in the public repo and should be regenerated locally from datasets the user is allowed to use.

## Limitations

Model behavior depends on the datasets used during preprocessing and training. Important review points include dataset bias, distribution shift, feature compatibility, and attack families that may be underrepresented in the training split.

Generated data is designed for augmentation experiments, balancing, and workflow testing. Important model claims should be checked against held-out real traffic.

## Ethical Use

This project is intended for research, education, and local demonstration. It should be used to study IDS workflows, dataset preparation, model evaluation, explainability, and drift monitoring.
