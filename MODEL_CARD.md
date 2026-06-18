# Model Card: CyberSynth IDS Baseline

## Model Details

- Model: LightGBM baseline IDS classifier
- Task: binary intrusion detection
- Output: normal or attack probability
- Interface: FastAPI dashboard and JSON API

## Intended Use

CyberSynth IDS is intended for cybersecurity research, education, portfolio demonstration, and local experimentation with intrusion detection workflows. It helps users prepare normalized network-flow data, score flows, inspect model behavior, generate controlled synthetic rows, and monitor drift.

## Inputs

The model uses normalized network-flow features such as protocol, service, ports, bytes, packets, timing fields, packet length statistics, flag counts, and derived traffic ratios.

Example feature groups:

- Protocol and service fields
- Source and destination ports
- Source and destination byte counts
- Forward and backward packet counts
- Packet length statistics
- Flow byte and packet rates
- Inter-arrival timing features
- TCP flag counts
- Active and idle timing features
- Derived traffic ratios

## Excluded From Training

The following metadata fields are kept for reporting, filtering, and analysis, but are excluded from the training feature matrix:

- `dataset_source`
- `source_file`
- `environment_type`
- `attack_category`
- `attack_subcategory`

## Preprocessing

Supported datasets are mapped into a shared network-flow schema before training. Numeric fields are converted to numeric values and imputed with training medians. Categorical fields such as protocol and service are filled with `missing` when needed and encoded with unknown-value handling so new categorical values do not break prediction.

Metadata fields remain available for reporting, dataset filters, per-category evaluation, and drift analysis, but they are not used as direct model features.

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

## Ethical And Security Notes

This project is intended for defensive research, education, and local demonstration. It should not be used as the only control for protecting a production network. Uploaded data should be reviewed before use, and sensitive network records should stay out of public repositories.

Security-relevant deployment settings include API-key protection, admin-token model reloads, upload limits, CORS allowlists, rate limiting, audit logging, and optional model hash verification.
