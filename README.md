# CyberSynth IDS

CyberSynth IDS is a local web tool for cybersecurity dataset experiments. It prepares network-flow datasets, trains a baseline intrusion detection model, generates filtered synthetic rows, validates uploaded CSV files, and shows model results through a FastAPI dashboard.

## What It Does

- Serves a FastAPI dashboard with separate pages for overview, flow prediction, dataset upload, synthetic dataset creation, model results, explainability, drift monitoring, and about.
- Validates uploaded CSV files before scoring, including row count, feature coverage, and missing model columns.
- Generates synthetic rows with label, attack category, source dataset, environment, and subcategory filters.
- Suggests clean dataset names from the selected labels, attack families, sources, and date.
- Trains a LightGBM baseline model with saved metrics, feature importance, threshold tuning, and per-category reporting.
- Explains single predictions with SHAP feature attribution.
- Compares uploaded CSV files against the training split with PSI drift checks.
- Supports optional API-key auth, admin-token model reload, rate limiting, audit logging, CORS allowlist, model hash verification, and security headers.

## How The Model Works With Different Datasets

The project is built around a shared network-flow schema. Each raw dataset can use different column names, labels, and attack names, so the preprocessing step maps them into common fields such as protocol, ports, byte counts, packet counts, flags, timing fields, labels, source dataset, environment, attack category, and attack subcategory.

During training, the model only learns from traffic features. Metadata columns such as `dataset_source`, `source_file`, `environment_type`, `attack_category`, and `attack_subcategory` are removed from the feature matrix so the model does not simply memorize where a row came from. These metadata fields are still kept for reporting, filtering, dataset generation, and per-category evaluation.

Numeric features are median-imputed. Categorical features such as `protocol` and `service` are imputed as `missing` and encoded with `handle_unknown="ignore"`, so a new service value in an uploaded CSV does not break prediction. If a new dataset uses a different feature meaning or very different traffic distribution, use the Drift page and retrain before trusting the scores.

Recommended workflow for adding a dataset:

1. Add or enable the dataset in `configs/active_datasets.yaml`.
2. Run preprocessing and splitting.
3. Train the baseline model.
4. Check overall metrics and per-category metrics.
5. Upload a held-out CSV from the new source and check drift.
6. Use generated data only for testing or augmentation, not as a replacement for final real-data evaluation.

## Project Layout

```text
configs/              Dataset and experiment configuration
docs/                 Project notes for model behavior
src/ingestion/         Preprocessing and train/test split scripts
src/evaluation/        Model training and synthetic-data evaluation scripts
src/generation/        Synthetic dataset generator
src/web/               FastAPI application
web/                   Dashboard HTML, CSS, and JavaScript
outputs/               Local metrics and report outputs
```

Large local files such as raw datasets, processed datasets, trained models, logs, local reports, cache folders, and private notes are ignored by Git.

## Quick Start

Install the web runtime dependencies:

```powershell
pip install -r requirements.txt
```

Run the dashboard:

```powershell
python -m uvicorn src.web.app:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

For full preprocessing, training, generation, and evaluation work:

```powershell
pip install -r requirements-pipeline.txt
```

For tests and linting tools:

```powershell
pip install -r requirements-dev.txt
```

## Main Commands

Preprocess datasets:

```powershell
python src\ingestion\preprocessor.py --config configs\active_datasets.yaml
python src\ingestion\validate_and_split.py
```

Train the baseline model:

```powershell
python src\evaluation\train_baseline.py
```

Create a synthetic dataset:

```powershell
python src\generation\generate_synthetic.py --rows 500000 --mode label_balanced --output data\synthetic\synthetic_dataset_v1.0.parquet --summary-output data\synthetic\synthetic_dataset_v1.0_summary.json
```

Evaluate a synthetic dataset:

```powershell
python src\evaluation\evaluate_synthetic.py --synthetic data\synthetic\synthetic_dataset_v1.0.parquet --output-dir outputs\reports
```

Run tests:

```powershell
pytest -q
```

## Website Workflow

1. Open **Overview** to check model status, latest dataset output, reports, and required files.
2. Open **Predict** to test one network flow.
3. Open **Use Dataset** to validate a CSV file before scoring it.
4. Open **Create Data** to choose filters and create a synthetic dataset.
5. Open **Results** to review model metrics, dataset mix, category detection, and reports.
6. Open **Why Result** to inspect the strongest features behind a single prediction.
7. Open **Data Drift** to compare a new CSV against the training split.
8. Open **About** for a simple guide and project limitations.

## Current Results

- Real-only LightGBM F1: `0.9916`
- Synthetic-only F1: `0.6789`
- Real plus synthetic F1: `0.9920`
- Average JS divergence: `0.043237`
- Exact real row matches: `0`

## Deployment

Use Docker for the public version because the app serves local model, metrics, and dataset artifacts. See `DEPLOYMENT.md` for the Docker and GitHub checklist.

## Notes

The current generator is a conditioned statistical generator with realism constraints. It is useful for testing and augmentation, but generated traffic should not replace real traffic for final model evaluation.

For deployment settings, copy `.env.example` to `.env` and set API keys, admin token, CORS origins, model hash, upload limit, rate limit, and runtime limits as needed.
