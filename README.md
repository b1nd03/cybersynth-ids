# CyberSynth IDS

CyberSynth IDS is a local web tool for cybersecurity dataset experiments. It prepares network-flow datasets, trains a baseline intrusion detection model, generates filtered synthetic rows, validates uploaded CSV files, and shows model results through a FastAPI dashboard.

## Highlights

- FastAPI dashboard for IDS prediction, CSV upload, synthetic dataset creation, results, explainability, and drift checks.
- LightGBM baseline model with saved metrics, feature importance, threshold tuning, and per-category reporting.
- Dataset generator with filters for label, attack family, source dataset, environment, and subcategory.
- Upload validation for row count, feature coverage, missing columns, and preview rows before scoring.
- Docker-ready local release with optional API-key auth, admin reload token, rate limiting, audit logs, CORS allowlist, model hash verification, and security headers.

## How It Works

CyberSynth IDS normalizes supported network-flow datasets into one shared schema, trains the model on traffic features, and keeps metadata for reports, filters, and dataset generation.

Uploaded CSV files are validated against the trained feature set before scoring. New datasets can be added through preprocessing, splitting, retraining, and drift checking.

The synthetic generator uses the training profile to create filtered experiment data while reducing exact real-row matches.

## Project Layout

```text
configs/              Dataset and experiment configuration
docs/                 Project notes for model behavior
src/ingestion/         Preprocessing and train/test split scripts
src/evaluation/        Model training and synthetic-data evaluation scripts
src/generation/        Synthetic dataset generator
src/web/               FastAPI application
web/                   Dashboard HTML, CSS, and JavaScript
models/                Trained model artifact for the Docker demo
outputs/               Metrics and report outputs
```

Raw downloads, generated synthetic datasets, logs, notebooks, cache folders, and private notes are ignored by Git. The public repository keeps only the small runtime artifacts needed for the Docker demo: the trained model, train/test splits, dataset quality report, metrics, and evaluation reports.

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
5. Open **Results**, **Why Result**, or **Data Drift** when deeper inspection is needed.

## Results Snapshot

- Baseline LightGBM F1: `0.9916`
- Real plus synthetic F1: `0.9920`
- Generated-data exact real row matches: `0`

## Deployment

Use Docker for the public version because the app serves local model, metrics, and dataset artifacts. See `DEPLOYMENT.md` for the Docker and GitHub checklist.

For deployment settings, copy `.env.example` to `.env` and set API keys, admin token, CORS origins, model hash, upload size, rate limits, and runtime limits as needed.
