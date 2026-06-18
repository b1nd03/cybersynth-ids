# Deployment

## Local Run

```powershell
python -m uvicorn src.web.app:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## Docker Run

Docker is the recommended public deployment path because the application depends on local model artifacts, Parquet files, native Python packages, and background generation work.

```powershell
docker build -t cybersynth-ids .
docker run --rm -p 8000:8000 cybersynth-ids
```

## Runtime Files

The dashboard expects these local files when all features are enabled:

- `models/baseline_lightgbm.joblib`
- `outputs/metrics/baseline_lightgbm_metrics.json`
- `data/processed/train.parquet`
- `data/processed/test.parquet`
- `web/index.html`
- `web/app.js`
- `web/styles.css`

The API can start without every artifact, but model prediction, drift checks, and report links depend on the files above.

## Environment

```text
CYBERSYNTH_MAX_UPLOAD_MB=25
CYBERSYNTH_RATE_LIMIT=30
CYBERSYNTH_API_KEY=
CYBERSYNTH_ADMIN_TOKEN=
CYBERSYNTH_MODEL_SHA256=
CYBERSYNTH_GEN_TIMEOUT=600
CYBERSYNTH_ALLOWED_ORIGINS=http://127.0.0.1:8000,http://localhost:8000
APP_ENV=development
```

Copy `.env.example` to `.env` for the full list of supported runtime settings.

For a public deployment:

1. Set `APP_ENV=production`.
2. Restrict `CYBERSYNTH_ALLOWED_ORIGINS`.
3. Set non-empty API and admin tokens.
4. Set `CYBERSYNTH_MODEL_SHA256` after training the model.
5. Keep raw datasets, generated datasets, logs, and private notes out of the repository.
6. Run `pytest -q` before publishing.

## GitHub Checklist

- Commit source code, configuration templates, docs, Docker files, tests, and the minimal runtime artifacts needed for the demo.
- Keep the included model and small train/test splits only if you want the Docker image to run immediately after cloning.
- Do not commit raw datasets, generated datasets, full unified datasets, logs, cache folders, local notebooks, or private notes.
- Recreate local artifacts by running the preprocessing, training, generation, and evaluation commands from `README.md`.
- Use GitHub Releases or external storage for larger artifacts.
- Keep the public repository focused on the runnable Docker demo and reproducible pipeline code.
