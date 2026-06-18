from __future__ import annotations

import asyncio
import collections
import hashlib
import io
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any


import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from src.generation.generate_synthetic import generate_synthetic


ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web"
MODEL_PATH = ROOT / "models" / "baseline_lightgbm.joblib"
METRICS_PATH = ROOT / "outputs" / "metrics" / "baseline_lightgbm_metrics.json"
QUALITY_PATH = ROOT / "data" / "processed" / "dataset_quality_report.json"
TEST_PATH = ROOT / "data" / "processed" / "test.parquet"
TRAIN_PATH = ROOT / "data" / "processed" / "train.parquet"
SYNTHETIC_DIR = ROOT / "data" / "synthetic"
REPORTS_DIR = ROOT / "outputs" / "reports"
STARTED_AT = datetime.now(timezone.utc)
MAX_UPLOAD_MB = int(os.getenv("CYBERSYNTH_MAX_UPLOAD_MB", "25"))
APP_ENV = os.getenv("APP_ENV", "development").lower()         # set to "production" to harden
API_KEY = os.getenv("CYBERSYNTH_API_KEY", "").strip()         # empty = auth disabled (local dev)
ADMIN_TOKEN = os.getenv("CYBERSYNTH_ADMIN_TOKEN", "").strip() # guards /api/reload-model
MODEL_HASH = os.getenv("CYBERSYNTH_MODEL_SHA256", "").strip() # optional: sha256 of model file
_GENERATION_TIMEOUT_SECONDS = int(os.getenv("CYBERSYNTH_GEN_TIMEOUT", "600"))

# Audit log
LOGS_DIR = ROOT / "logs"
AUDIT_LOG_PATH = LOGS_DIR / "audit.jsonl"
_audit_lock = threading.Lock()


def _write_audit(record: dict[str, Any]) -> None:
    """Append one JSON line to the audit log (non-blocking on failure)."""
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with _audit_lock:
            with AUDIT_LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:  # never let logging crash the request
        pass

# CORS
_raw_origins = os.getenv(
    "CYBERSYNTH_ALLOWED_ORIGINS",
    "http://127.0.0.1:8000,http://localhost:8000",
)
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# Rate limiter (sliding-window, in-memory)
_RATE_LIMIT_WINDOW = 60          # seconds
_RATE_LIMIT_MAX = int(os.getenv("CYBERSYNTH_RATE_LIMIT", "30"))  # requests per window
_rate_store: dict[str, collections.deque] = {}
_rate_lock = threading.Lock()


def _check_rate_limit(client_ip: str) -> bool:
    """Return True if the client is within the allowed rate, False if throttled."""
    now = time.monotonic()
    with _rate_lock:
        if client_ip not in _rate_store:
            _rate_store[client_ip] = collections.deque()
        window = _rate_store[client_ip]
        # evict timestamps older than the window
        while window and window[0] < now - _RATE_LIMIT_WINDOW:
            window.popleft()
        if len(window) >= _RATE_LIMIT_MAX:
            return False
        window.append(now)
        return True


_ALLOWED_SYNTHETIC_SUFFIXES = {".parquet", ".json", ".csv", ".txt"}
_ALLOWED_REPORT_SUFFIXES = {".json", ".md", ".tex", ".pdf"}
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}
_generation_lock = threading.Lock()
_model_bundle_mtime: float = -1.0

# Plain-text magic bytes (first 512 bytes must not look like binary)
_BINARY_SIGNATURES = [
    b"\x50\x4b",   # ZIP / xlsx
    b"\x1f\x8b",   # gzip
    b"\x4d\x5a",   # PE / EXE
    b"\x89\x50\x4e\x47",  # PNG
    b"\xff\xd8\xff",      # JPEG
]
_CSV_CONTENT_TYPES = {
    "",
    "text/csv",
    "application/csv",
    "application/vnd.ms-excel",
    "text/plain",
}


def _looks_binary(data: bytes) -> bool:
    header = data[:512]
    for sig in _BINARY_SIGNATURES:
        if header.startswith(sig):
            return True
    # Heuristic: more than 10 % non-printable bytes -> likely binary
    non_printable = sum(1 for b in header if b < 0x09 or (0x0E <= b <= 0x1F) or b == 0x7F)
    return non_printable / max(len(header), 1) > 0.10

app = FastAPI(title="CyberSynth IDS")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "X-Admin-Token"],
)


# Security-header middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self';"
    )
    return response


# Audit-log middleware
@app.middleware("http")
async def audit_log(request: Request, call_next):
    """Record every API call to logs/audit.jsonl."""
    start = time.monotonic()
    response = await call_next(request)
    elapsed_ms = round((time.monotonic() - start) * 1000)
    path = request.url.path
    # Only log /api/ calls to keep the log clean
    if path.startswith("/api/"):
        client_ip = request.client.host if request.client else "unknown"
        _write_audit({
            "ts": datetime.now(timezone.utc).isoformat(),
            "ip": client_ip,
            "method": request.method,
            "path": path,
            "status": response.status_code,
            "ms": elapsed_ms,
        })
    return response


# API-key authentication middleware (opt-in)
_UNAUTHENTICATED_PATHS = {"/", "/favicon.ico", "/api/health"}


@app.middleware("http")
async def authenticate(request: Request, call_next):
    """Enforce API-key auth when CYBERSYNTH_API_KEY is set in the environment."""
    if not API_KEY:
        # Auth disabled - local development mode.
        return await call_next(request)
    path = request.url.path
    # Allow unauthenticated access to static assets and the health probe.
    if path in _UNAUTHENTICATED_PATHS or path.startswith("/static/"):
        return await call_next(request)
    provided = request.headers.get("X-API-Key", "")
    if provided != API_KEY:
        return JSONResponse(
            status_code=401,
            content={"detail": "Unauthorized: valid X-API-Key header required."},
        )
    return await call_next(request)


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class PredictionRequest(BaseModel):
    features: dict[str, Any]

    @field_validator("features")
    @classmethod
    def _validate_features(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(value) > 500:
            raise ValueError("features must contain at most 500 keys")
        for k, v in value.items():
            if not isinstance(k, str) or len(k) > 128:
                raise ValueError("feature key too long (max 128 chars)")
            if isinstance(v, str) and len(v) > 1024:
                raise ValueError(f"string value for '{k}' exceeds 1024 characters")
        return value


class GenerateSyntheticRequest(BaseModel):
    rows: int = 100_000
    mode: str = "proportional"
    minimum_per_category: int = 25
    noise_scale: float = 0.08
    clip_quantile: float = 0.005
    random_state: int = 42
    categories: list[str] | None = None
    subcategories: list[str] | None = None
    dataset_sources: list[str] | None = None
    environment_types: list[str] | None = None
    labels: list[int] | None = None
    drop_exact_matches: bool = True
    output_name: str = "synthetic_dataset"


def to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def safe_output_stem(value: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())[:80].strip("._-")
    return stem or "synthetic_dataset"


def _safe_filename(filename: str, allowed_suffixes: set[str]) -> str:
    name = Path(filename).name
    if not name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    stem_upper = Path(name).stem.upper()
    if stem_upper in _WINDOWS_RESERVED:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if Path(name).suffix.lower() not in allowed_suffixes:
        raise HTTPException(status_code=400, detail=f"File type not allowed. Permitted: {', '.join(sorted(allowed_suffixes))}")
    return name


def synthetic_file_path(filename: str) -> Path:
    name = _safe_filename(filename, _ALLOWED_SYNTHETIC_SUFFIXES)
    path = (SYNTHETIC_DIR / name).resolve()
    if path.parent != SYNTHETIC_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid synthetic filename")
    return path


def report_file_path(filename: str) -> Path:
    name = _safe_filename(filename, _ALLOWED_REPORT_SUFFIXES)
    path = (REPORTS_DIR / name).resolve()
    if path.parent != REPORTS_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid report filename")
    return path


def relative_path(path: Path) -> str:
    """Return a relative path string, or a placeholder in production mode."""
    if APP_ENV == "production":
        return "[hidden in production]"
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def file_metadata(path: Path, download_url: str | None = None) -> dict[str, Any]:
    stat = path.stat()
    payload: dict[str, Any] = {
        "name": path.name,
        "path": relative_path(path),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }
    if download_url:
        payload["download_url"] = download_url
    return payload


def check_file(label: str, path: Path, required: bool = True) -> dict[str, Any]:
    exists = path.exists()
    payload = {
        "label": label,
        "path": relative_path(path),
        "required": required,
        "ok": exists,
        "status": "pass" if exists else ("missing" if required else "optional"),
    }
    if exists and path.is_file():
        payload["size_bytes"] = path.stat().st_size
    return payload


def validation_summary(frame: pd.DataFrame, filename: str) -> dict[str, Any]:
    bundle = model_bundle()
    feature_columns = bundle["feature_columns"]
    numeric_columns = set(bundle["numeric_columns"])
    categorical_columns = set(bundle["categorical_columns"])
    present_features = [column for column in feature_columns if column in frame.columns]
    missing_features = [column for column in feature_columns if column not in frame.columns]
    ignored_columns = [column for column in frame.columns if column not in feature_columns]
    coverage = len(present_features) / len(feature_columns) if feature_columns else 0.0

    parse_issues = []
    total_parse_issues = 0
    for column in present_features:
        if column not in numeric_columns:
            continue
        raw = frame[column]
        parsed = pd.to_numeric(raw, errors="coerce")
        raw_text = raw.astype("string")
        invalid = parsed.isna() & raw.notna() & raw_text.str.strip().ne("")
        invalid_count = int(invalid.sum())
        if invalid_count:
            total_parse_issues += invalid_count
            parse_issues.append({"column": column, "invalid_values": invalid_count})

    preview_columns = (present_features + ignored_columns)[:8]
    preview_frame = frame[preview_columns].head(5).copy() if preview_columns else frame.head(5).copy()
    preview_frame = preview_frame.astype(object).where(pd.notna(preview_frame), None)
    preview_rows = [
        {str(key): to_builtin(value) for key, value in row.items()}
        for row in preview_frame.to_dict(orient="records")
    ]

    recommendations = []
    if missing_features:
        recommendations.append(
            f"{len(missing_features)} model features are missing; the scorer will fill missing numeric values with NaN and categorical values with blanks."
        )
    if ignored_columns:
        recommendations.append(f"{len(ignored_columns)} extra columns will be ignored during scoring.")
    if total_parse_issues:
        recommendations.append(f"{total_parse_issues} numeric cells could not be parsed and will become NaN.")
    if len(frame) > 5000:
        recommendations.append("Batch scoring uses the first 5,000 rows for browser responsiveness.")
    if not recommendations:
        recommendations.append("CSV structure is ready for scoring.")

    if not len(frame):
        readiness = "not_ready"
    elif coverage >= 0.75 and total_parse_issues == 0:
        readiness = "ready"
    elif coverage >= 0.35:
        readiness = "review"
    else:
        readiness = "not_ready"

    return {
        "filename": filename,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "feature_coverage": coverage,
        "feature_columns_expected": len(feature_columns),
        "feature_columns_present": len(present_features),
        "numeric_columns_present": len([column for column in present_features if column in numeric_columns]),
        "categorical_columns_present": len([column for column in present_features if column in categorical_columns]),
        "missing_features": missing_features[:30],
        "missing_feature_count": len(missing_features),
        "ignored_columns": ignored_columns[:30],
        "ignored_column_count": len(ignored_columns),
        "numeric_parse_issues": parse_issues[:20],
        "numeric_parse_issue_count": total_parse_issues,
        "preview_columns": preview_columns,
        "preview_rows": preview_rows,
        "readiness": readiness,
        "recommendations": recommendations,
    }


def _verify_model_hash(path: Path) -> None:
    """If CYBERSYNTH_MODEL_SHA256 is set, verify the model file hash before loading."""
    if not MODEL_HASH:
        return  # Hash verification not configured - skip.
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha.update(chunk)
    actual = sha.hexdigest()
    if actual != MODEL_HASH:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Model file integrity check failed. "
                f"Expected SHA-256 {MODEL_HASH[:16]}..., got {actual[:16]}... "
                "Do not load an untrusted model file."
            ),
        )


@lru_cache(maxsize=1)
def _cached_model_bundle() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded - run: python src/evaluation/train_baseline.py",
        )
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Metrics not found - run: python src/evaluation/train_baseline.py",
        )

    _verify_model_hash(MODEL_PATH)
    metrics = load_json(METRICS_PATH)
    quality = load_json(QUALITY_PATH)
    model = joblib.load(MODEL_PATH)
    feature_columns = metrics.get("feature_columns", [])
    categorical = set(metrics.get("categorical_feature_columns", []))
    numeric = [col for col in feature_columns if col not in categorical]

    return {
        "model": model,
        "metrics": metrics,
        "quality": quality,
        "feature_columns": feature_columns,
        "categorical_columns": sorted(categorical),
        "numeric_columns": numeric,
        "decision_threshold": float(metrics.get("decision_threshold", 0.5)),
    }


def model_bundle() -> dict[str, Any]:
    """Return the cached model bundle, auto-invalidating when the model file changes on disk."""
    global _model_bundle_mtime
    try:
        mtime = MODEL_PATH.stat().st_mtime
    except FileNotFoundError:
        mtime = -1.0
    if mtime != _model_bundle_mtime:
        _cached_model_bundle.cache_clear()
        _model_bundle_mtime = mtime
    return _cached_model_bundle()


def align_features(frame: pd.DataFrame) -> pd.DataFrame:
    bundle = model_bundle()
    feature_columns = bundle["feature_columns"]
    categorical = set(bundle["categorical_columns"])

    aligned = pd.DataFrame(index=frame.index)
    for column in feature_columns:
        if column in frame.columns:
            aligned[column] = frame[column]
        elif column in categorical:
            aligned[column] = ""
        else:
            aligned[column] = np.nan

    for column in bundle["numeric_columns"]:
        aligned[column] = pd.to_numeric(aligned[column], errors="coerce")
    for column in categorical:
        aligned[column] = aligned[column].astype("string").fillna("")
    return aligned[feature_columns]


def predict_frame(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    bundle = model_bundle()
    aligned = align_features(frame)
    model = bundle["model"]
    # Pass as a named DataFrame so LightGBM receives correct feature names
    # and does not emit the 'X does not have valid feature names' warning.
    probabilities = model.predict_proba(aligned[bundle["feature_columns"]])[:, 1]
    predictions = (probabilities >= bundle["decision_threshold"]).astype(int)
    return predictions, probabilities


def risk_level(probability: float) -> str:
    if probability >= 0.85:
        return "Critical"
    if probability >= 0.65:
        return "High"
    if probability >= 0.35:
        return "Review"
    return "Low"


def verdict_text(prediction: int) -> str:
    return "Attack" if int(prediction) == 1 else "Normal"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/health")
def health() -> dict[str, Any]:
    checks = [
        check_file("Model artifact", MODEL_PATH),
        check_file("Model metrics", METRICS_PATH),
        check_file("Training split", TRAIN_PATH),
        check_file("Test split", TEST_PATH),
        check_file("Dataset quality report", QUALITY_PATH, required=False),
        check_file("Evaluation report", REPORTS_DIR / "evaluation_report.md", required=False),
    ]
    required_ok = all(item["ok"] for item in checks if item["required"])
    optional_ready = sum(1 for item in checks if item["ok"] and not item["required"])
    uptime = datetime.now(timezone.utc) - STARTED_AT

    bundle = model_bundle() if required_ok else None
    return {
        "status": "ready" if required_ok else "degraded",
        "started_at": STARTED_AT.isoformat(),
        "uptime_seconds": int(uptime.total_seconds()),
        "checks": checks,
        "optional_ready": optional_ready,
        "model": {
            "name": bundle["metrics"].get("model", "LightGBMClassifier") if bundle else None,
            "decision_threshold": bundle["decision_threshold"] if bundle else None,
            "features": len(bundle["feature_columns"]) if bundle else 0,
        },
        "limits": {
            "csv_upload_mb": MAX_UPLOAD_MB,
            "csv_scoring_rows": 5000,
            "synthetic_generation_rows": 1_000_000,
        },
    }


@app.get("/api/artifacts")
def artifacts() -> dict[str, Any]:
    synthetic_files = []
    if SYNTHETIC_DIR.exists():
        for path in sorted(SYNTHETIC_DIR.glob("*"), key=lambda item: item.stat().st_mtime, reverse=True):
            if not path.is_file() or path.suffix.lower() not in {".parquet", ".json", ".csv", ".txt"}:
                continue
            metadata = file_metadata(path, f"/api/download-synthetic/{path.name}")
            if path.suffix.lower() == ".parquet":
                summary = load_json(SYNTHETIC_DIR / f"{path.stem}_summary.json")
                if summary:
                    metadata["rows"] = summary.get("rows_generated")
                    metadata["mode"] = summary.get("mode")
                    metadata["quality"] = summary.get("quality", {})
            synthetic_files.append(metadata)

    report_files = []
    if REPORTS_DIR.exists():
        for path in sorted(REPORTS_DIR.glob("*"), key=lambda item: item.stat().st_mtime, reverse=True):
            if not path.is_file() or path.suffix.lower() not in {".json", ".md", ".tex", ".pdf"}:
                continue
            report_files.append(file_metadata(path, f"/api/download-report/{path.name}"))

    return {
        "synthetic": synthetic_files[:20],
        "reports": report_files[:10],
        "paths": {
            "synthetic_dir": relative_path(SYNTHETIC_DIR),
            "reports_dir": relative_path(REPORTS_DIR),
        },
    }


@app.get("/api/status")
def status() -> dict[str, Any]:
    bundle = model_bundle()
    metrics = bundle["metrics"]
    quality = bundle["quality"]
    test_metrics = metrics.get("test", {})

    try:
        model_mtime = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime, timezone.utc).isoformat()
    except FileNotFoundError:
        model_mtime = None

    return {
        "model": metrics.get("model", "LightGBMClassifier"),
        "model_modified_at": model_mtime,
        "target": metrics.get("target", "binary_label"),
        "features": {
            "total": len(bundle["feature_columns"]),
            "numeric": len(bundle["numeric_columns"]),
            "categorical": len(bundle["categorical_columns"]),
            "columns": bundle["feature_columns"],
        },
        "training": {
            "strategy": metrics.get("training_strategy", "lightgbm_baseline"),
            "selected_candidate": metrics.get("selected_candidate"),
            "decision_threshold": metrics.get("decision_threshold", 0.5),
            "threshold_tuning": metrics.get("threshold_tuning", {}),
            "feature_importance": metrics.get("feature_importance", {}),
        },
        "test": {
            "accuracy": test_metrics.get("accuracy"),
            "precision": test_metrics.get("precision"),
            "recall": test_metrics.get("recall"),
            "f1": test_metrics.get("f1"),
            "roc_auc": test_metrics.get("roc_auc"),
            "average_precision": test_metrics.get("average_precision"),
            "confusion_matrix": test_metrics.get("confusion_matrix"),
            "attack_category_metrics": test_metrics.get("attack_category_metrics", []),
        },
        "data": {
            "train_rows": metrics.get("train_rows"),
            "split_counts": quality.get("split_counts", {}),
            "rows_after_dedup": quality.get("rows_after_dedup"),
            "duplicates_removed": quality.get("duplicates_removed"),
            "dataset_counts": quality.get("dataset_counts", {}),
        },
    }


@app.get("/api/sample")
def sample(kind: str = "attack") -> dict[str, Any]:
    if not TEST_PATH.exists():
        raise HTTPException(status_code=404, detail="Missing test split")
    df = pd.read_parquet(TEST_PATH)
    desired = 1 if kind.lower() == "attack" else 0
    subset = df[df["label"].astype(int).eq(desired)]
    if subset.empty:
        subset = df
    candidates = subset.head(2000).copy()
    _, probabilities = predict_frame(candidates)
    pick = int(np.argmax(probabilities) if desired else np.argmin(probabilities))
    row = candidates.iloc[pick]
    bundle = model_bundle()
    features = {
        col: to_builtin(row[col])
        for col in bundle["feature_columns"]
        if col in row.index and pd.notna(row[col])
    }
    return {
        "kind": "attack" if desired else "normal",
        "features": features,
        "truth": {
            "label": int(row.get("label", desired)),
            "attack_category": str(row.get("attack_category", "")),
            "dataset_source": str(row.get("dataset_source", "")),
        },
    }


@app.get("/api/generator-options")
def generator_options() -> dict[str, Any]:
    if not TRAIN_PATH.exists():
        raise HTTPException(status_code=404, detail="Missing training split")
    df = pd.read_parquet(
        TRAIN_PATH,
        columns=["attack_category", "attack_subcategory", "label", "dataset_source", "environment_type"],
    )
    def counts_payload(column: str) -> list[dict[str, Any]]:
        values = df[column].astype("string").fillna("Unknown").value_counts()
        return [{"name": str(name), "rows": int(count)} for name, count in values.items()]

    categories = []
    for category, count in df["attack_category"].value_counts().items():
        subset = df[df["attack_category"].eq(category)]
        categories.append(
            {
                "name": str(category),
                "rows": int(count),
                "label": int(subset["label"].mode().iloc[0]) if not subset.empty else 1,
                "top_subcategories": subset["attack_subcategory"].value_counts().head(5).index.astype(str).tolist(),
            }
        )
    return {
        "input": relative_path(TRAIN_PATH),
        "total_rows": int(len(df)),
        "categories": categories,
        "dataset_sources": counts_payload("dataset_source"),
        "environment_types": counts_payload("environment_type"),
        "subcategories": counts_payload("attack_subcategory"),
        "labels": [
            {"name": "Normal", "value": 0, "rows": int((df["label"].astype(int) == 0).sum())},
            {"name": "Attack", "value": 1, "rows": int((df["label"].astype(int) == 1).sum())},
        ],
        "defaults": {
            "rows": 100_000,
            "mode": "proportional",
            "minimum_per_category": 25,
            "noise_scale": 0.08,
            "clip_quantile": 0.005,
            "random_state": 42,
            "output_name": "synthetic_dataset",
        },
    }


@app.post("/api/generate-synthetic")
async def generate_synthetic_endpoint(request: Request, payload: GenerateSyntheticRequest) -> dict[str, Any]:
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded - max {_RATE_LIMIT_MAX} requests per {_RATE_LIMIT_WINDOW}s.",
        )
    if not TRAIN_PATH.exists():
        raise HTTPException(status_code=404, detail="Missing training split")
    if payload.mode not in {"proportional", "balanced", "label_balanced"}:
        raise HTTPException(status_code=400, detail="mode must be proportional, balanced, or label_balanced")
    if payload.rows < 100 or payload.rows > 1_000_000:
        raise HTTPException(status_code=400, detail="rows must be between 100 and 1,000,000")
    if payload.minimum_per_category < 0 or payload.minimum_per_category > payload.rows:
        raise HTTPException(status_code=400, detail="minimum_per_category is out of range")
    if payload.noise_scale < 0 or payload.noise_scale > 0.5:
        raise HTTPException(status_code=400, detail="noise_scale must be between 0 and 0.5")
    if payload.clip_quantile < 0 or payload.clip_quantile > 0.1:
        raise HTTPException(status_code=400, detail="clip_quantile must be between 0 and 0.1")

    options = generator_options()
    allowed_categories = {item["name"] for item in options["categories"]}
    requested_categories = set(payload.categories or [])
    unknown = sorted(requested_categories - allowed_categories)
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown categories: {', '.join(unknown)}")
    allowed_subcategories = {item["name"] for item in options["subcategories"]}
    requested_subcategories = set(payload.subcategories or [])
    unknown = sorted(requested_subcategories - allowed_subcategories)
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown subcategories: {', '.join(unknown)}")
    allowed_sources = {item["name"] for item in options["dataset_sources"]}
    requested_sources = set(payload.dataset_sources or [])
    unknown = sorted(requested_sources - allowed_sources)
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown dataset sources: {', '.join(unknown)}")
    allowed_environments = {item["name"] for item in options["environment_types"]}
    requested_environments = set(payload.environment_types or [])
    unknown = sorted(requested_environments - allowed_environments)
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown environment types: {', '.join(unknown)}")
    requested_labels = set(payload.labels or [])
    unknown_labels = sorted(requested_labels - {0, 1})
    if unknown_labels:
        raise HTTPException(status_code=400, detail="labels must be 0 and/or 1")

    if not _generation_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="A generation job is already running. Please wait for it to complete.",
        )

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    stem = safe_output_stem(payload.output_name)
    output_path = SYNTHETIC_DIR / f"{stem}.parquet"
    summary_path = SYNTHETIC_DIR / f"{stem}_summary.json"

    def _run() -> dict:
        try:
            return generate_synthetic(
                input_path=TRAIN_PATH,
                output_path=output_path,
                summary_path=summary_path,
                rows=payload.rows,
                mode=payload.mode,
                minimum_per_category=payload.minimum_per_category,
                noise_scale=payload.noise_scale,
                random_state=payload.random_state,
                categories=requested_categories or None,
                subcategories=requested_subcategories or None,
                dataset_sources=requested_sources or None,
                environment_types=requested_environments or None,
                labels=requested_labels or None,
                clip_quantile=payload.clip_quantile,
                drop_exact_matches=payload.drop_exact_matches,
            )
        except SystemExit as exc:
            raise RuntimeError(str(exc)) from exc

    try:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, _run)
        summary = await asyncio.wait_for(future, timeout=_GENERATION_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,
            detail=f"Generation timed out after {_GENERATION_TIMEOUT_SECONDS}s.",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        _generation_lock.release()

    return {
        "summary": summary,
        "output_file": output_path.name,
        "summary_file": summary_path.name,
        "download_url": f"/api/download-synthetic/{output_path.name}",
        "summary_url": f"/api/download-synthetic/{summary_path.name}",
    }


@app.get("/api/download-synthetic/{filename}")
def download_synthetic(filename: str) -> FileResponse:
    path = synthetic_file_path(filename)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Synthetic file not found")
    return FileResponse(path, filename=path.name)


@app.get("/api/download-report/{filename}")
def download_report(filename: str) -> FileResponse:
    path = report_file_path(filename)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Report file not found")
    return FileResponse(path, filename=path.name)


@app.post("/api/predict")
def predict(request: Request, payload: PredictionRequest) -> dict[str, Any]:
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded - max {_RATE_LIMIT_MAX} requests per {_RATE_LIMIT_WINDOW}s.",
        )
    frame = pd.DataFrame([payload.features])
    predictions, probabilities = predict_frame(frame)
    probability = float(probabilities[0])
    prediction = int(predictions[0])
    return {
        "prediction": prediction,
        "verdict": verdict_text(prediction),
        "attack_probability": probability,
        "normal_probability": 1.0 - probability,
        "risk": risk_level(probability),
    }


def _read_and_validate_csv_upload(content: bytes, filename: str) -> pd.DataFrame:
    """Shared CSV validation: size, magic bytes, and pandas parse."""
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"CSV limit is {MAX_UPLOAD_MB} MB")
    if _looks_binary(content):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file does not appear to be plain-text CSV (binary signature detected).",
        )
    try:
        frame = pd.read_csv(io.BytesIO(content), low_memory=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}") from exc
    if frame.empty:
        raise HTTPException(status_code=400, detail="CSV has no rows")
    return frame


def _validate_csv_upload_metadata(file: UploadFile) -> str:
    """Validate upload filename and browser-provided content type."""
    filename = file.filename or "upload.csv"
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file")

    content_type = (file.content_type or "").split(";", 1)[0].strip().lower()
    if content_type not in _CSV_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Upload content type must be CSV text")
    return filename


@app.post("/api/validate-csv")
async def validate_csv(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded - max {_RATE_LIMIT_MAX} requests per {_RATE_LIMIT_WINDOW}s.",
        )
    filename = _validate_csv_upload_metadata(file)
    content = await file.read()
    frame = _read_and_validate_csv_upload(content, filename)
    return validation_summary(frame, filename)


@app.post("/api/predict-csv")
async def predict_csv(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded - max {_RATE_LIMIT_MAX} requests per {_RATE_LIMIT_WINDOW}s.",
        )
    filename = _validate_csv_upload_metadata(file)
    content = await file.read()
    frame = _read_and_validate_csv_upload(content, filename)

    validation = validation_summary(frame, filename)
    max_rows = 5000
    limited = frame.head(max_rows).copy()
    predictions, probabilities = predict_frame(limited)

    results = []
    for idx, (pred, prob) in enumerate(zip(predictions, probabilities, strict=False)):
        results.append(
            {
                "row": idx + 1,
                "verdict": verdict_text(int(pred)),
                "attack_probability": float(prob),
                "risk": risk_level(float(prob)),
            }
        )

    attack_count = int(np.sum(predictions == 1))
    normal_count = int(np.sum(predictions == 0))
    return {
        "filename": filename,
        "rows_received": int(len(frame)),
        "rows_scored": int(len(limited)),
        "truncated": len(frame) > max_rows,
        "attack_count": attack_count,
        "normal_count": normal_count,
        "mean_attack_probability": float(np.mean(probabilities)),
        "validation": validation,
        "results": results[:100],
    }


@app.post("/api/reload-model")
def reload_model(request: Request) -> dict[str, Any]:
    """Clear the model cache so the next request reloads from disk.

    When CYBERSYNTH_ADMIN_TOKEN is set, callers must supply the token
    via the X-Admin-Token request header.
    """
    if ADMIN_TOKEN:
        provided = request.headers.get("X-Admin-Token", "")
        if provided != ADMIN_TOKEN:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: valid X-Admin-Token header required to reload the model.",
            )
    global _model_bundle_mtime
    _cached_model_bundle.cache_clear()
    _model_bundle_mtime = -1.0
    return {
        "status": "reloaded",
        "message": "Model cache cleared - next request will reload from disk.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/generation-status")
def generation_status() -> dict[str, Any]:
    """Check whether a synthetic generation job is currently running."""
    busy = _generation_lock.locked()
    return {"busy": busy, "status": "generating" if busy else "idle"}


@app.get("/api/experiment-card")
def experiment_card() -> dict[str, Any]:
    """Export a full reproducibility card for this model run."""
    bundle = model_bundle()
    metrics = bundle["metrics"]
    quality = bundle["quality"]

    try:
        model_mtime = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime, timezone.utc).isoformat()
    except FileNotFoundError:
        model_mtime = None

    return {
        "project": "CyberSynth IDS",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "algorithm": metrics.get("model", "LightGBMClassifier"),
            "training_strategy": metrics.get("training_strategy"),
            "decision_threshold": float(metrics.get("decision_threshold", 0.5)),
            "selected_candidate": metrics.get("selected_candidate"),
            "feature_count": len(bundle["feature_columns"]),
            "numeric_features": len(bundle["numeric_columns"]),
            "categorical_features": len(bundle["categorical_columns"]),
            "train_rows": metrics.get("train_rows"),
            "model_file": MODEL_PATH.name,
            "model_modified_at": model_mtime,
        },
        "performance": {
            "test_f1": metrics.get("test", {}).get("f1"),
            "test_recall": metrics.get("test", {}).get("recall"),
            "test_precision": metrics.get("test", {}).get("precision"),
            "test_roc_auc": metrics.get("test", {}).get("roc_auc"),
            "test_average_precision": metrics.get("test", {}).get("average_precision"),
            "test_accuracy": metrics.get("test", {}).get("accuracy"),
            "threshold_tuning": metrics.get("threshold_tuning", {}),
            "candidate_results": metrics.get("candidate_results", []),
        },
        "data": {
            "datasets_used": sorted(quality.get("dataset_counts", {}).keys()),
            "dataset_counts": quality.get("dataset_counts", {}),
            "train_rows": metrics.get("train_rows"),
            "split_counts": quality.get("split_counts", {}),
            "rows_after_dedup": quality.get("rows_after_dedup"),
            "duplicates_removed": quality.get("duplicates_removed"),
        },
        "feature_importance_top10": (metrics.get("feature_importance", {}).get("top_original", []))[:10],
        "feature_columns": bundle["feature_columns"],
        "numeric_feature_columns": bundle["numeric_columns"],
        "categorical_feature_columns": bundle["categorical_columns"],
    }


@app.post("/api/explain")
def explain(request: Request, payload: PredictionRequest) -> dict[str, Any]:
    """SHAP feature attributions for a single prediction.

    Handles sklearn Pipeline models by extracting preprocessor + raw classifier.
    Requires: pip install shap
    """
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded - max {_RATE_LIMIT_MAX} requests per {_RATE_LIMIT_WINDOW}s.",
        )

    try:
        import shap
    except ImportError:
        raise HTTPException(status_code=501, detail="SHAP not installed - run: pip install shap")

    from sklearn.pipeline import Pipeline as SkPipeline

    bundle = model_bundle()
    frame = pd.DataFrame([payload.features])
    aligned = align_features(frame)
    model = bundle["model"]

    #  Unwrap Pipeline -> (preprocessor, raw_classifier) 
    if isinstance(model, SkPipeline):
        step_items = list(model.named_steps.items())
        if len(step_items) >= 2:
            preprocessor = SkPipeline(steps=step_items[:-1])
            classifier = step_items[-1][1]
        else:
            preprocessor = None
            classifier = step_items[0][1]
    else:
        preprocessor = None
        classifier = model

    try:
        if preprocessor is not None:
            transformed = preprocessor.transform(aligned[bundle["feature_columns"]])
            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()
            try:
                ct = model.named_steps.get("preprocess") or model.named_steps.get("preprocessor")
                feat_names = list(ct.get_feature_names_out()) if ct else []
            except Exception:
                feat_names = []
            if len(feat_names) != transformed.shape[1]:
                feat_names = [f"feature_{i}" for i in range(transformed.shape[1])]
            transformed_df = pd.DataFrame(transformed, columns=feat_names)
        else:
            transformed_df = aligned[bundle["feature_columns"]]
            feat_names = list(bundle["feature_columns"])

        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(transformed_df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SHAP failed: {exc}") from exc

    # LightGBM binary: shap_values is [class-0, class-1]
    if isinstance(shap_values, list) and len(shap_values) == 2:
        attack_shap = shap_values[1][0]
        base_value = float(explainer.expected_value[1])
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        attack_shap = shap_values[0]
        ev = explainer.expected_value
        base_value = float(ev if not hasattr(ev, "__len__") else ev[0])
    else:
        attack_shap = np.asarray(shap_values).ravel()
        ev = explainer.expected_value
        base_value = float(ev if not hasattr(ev, "__len__") else ev[0])

    # Clean OHE names: "categorical__protocol_TCP" -> "protocol = TCP"
    def _clean(name: str) -> str:
        name = str(name)
        for pfx in ("numeric__", "categorical__", "remainder__"):
            if name.startswith(pfx):
                name = name[len(pfx):]
                break
        if "_" in name:
            col, val = name.rsplit("_", 1)
            if len(val) <= 20 and val.replace("-", "").replace(".", "").isalnum():
                return f"{col} = {val}"
        return name

    attributions = sorted(
        [{"feature": _clean(feat_names[i]), "shap_value": float(attack_shap[i])}
         for i in range(min(len(feat_names), len(attack_shap)))],
        key=lambda x: abs(x["shap_value"]), reverse=True,
    )

    predictions, probabilities = predict_frame(frame)
    probability = float(probabilities[0])
    return {
        "prediction": int(predictions[0]),
        "verdict": verdict_text(int(predictions[0])),
        "attack_probability": probability,
        "risk": risk_level(probability),
        "explanation": {
            "base_value": base_value,
            "top_attack_drivers": [a for a in attributions if a["shap_value"] > 0][:8],
            "top_normal_drivers": [a for a in attributions if a["shap_value"] < 0][:8],
            "all_attributions": attributions[:20],
        },
    }


def _read_audit_entries(limit: int) -> list[dict]:
    entries: list[dict] = []
    if AUDIT_LOG_PATH.exists():
        try:
            for raw in reversed(AUDIT_LOG_PATH.read_text(encoding="utf-8").splitlines()):
                raw = raw.strip()
                if raw:
                    try:
                        entries.append(json.loads(raw))
                    except json.JSONDecodeError:
                        pass
                if len(entries) >= limit:
                    break
        except OSError:
            pass
    return entries


@app.get("/api/audit-log")
def audit_log_view(limit: int = 100) -> dict[str, Any]:
    """Recent audit entries (newest first). Protected when ADMIN_TOKEN is set."""
    if ADMIN_TOKEN:
        raise HTTPException(
            status_code=403,
            detail="Audit log is admin-protected. Use /api/audit-log-admin with X-Admin-Token.",
        )
    limit = max(1, min(limit, 500))
    entries = _read_audit_entries(limit)
    return {"count": len(entries), "limit": limit, "log_path": relative_path(AUDIT_LOG_PATH), "entries": entries}


@app.get("/api/audit-log-admin")
def audit_log_admin(request: Request, limit: int = 200) -> dict[str, Any]:
    """Admin-only audit log - requires X-Admin-Token header."""
    if ADMIN_TOKEN:
        if request.headers.get("X-Admin-Token", "") != ADMIN_TOKEN:
            raise HTTPException(status_code=401, detail="Unauthorized: valid X-Admin-Token required.")
    limit = max(1, min(limit, 1000))
    entries = _read_audit_entries(limit)
    return {"count": len(entries), "limit": limit, "log_path": relative_path(AUDIT_LOG_PATH), "entries": entries}


def _psi(expected: pd.Series, actual: pd.Series, n_bins: int = 10) -> float:
    """Compute Population Stability Index between two numeric distributions."""
    combined = pd.concat([expected, actual], ignore_index=True).dropna()
    if combined.empty or combined.std() == 0:
        return 0.0
    bins = pd.cut(combined, bins=n_bins, duplicates="drop").cat.categories
    def _bucket(series: pd.Series) -> pd.Series:
        return pd.cut(series, bins=bins, include_lowest=True).value_counts(normalize=True).reindex(bins, fill_value=0.0)
    exp_pct = _bucket(expected)
    act_pct = _bucket(actual)
    eps = 1e-6
    psi_val = float(((act_pct + eps) / (exp_pct + eps) - 1).sub(
        np.log((act_pct + eps) / (exp_pct + eps))
    ).mul(exp_pct + eps).sum())
    return round(max(0.0, psi_val), 6)


@app.post("/api/drift")
async def drift_detection(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    """Compute PSI drift score per feature between training data and an uploaded CSV.

    Returns:
    - Per-feature PSI (Population Stability Index)
    - Drift severity: stable (<0.1), moderate (0.1-0.25), high (>0.25)
    - Overall drift summary
    """
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded - max {_RATE_LIMIT_MAX} requests per {_RATE_LIMIT_WINDOW}s.",
        )
    filename = file.filename or "upload.csv"
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file")

    content = await file.read()
    uploaded_frame = _read_and_validate_csv_upload(content, filename)

    if not TRAIN_PATH.exists():
        raise HTTPException(status_code=404, detail="Missing training split - cannot compute drift")

    bundle = model_bundle()
    numeric_features = bundle["numeric_columns"]

    # Load only the numeric model features from the training split.
    train_frame = pd.read_parquet(TRAIN_PATH, columns=list(numeric_features))

    results = []
    drifted_count = 0

    for feature in numeric_features:
        if feature not in train_frame.columns or feature not in uploaded_frame.columns:
            results.append({
                "feature": feature,
                "psi": None,
                "severity": "missing",
                "in_upload": feature in uploaded_frame.columns,
                "in_training": feature in train_frame.columns,
            })
            continue

        train_series = pd.to_numeric(train_frame[feature], errors="coerce").dropna()
        upload_series = pd.to_numeric(uploaded_frame[feature], errors="coerce").dropna()

        if train_series.empty or upload_series.empty:
            results.append({"feature": feature, "psi": None, "severity": "empty"})
            continue

        psi_val = _psi(train_series, upload_series)
        if psi_val < 0.1:
            severity = "stable"
        elif psi_val < 0.25:
            severity = "moderate"
            drifted_count += 1
        else:
            severity = "high"
            drifted_count += 1

        results.append({"feature": feature, "psi": psi_val, "severity": severity})

    results.sort(key=lambda x: (x.get("psi") or -1), reverse=True)
    high_drift = [r for r in results if r.get("severity") == "high"]
    moderate_drift = [r for r in results if r.get("severity") == "moderate"]
    drift_pct = drifted_count / max(len(results), 1)

    return {
        "filename": filename,
        "features_analysed": len([r for r in results if r.get("psi") is not None]),
        "drifted_features": drifted_count,
        "drift_fraction": round(drift_pct, 4),
        "overall_severity": "high" if drift_pct > 0.3 else "moderate" if drift_pct > 0.1 else "stable",
        "retraining_recommended": drift_pct > 0.2,
        "high_drift_features": high_drift[:10],
        "moderate_drift_features": moderate_drift[:10],
        "all_features": results,
    }
