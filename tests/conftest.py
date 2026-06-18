"""
conftest.py - Shared fixtures for CyberSynth IDS test suite.

Patches all filesystem-dependent functions so tests run without
requiring real model files, processed datasets, or output directories.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#  Ensure auth is disabled for tests unless explicitly overridden 
os.environ.setdefault("CYBERSYNTH_API_KEY", "")
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("CYBERSYNTH_RATE_LIMIT", "1000")  # high limit so tests don't throttle


#  Canonical fake metrics / quality payloads 
FAKE_METRICS: dict = {
    "model": "LightGBMClassifier",
    "training_strategy": "lightgbm_baseline",
    "decision_threshold": 0.5,
    "train_rows": 50000,
    "target": "binary_label",
    "feature_columns": ["duration", "src_bytes", "dst_bytes", "protocol", "service"],
    "categorical_feature_columns": ["protocol", "service"],
    "threshold_tuning": {},
    "feature_importance": {
        "top_original": [
            {"feature": "duration", "gain": 100.0},
            {"feature": "src_bytes", "gain": 80.0},
        ]
    },
    "test": {
        "accuracy": 0.99,
        "precision": 0.98,
        "recall": 0.97,
        "f1": 0.975,
        "roc_auc": 0.995,
        "average_precision": 0.994,
        "confusion_matrix": [[9800, 50], [100, 9950]],
        "attack_category_metrics": [
            {
                "attack_category": "DDoS",
                "label": 1,
                "rows": 1000,
                "detection_rate": 0.98,
                "false_positive_rate": 0.01,
                "mean_attack_probability": 0.92,
            }
        ],
    },
    "candidate_results": [],
    "selected_candidate": "lgbm_v1",
}

FAKE_QUALITY: dict = {
    "rows_after_dedup": 48000,
    "duplicates_removed": 2000,
    "dataset_counts": {"CIC-IDS2017": 25000, "TON_IoT": 23000},
    "split_counts": {"train": 40000, "test": 8000},
}


def _make_fake_model() -> MagicMock:
    """Return a scikit-learn-compatible mock classifier."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.2, 0.8]])
    return model


def _make_fake_bundle() -> dict:
    feature_columns = FAKE_METRICS["feature_columns"]
    categorical = set(FAKE_METRICS["categorical_feature_columns"])
    numeric = [c for c in feature_columns if c not in categorical]
    return {
        "model": _make_fake_model(),
        "metrics": FAKE_METRICS,
        "quality": FAKE_QUALITY,
        "feature_columns": feature_columns,
        "categorical_columns": sorted(categorical),
        "numeric_columns": numeric,
        "decision_threshold": 0.5,
    }


@pytest.fixture(scope="session")
def client():
    """
    Provide a TestClient with all filesystem calls patched out.
    Session-scoped so the app is only imported once.
    """
    fake_bundle = _make_fake_bundle()
    feature_columns = FAKE_METRICS["feature_columns"]

    # Minimal training-split dataframe returned when pd.read_parquet is called
    _fake_train = pd.DataFrame(
        {col: [0.0, 1.0, 2.0] for col in feature_columns}
    ).assign(binary_label=[0, 1, 0], attack_category=["Normal", "DDoS", "Normal"])

    with (
        patch("src.web.app.MODEL_PATH", new=MagicMock(exists=lambda: True, stat=MagicMock(return_value=MagicMock(st_mtime=1.0, st_size=1024)))),
        patch("src.web.app.METRICS_PATH", new=MagicMock(exists=lambda: True)),
        patch("src.web.app.TRAIN_PATH", new=MagicMock(exists=lambda: True)),
        patch("src.web.app.TEST_PATH", new=MagicMock(exists=lambda: True)),
        patch("src.web.app.model_bundle", return_value=fake_bundle),
        patch("src.web.app._cached_model_bundle", return_value=fake_bundle),
        patch("src.web.app.MODEL_HASH", new=""),
        patch("src.web.app.pd.read_parquet", return_value=_fake_train),
    ):
        from src.web.app import app
        yield TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def minimal_csv_bytes() -> bytes:
    """A minimal valid CSV with the five feature columns."""
    return (
        b"duration,src_bytes,dst_bytes,protocol,service\n"
        b"0.42,1200,2400,TCP,http\n"
        b"1.10,500,800,UDP,dns\n"
    )


@pytest.fixture()
def oversized_csv_bytes() -> bytes:
    """A CSV that exceeds MAX_UPLOAD_MB (> 25 MB)."""
    header = b"duration,src_bytes\n"
    row = b"0.1,100\n"
    target = 26 * 1024 * 1024  # 26 MB
    return header + row * (target // len(row) + 1)
