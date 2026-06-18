"""
Evaluate synthetic cybersecurity data against real held-out IDS data.

This script produces:
    - outputs/reports/evaluation_report.json
    - outputs/reports/evaluation_report.md
    - outputs/reports/evaluation_report.tex

It implements practical fidelity, utility, and privacy checks while staying
fast enough to run on the local coursework machine.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.train_baseline import build_preprocessor, feature_columns, tune_threshold

META_COLUMNS = {
    "dataset_source",
    "source_file",
    "environment_type",
    "attack_category",
    "attack_subcategory",
    "label",
}


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def align_to_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    aligned = pd.DataFrame(index=frame.index)
    for column in columns:
        aligned[column] = frame[column] if column in frame.columns else np.nan
    return aligned[columns]


def sample_frame(frame: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.copy()
    return frame.sample(max_rows, random_state=random_state).reset_index(drop=True)


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [
        col
        for col in frame.columns
        if col not in META_COLUMNS and pd.api.types.is_numeric_dtype(frame[col])
    ]


def categorical_columns(frame: pd.DataFrame) -> list[str]:
    return [
        col
        for col in frame.columns
        if col not in META_COLUMNS and not pd.api.types.is_numeric_dtype(frame[col])
    ]


def js_divergence(real_values: pd.Series, synthetic_values: pd.Series, bins: int) -> float:
    real = pd.to_numeric(real_values, errors="coerce").dropna()
    synth = pd.to_numeric(synthetic_values, errors="coerce").dropna()
    if real.empty or synth.empty:
        return 0.0
    low = float(min(real.min(), synth.min()))
    high = float(max(real.max(), synth.max()))
    if high <= low:
        return 0.0
    real_hist, edges = np.histogram(real, bins=bins, range=(low, high), density=False)
    synth_hist, _ = np.histogram(synth, bins=edges, density=False)
    real_prob = real_hist.astype(float) + 1e-12
    synth_prob = synth_hist.astype(float) + 1e-12
    real_prob /= real_prob.sum()
    synth_prob /= synth_prob.sum()
    return float(jensenshannon(real_prob, synth_prob, base=2.0) ** 2)


def fidelity_metrics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    max_rows: int,
    bins: int,
    random_state: int,
) -> dict[str, Any]:
    real_sample = sample_frame(real, max_rows, random_state)
    synth_sample = sample_frame(synthetic, max_rows, random_state + 1)
    num_cols = [col for col in numeric_columns(real_sample) if col in synth_sample.columns]

    feature_rows = []
    for col in num_cols:
        real_col = pd.to_numeric(real_sample[col], errors="coerce").dropna()
        synth_col = pd.to_numeric(synth_sample[col], errors="coerce").dropna()
        if real_col.empty or synth_col.empty:
            continue
        ks = ks_2samp(real_col, synth_col)
        feature_rows.append(
            {
                "feature": col,
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "js_divergence": js_divergence(real_col, synth_col, bins),
                "wasserstein_distance": float(wasserstein_distance(real_col, synth_col)),
            }
        )

    if num_cols:
        real_corr = real_sample[num_cols].corr(method="pearson").fillna(0.0)
        synth_corr = synth_sample[num_cols].corr(method="pearson").fillna(0.0)
        delta = real_corr - synth_corr
        real_norm = float(np.linalg.norm(real_corr.to_numpy(), ord="fro")) or 1.0
        corr_delta = float(np.linalg.norm(delta.to_numpy(), ord="fro"))
        corr_preservation = max(0.0, 1.0 - corr_delta / real_norm)
    else:
        corr_delta = 0.0
        corr_preservation = 1.0

    avg_js = float(np.mean([row["js_divergence"] for row in feature_rows])) if feature_rows else 0.0
    ks_fail_rate = (
        float(np.mean([row["ks_pvalue"] < 0.05 for row in feature_rows])) if feature_rows else 0.0
    )
    return {
        "rows_real_sampled": int(len(real_sample)),
        "rows_synthetic_sampled": int(len(synth_sample)),
        "numeric_features_compared": len(feature_rows),
        "average_js_divergence": avg_js,
        "ks_failure_rate_p_lt_0_05": ks_fail_rate,
        "correlation_frobenius_delta": corr_delta,
        "correlation_preservation": corr_preservation,
        "per_feature": sorted(feature_rows, key=lambda item: item["js_divergence"], reverse=True),
    }


def train_ids_model(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    numeric: list[str],
    categorical: list[str],
    random_state: int,
) -> tuple[Pipeline, float]:
    preprocessor = build_preprocessor(numeric, categorical)
    x_train = preprocessor.fit_transform(train[features])
    x_val = preprocessor.transform(val[features])
    y_train = train["label"].astype(int)
    y_val = val["label"].astype(int)

    classifier = LGBMClassifier(
        objective="binary",
        n_estimators=700,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=60,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    classifier.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[early_stopping(40, first_metric_only=True, verbose=False), log_evaluation(period=0)],
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )
    val_score = model.predict_proba(val[features])[:, 1]
    threshold = tune_threshold(y_val, val_score)["threshold"]
    return model, threshold


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    metric_name: str,
    rounds: int,
    random_state: int,
) -> dict[str, float]:
    rng = np.random.default_rng(random_state)
    values = []
    n = len(y_true)
    for _ in range(rounds):
        idx = rng.integers(0, n, size=n)
        true = y_true[idx]
        pred = y_pred[idx]
        score = y_score[idx]
        if metric_name == "f1":
            values.append(f1_score(true, pred, zero_division=0))
        elif metric_name == "precision":
            values.append(precision_score(true, pred, zero_division=0))
        elif metric_name == "recall":
            values.append(recall_score(true, pred, zero_division=0))
        elif metric_name == "roc_auc" and len(np.unique(true)) == 2:
            values.append(roc_auc_score(true, score))
    if not values:
        return {"low": 0.0, "high": 0.0}
    return {
        "low": float(np.percentile(values, 2.5)),
        "high": float(np.percentile(values, 97.5)),
    }


def evaluate_predictions(
    model: Pipeline,
    threshold: float,
    test: pd.DataFrame,
    features: list[str],
    bootstrap_rounds: int,
    random_state: int,
) -> dict[str, Any]:
    y_true = test["label"].astype(int).to_numpy()
    y_score = model.predict_proba(test[features])[:, 1]
    y_pred = (y_score >= threshold).astype(int)
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "confidence_intervals_95": {},
    }
    for name in ["f1", "precision", "recall", "roc_auc"]:
        metrics["confidence_intervals_95"][name] = bootstrap_ci(
            y_true, y_pred, y_score, name, bootstrap_rounds, random_state
        )
    return metrics


def utility_metrics(
    real_train: pd.DataFrame,
    real_val: pd.DataFrame,
    real_test: pd.DataFrame,
    synthetic: pd.DataFrame,
    max_train_rows: int,
    bootstrap_rounds: int,
    random_state: int,
) -> dict[str, Any]:
    features, numeric, categorical, dropped_empty = feature_columns(real_train)
    synthetic = align_to_columns(synthetic, real_train.columns.tolist())
    synthetic = synthetic.dropna(subset=["label"]).copy()
    synthetic["label"] = synthetic["label"].astype(int)

    real_train_sample = sample_frame(real_train, max_train_rows, random_state)
    synthetic_train = sample_frame(synthetic, min(len(real_train_sample), len(synthetic)), random_state + 1)
    augmented_train = pd.concat([real_train_sample, synthetic_train], ignore_index=True)

    protocols = {
        "real_only": real_train_sample,
        "synthetic_only": synthetic_train,
        "augmented": augmented_train,
    }

    results = {}
    for name, train_frame in protocols.items():
        print(f"[utility] Training {name} LightGBM on {len(train_frame):,} rows")
        model, threshold = train_ids_model(
            train=train_frame,
            val=real_val,
            features=features,
            numeric=numeric,
            categorical=categorical,
            random_state=random_state,
        )
        results[name] = evaluate_predictions(
            model=model,
            threshold=threshold,
            test=real_test,
            features=features,
            bootstrap_rounds=bootstrap_rounds,
            random_state=random_state,
        )

    baseline_f1 = results["real_only"]["f1"] or 1.0
    results["relative_performance"] = {
        "synthetic_only_f1_percent_of_real": float(results["synthetic_only"]["f1"] / baseline_f1 * 100),
        "augmented_f1_percent_of_real": float(results["augmented"]["f1"] / baseline_f1 * 100),
    }
    results["feature_columns"] = features
    results["dropped_empty_feature_columns"] = dropped_empty
    return results


def exact_match_count(synthetic: pd.DataFrame, real: pd.DataFrame) -> int:
    columns = [col for col in synthetic.columns if col in real.columns and col not in {"dataset_source", "source_file"}]
    if not columns:
        return 0
    real_hash = pd.util.hash_pandas_object(real[columns].astype("string"), index=False)
    synth_hash = pd.util.hash_pandas_object(synthetic[columns].astype("string"), index=False)
    return int(synth_hash.isin(set(real_hash.to_numpy())).sum())


def privacy_metrics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    max_rows: int,
    random_state: int,
) -> dict[str, Any]:
    real_sample = sample_frame(real, max_rows, random_state)
    synth_sample = sample_frame(synthetic, max_rows, random_state + 1)
    num_cols = [col for col in numeric_columns(real_sample) if col in synth_sample.columns]
    if not num_cols:
        return {"nearest_neighbor": {}, "exact_real_row_matches": exact_match_count(synthetic, real)}

    real_num = real_sample[num_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    synth_num = synth_sample[num_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    medians = real_num.median(numeric_only=True).fillna(0.0)
    real_num = real_num.fillna(medians)
    synth_num = synth_num.fillna(medians)

    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_num)
    synth_scaled = scaler.transform(synth_num)

    n_neighbors = 2 if len(real_scaled) > 1 else 1
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(real_scaled)
    distances, _ = nn.kneighbors(synth_scaled)
    dcr = distances[:, 0]
    if distances.shape[1] > 1:
        nndr = distances[:, 0] / np.maximum(distances[:, 1], 1e-12)
    else:
        nndr = np.ones_like(dcr)

    quasi = [col for col in ["protocol", "service", "dst_port", "attack_category"] if col in synthetic.columns]
    if quasi:
        k_min = int(synthetic.groupby(quasi, dropna=False).size().min())
    else:
        k_min = 0

    return {
        "rows_real_sampled": int(len(real_sample)),
        "rows_synthetic_sampled": int(len(synth_sample)),
        "exact_real_row_matches": exact_match_count(synthetic, real),
        "nearest_neighbor": {
            "dcr_mean": float(np.mean(dcr)),
            "dcr_median": float(np.median(dcr)),
            "dcr_p05": float(np.percentile(dcr, 5)),
            "dcr_p95": float(np.percentile(dcr, 95)),
            "nndr_mean": float(np.mean(nndr)),
            "nndr_median": float(np.median(nndr)),
        },
        "k_anonymity_min": k_min,
        "quasi_identifiers": quasi,
    }


def build_report(
    fidelity: dict[str, Any],
    utility: dict[str, Any],
    privacy: dict[str, Any],
    synthetic: pd.DataFrame,
) -> dict[str, Any]:
    real_f1 = utility["real_only"]["f1"]
    synth_ratio = utility["relative_performance"]["synthetic_only_f1_percent_of_real"]
    augmented_ratio = utility["relative_performance"]["augmented_f1_percent_of_real"]
    return {
        "executive_summary": [
            f"Generated synthetic dataset contains {len(synthetic):,} rows.",
            f"Real-only F1 is {real_f1:.4f}; synthetic-only reaches {synth_ratio:.2f}% of real-only F1.",
            f"Augmented training reaches {augmented_ratio:.2f}% of real-only F1 on the held-out real test set.",
        ],
        "synthetic_dataset": {
            "rows": int(len(synthetic)),
            "columns": int(len(synthetic.columns)),
            "category_counts": synthetic["attack_category"].value_counts().to_dict(),
            "label_counts": synthetic["label"].value_counts().sort_index().to_dict(),
        },
        "fidelity": fidelity,
        "utility": utility,
        "privacy": privacy,
        "recommendation": recommendation(fidelity, utility, privacy),
    }


def recommendation(fidelity: dict[str, Any], utility: dict[str, Any], privacy: dict[str, Any]) -> str:
    avg_js = fidelity.get("average_js_divergence", 1.0)
    synth_ratio = utility["relative_performance"]["synthetic_only_f1_percent_of_real"]
    dcr_p05 = privacy.get("nearest_neighbor", {}).get("dcr_p05", 0.0)
    if avg_js < 0.05 and synth_ratio >= 85 and dcr_p05 > 0.25:
        return "Synthetic dataset passes the configured research thresholds."
    return (
        "Synthetic dataset is usable for experimentation, but future work should add CTGAN/TVAE "
        "backends and stronger privacy auditing before public release."
    )


def write_reports(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "evaluation_report.json"
    md_path = output_dir / "evaluation_report.md"
    tex_path = output_dir / "evaluation_report.tex"
    json_path.write_text(json.dumps(report, indent=2, default=to_builtin), encoding="utf-8")

    summary = "\n".join(f"- {line}" for line in report["executive_summary"])
    utility = report["utility"]
    fidelity = report["fidelity"]
    privacy = report["privacy"]
    md = f"""# CyberSynth Evaluation Report

## Executive Summary

{summary}

## Synthetic Dataset

- Rows: {report['synthetic_dataset']['rows']:,}
- Columns: {report['synthetic_dataset']['columns']}
- Labels: {report['synthetic_dataset']['label_counts']}

## Fidelity

- Average JS divergence: {fidelity['average_js_divergence']:.6f}
- KS failure rate (p < 0.05): {fidelity['ks_failure_rate_p_lt_0_05']:.4f}
- Correlation preservation: {fidelity['correlation_preservation']:.4f}

## Utility on Held-Out Real Test Set

| Protocol | F1 | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Real only | {utility['real_only']['f1']:.4f} | {utility['real_only']['precision']:.4f} | {utility['real_only']['recall']:.4f} | {utility['real_only']['roc_auc']:.4f} |
| Synthetic only | {utility['synthetic_only']['f1']:.4f} | {utility['synthetic_only']['precision']:.4f} | {utility['synthetic_only']['recall']:.4f} | {utility['synthetic_only']['roc_auc']:.4f} |
| Real + synthetic | {utility['augmented']['f1']:.4f} | {utility['augmented']['precision']:.4f} | {utility['augmented']['recall']:.4f} | {utility['augmented']['roc_auc']:.4f} |

Synthetic-only relative F1: {utility['relative_performance']['synthetic_only_f1_percent_of_real']:.2f}%

Augmented relative F1: {utility['relative_performance']['augmented_f1_percent_of_real']:.2f}%

## Privacy

- Exact real row matches: {privacy['exact_real_row_matches']:,}
- DCR p05: {privacy['nearest_neighbor']['dcr_p05']:.4f}
- DCR median: {privacy['nearest_neighbor']['dcr_median']:.4f}
- NNDR median: {privacy['nearest_neighbor']['nndr_median']:.4f}
- k-anonymity minimum: {privacy['k_anonymity_min']}

## Recommendation

{report['recommendation']}
"""
    md_path.write_text(md, encoding="utf-8")

    tex = md.replace("# ", "\\section*{").replace("\n\n", "}\n\n", 1)
    tex_path.write_text(
        "\\documentclass{article}\n\\begin{document}\n"
        "\\section*{CyberSynth Evaluation Report}\n"
        + md.replace("_", "\\_").replace("%", "\\%")
        + "\n\\end{document}\n",
        encoding="utf-8",
    )
    print(f"Saved {json_path.relative_to(ROOT)}")
    print(f"Saved {md_path.relative_to(ROOT)}")
    print(f"Saved {tex_path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CyberSynth synthetic data")
    parser.add_argument("--real-train", default="data/processed/train.parquet")
    parser.add_argument("--real-val", default="data/processed/val.parquet")
    parser.add_argument("--real-test", default="data/processed/test.parquet")
    parser.add_argument("--synthetic", default="data/synthetic/synthetic_dataset_v1.0.parquet")
    parser.add_argument("--output-dir", default="outputs/reports")
    parser.add_argument("--max-fidelity-rows", type=int, default=20_000)
    parser.add_argument("--max-privacy-rows", type=int, default=5_000)
    parser.add_argument("--max-train-rows", type=int, default=220_378)
    parser.add_argument("--bootstrap-rounds", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    real_train = pd.read_parquet(resolve_path(args.real_train))
    real_val = pd.read_parquet(resolve_path(args.real_val))
    real_test = pd.read_parquet(resolve_path(args.real_test))
    synthetic = pd.read_parquet(resolve_path(args.synthetic))
    synthetic = align_to_columns(synthetic, real_train.columns.tolist())

    print("[fidelity] Comparing real train and synthetic distributions")
    fidelity = fidelity_metrics(
        real=real_train,
        synthetic=synthetic,
        max_rows=args.max_fidelity_rows,
        bins=30,
        random_state=args.random_state,
    )
    print("[utility] Training IDS models for protocol comparison")
    utility = utility_metrics(
        real_train=real_train,
        real_val=real_val,
        real_test=real_test,
        synthetic=synthetic,
        max_train_rows=args.max_train_rows,
        bootstrap_rounds=args.bootstrap_rounds,
        random_state=args.random_state,
    )
    print("[privacy] Computing distance-to-real checks")
    privacy = privacy_metrics(
        real=real_train,
        synthetic=synthetic,
        max_rows=args.max_privacy_rows,
        random_state=args.random_state,
    )
    report = build_report(fidelity=fidelity, utility=utility, privacy=privacy, synthetic=synthetic)
    write_reports(report, resolve_path(args.output_dir))


if __name__ == "__main__":
    main()
