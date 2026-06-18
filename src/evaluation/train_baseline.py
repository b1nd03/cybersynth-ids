"""
Train and evaluate an improved LightGBM IDS model.

The model predicts:
    label = 0 -> Normal
    label = 1 -> Attack

The training flow now uses:
    - fixed train/validation/test splits
    - LightGBM early stopping on the validation split
    - a small validation-based model search
    - a validation-tuned decision threshold
    - per-attack-category reporting and feature importance
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[2]

LEAKY_COLUMNS = {
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


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    candidate_features = [c for c in df.columns if c not in LEAKY_COLUMNS]
    dropped_empty = [c for c in candidate_features if df[c].isna().all()]
    features = [c for c in candidate_features if c not in dropped_empty]
    categorical = [
        c
        for c in features
        if str(df[c].dtype) in {"string", "object", "category"} or c in {"protocol", "service"}
    ]
    numeric = [c for c in features if c not in categorical]
    return features, numeric, categorical, dropped_empty


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def candidate_configs(random_state: int) -> list[dict[str, Any]]:
    common = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": 1200,
        "learning_rate": 0.035,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }
    return [
        {
            "name": "regularized_balanced",
            "params": {
                **common,
                "num_leaves": 63,
                "min_child_samples": 80,
                "reg_alpha": 0.05,
                "reg_lambda": 2.0,
                "class_weight": "balanced",
            },
        },
        {
            "name": "plain_probability",
            "params": {
                **common,
                "num_leaves": 63,
                "min_child_samples": 80,
                "reg_alpha": 0.05,
                "reg_lambda": 2.0,
            },
        },
        {
            "name": "deeper_recall",
            "params": {
                **common,
                "num_leaves": 127,
                "min_child_samples": 40,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "class_weight": "balanced",
            },
        },
    ]


def tune_threshold(y_true: pd.Series, y_score: np.ndarray) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return {
            "threshold": 0.5,
            "precision": float(precision_score(y_true, y_score >= 0.5, zero_division=0)),
            "recall": float(recall_score(y_true, y_score >= 0.5, zero_division=0)),
            "f1": float(f1_score(y_true, y_score >= 0.5, zero_division=0)),
        }

    pr = precision[:-1]
    rc = recall[:-1]
    f1 = np.divide(2 * pr * rc, pr + rc, out=np.zeros_like(pr), where=(pr + rc) > 0)
    best_index = int(np.nanargmax(f1))
    return {
        "threshold": float(thresholds[best_index]),
        "precision": float(pr[best_index]),
        "recall": float(rc[best_index]),
        "f1": float(f1[best_index]),
    }


def evaluate(model: Pipeline, df: pd.DataFrame, split_name: str, threshold: float) -> dict[str, Any]:
    y_true = df["label"].astype(int)
    x = df.drop(columns=["label"])
    y_score = model.predict_proba(x)[:, 1]
    y_pred = (y_score >= threshold).astype(int)

    metrics: dict[str, Any] = {
        "split": split_name,
        "rows": int(len(df)),
        "decision_threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["Normal", "Attack"],
            zero_division=0,
            output_dict=True,
        ),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["average_precision"] = float(average_precision_score(y_true, y_score))

    if "attack_category" in df.columns:
        metrics["attack_category_metrics"] = category_metrics(df, y_pred, y_score)

    return metrics


def category_metrics(df: pd.DataFrame, y_pred: np.ndarray, y_score: np.ndarray) -> list[dict[str, Any]]:
    report_frame = df[["attack_category", "label"]].copy()
    report_frame["prediction"] = y_pred
    report_frame["attack_probability"] = y_score

    rows = []
    for category, group in report_frame.groupby("attack_category", dropna=False):
        label = int(group["label"].mode().iloc[0])
        attack_rate = float(group["prediction"].mean())
        row = {
            "attack_category": str(category),
            "rows": int(len(group)),
            "label": label,
            "mean_attack_probability": float(group["attack_probability"].mean()),
        }
        if label == 1:
            row["detection_rate"] = attack_rate
        else:
            row["false_positive_rate"] = attack_rate
            row["specificity"] = 1.0 - attack_rate
        rows.append(row)

    return sorted(rows, key=lambda item: item["rows"], reverse=True)


def feature_importance(model: Pipeline, top_n: int = 30) -> dict[str, list[dict[str, Any]]]:
    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]

    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        names = np.array([f"feature_{i}" for i in range(classifier.n_features_in_)])

    gain = classifier.booster_.feature_importance(importance_type="gain")
    raw = [
        {"feature": str(name), "gain": float(value)}
        for name, value in zip(names, gain, strict=False)
        if float(value) > 0
    ]
    raw.sort(key=lambda item: item["gain"], reverse=True)

    aggregated: dict[str, float] = {}
    for item in raw:
        name = item["feature"]
        if "__" in name:
            source = name.split("__", 1)[1]
        else:
            source = name
        if "_" in source and name.startswith("categorical__"):
            source = source.split("_", 1)[0]
        aggregated[source] = aggregated.get(source, 0.0) + item["gain"]

    by_original = [
        {"feature": key, "gain": float(value)}
        for key, value in sorted(aggregated.items(), key=lambda item: item[1], reverse=True)
    ]
    return {
        "top_transformed": raw[:top_n],
        "top_original": by_original[:top_n],
    }


def fit_best_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    numeric: list[str],
    categorical: list[str],
    random_state: int,
    early_stopping_rounds: int,
) -> tuple[Pipeline, float, dict[str, Any], list[dict[str, Any]]]:
    preprocessor = build_preprocessor(numeric, categorical)
    x_train = preprocessor.fit_transform(train_df[features])
    x_val = preprocessor.transform(val_df[features])
    y_train = train_df["label"].astype(int)
    y_val = val_df["label"].astype(int)

    best_model: Pipeline | None = None
    best_threshold = 0.5
    best_threshold_metrics: dict[str, Any] = {}
    best_score = -1.0
    candidate_results = []

    for candidate in candidate_configs(random_state):
        print(f"Trying LightGBM candidate: {candidate['name']}")
        classifier = LGBMClassifier(**candidate["params"])
        classifier.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[
                early_stopping(early_stopping_rounds, first_metric_only=True, verbose=False),
                log_evaluation(period=0),
            ],
        )
        model = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("classifier", classifier),
            ]
        )
        val_score = classifier.predict_proba(x_val)[:, 1]
        threshold_metrics = tune_threshold(y_val, val_score)
        default_pred = (val_score >= 0.5).astype(int)
        candidate_result = {
            "name": candidate["name"],
            "best_iteration": int(getattr(classifier, "best_iteration_", 0) or classifier.n_estimators),
            "validation_default_f1": float(f1_score(y_val, default_pred, zero_division=0)),
            "validation_tuned_threshold": threshold_metrics,
            "params": candidate["params"],
        }
        candidate_results.append(candidate_result)
        print(
            "  val default f1="
            f"{candidate_result['validation_default_f1']:.4f} "
            "tuned f1="
            f"{threshold_metrics['f1']:.4f} "
            "threshold="
            f"{threshold_metrics['threshold']:.4f}"
        )

        if threshold_metrics["f1"] > best_score:
            best_score = threshold_metrics["f1"]
            best_model = model
            best_threshold = threshold_metrics["threshold"]
            best_threshold_metrics = threshold_metrics

    if best_model is None:
        raise RuntimeError("No model candidate was trained")

    return best_model, best_threshold, best_threshold_metrics, candidate_results


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

    parser = argparse.ArgumentParser(description="Train improved LightGBM IDS model")
    parser.add_argument("--train", default="data/processed/train.parquet")
    parser.add_argument("--val", default="data/processed/val.parquet")
    parser.add_argument("--test", default="data/processed/test.parquet")
    parser.add_argument("--model-output", default="models/baseline_lightgbm.joblib")
    parser.add_argument("--metrics-output", default="outputs/metrics/baseline_lightgbm_metrics.json")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--early-stopping-rounds", type=int, default=60)
    args = parser.parse_args()

    train_path = resolve_path(args.train)
    val_path = resolve_path(args.val)
    test_path = resolve_path(args.test)
    model_path = resolve_path(args.model_output)
    metrics_path = resolve_path(args.metrics_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {train_path.relative_to(ROOT)}")
    train_df = pd.read_parquet(train_path)
    print(f"Reading {val_path.relative_to(ROOT)}")
    val_df = pd.read_parquet(val_path)
    print(f"Reading {test_path.relative_to(ROOT)}")
    test_df = pd.read_parquet(test_path)

    features, numeric, categorical, dropped_empty = feature_columns(train_df)
    print(f"Training LightGBM on {len(train_df):,} rows")
    print(f"Features: {len(features)} total, {len(numeric)} numeric, {len(categorical)} categorical")
    if dropped_empty:
        print(f"Dropped all-empty features: {', '.join(dropped_empty)}")

    model, threshold, threshold_metrics, candidate_results = fit_best_model(
        train_df=train_df,
        val_df=val_df,
        features=features,
        numeric=numeric,
        categorical=categorical,
        random_state=args.random_state,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    val_metrics = evaluate(model, val_df, "val", threshold)
    test_metrics = evaluate(model, test_df, "test", threshold)
    metrics = {
        "model": "LightGBMClassifier",
        "target": "binary_label",
        "training_strategy": "lightgbm_early_stopping_threshold_tuning",
        "selected_candidate": max(
            candidate_results,
            key=lambda item: item["validation_tuned_threshold"]["f1"],
        )["name"],
        "candidate_results": candidate_results,
        "decision_threshold": float(threshold),
        "threshold_tuning": threshold_metrics,
        "feature_columns": features,
        "numeric_feature_columns": numeric,
        "categorical_feature_columns": categorical,
        "dropped_empty_feature_columns": dropped_empty,
        "feature_importance": feature_importance(model),
        "train_rows": int(len(train_df)),
        "val": val_metrics,
        "test": test_metrics,
    }

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print()
    print("Validation:")
    print(
        f"  accuracy={val_metrics['accuracy']:.4f} "
        f"precision={val_metrics['precision']:.4f} "
        f"recall={val_metrics['recall']:.4f} "
        f"f1={val_metrics['f1']:.4f} "
        f"roc_auc={val_metrics.get('roc_auc', float('nan')):.4f}"
    )
    print("Test:")
    print(
        f"  accuracy={test_metrics['accuracy']:.4f} "
        f"precision={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f} "
        f"f1={test_metrics['f1']:.4f} "
        f"roc_auc={test_metrics.get('roc_auc', float('nan')):.4f}"
    )
    print(f"Decision threshold: {threshold:.4f}")
    print(f"Saved model: {model_path.relative_to(ROOT)}")
    print(f"Saved metrics: {metrics_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
