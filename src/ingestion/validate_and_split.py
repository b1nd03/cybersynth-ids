"""
Validate the unified dataset and create train/validation/test Parquet splits.

The split is stratified by attack category where possible, while rare categories
fall back to the binary label to keep sklearn's splitter happy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
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


def safe_stratify_key(df: pd.DataFrame, min_count: int = 10) -> pd.Series:
    base = df["attack_category"].astype("string").fillna("Unknown")
    counts = base.value_counts(dropna=False)
    rare = base.map(counts).fillna(0).lt(min_count)
    fallback = "label_" + df["label"].astype(str)
    return base.mask(rare, fallback)


def build_quality_report(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
    feature_cols = [c for c in df_after.columns if c not in META_COLUMNS]
    numeric_cols = df_after[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    missing_total = df_after.isna().sum().sort_values(ascending=False)
    missing_nonzero = missing_total[missing_total > 0].head(30)

    return {
        "rows_before_dedup": int(len(df_before)),
        "rows_after_dedup": int(len(df_after)),
        "duplicates_removed": int(len(df_before) - len(df_after)),
        "columns": int(len(df_after.columns)),
        "feature_columns": feature_cols,
        "numeric_feature_columns": numeric_cols,
        "categorical_feature_columns": categorical_cols,
        "dataset_counts": df_after["dataset_source"].value_counts().to_dict(),
        "label_counts": df_after["label"].value_counts().sort_index().to_dict(),
        "attack_category_counts": df_after["attack_category"].value_counts().to_dict(),
        "missing_values_top": missing_nonzero.astype(int).to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and split unified dataset")
    parser.add_argument("--input", default="data/processed/unified_dataset.parquet")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--report", default="data/processed/dataset_quality_report.json")
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if round(args.train_size + args.val_size + args.test_size, 6) != 1.0:
        raise SystemExit("train-size + val-size + test-size must equal 1.0")

    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    report_path = resolve_path(args.report)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {input_path.relative_to(ROOT)}")
    df_before = pd.read_parquet(input_path)
    df = df_before.copy()

    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype("int8")
    df["attack_category"] = df["attack_category"].astype("string").fillna("Unknown")
    df["attack_subcategory"] = df["attack_subcategory"].astype("string").fillna("Unknown")
    df["protocol"] = df["protocol"].astype("string").fillna("")
    df["service"] = df["service"].astype("string").fillna("")

    df = df.drop_duplicates().reset_index(drop=True)
    report = build_quality_report(df_before, df)

    stratify = safe_stratify_key(df)
    train_df, temp_df = train_test_split(
        df,
        train_size=args.train_size,
        random_state=args.random_state,
        shuffle=True,
        stratify=stratify,
    )

    temp_stratify = safe_stratify_key(temp_df, min_count=4)
    relative_val_size = args.val_size / (args.val_size + args.test_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_size,
        random_state=args.random_state,
        shuffle=True,
        stratify=temp_stratify,
    )

    split_counts = {
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "test": int(len(test_df)),
    }
    report["split_counts"] = split_counts
    report["split_label_counts"] = {
        "train": train_df["label"].value_counts().sort_index().to_dict(),
        "val": val_df["label"].value_counts().sort_index().to_dict(),
        "test": test_df["label"].value_counts().sort_index().to_dict(),
    }

    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Wrote splits:")
    for name, count in split_counts.items():
        print(f"  {name:<5} {count:,} rows")
    print(f"Quality report: {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
