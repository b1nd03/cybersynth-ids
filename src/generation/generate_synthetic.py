"""
Generate a synthetic cybersecurity dataset from the processed real training data.

This is the project's first working dataset generator. It is a conditioned
statistical bootstrap generator:

- samples within each attack category
- adds category-specific numeric variation
- samples categorical fields from observed category distributions
- preserves the unified schema created by src/ingestion/preprocessor.py

It is intentionally dependency-light so it works before heavier TVAE/CTGAN
libraries are installed. Later generator backends can use the same input/output
contract.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
warnings.filterwarnings("ignore", message="Mean of empty slice")

META_COLUMNS = {
    "dataset_source",
    "source_file",
    "environment_type",
    "attack_category",
    "attack_subcategory",
    "label",
}

INTEGER_LIKE_COLUMNS = {
    "src_port",
    "dst_port",
    "total_packets_fwd",
    "total_packets_bwd",
    "fin_flag_count",
    "syn_flag_count",
    "rst_flag_count",
    "psh_flag_count",
    "ack_flag_count",
    "urg_flag_count",
    "cwe_flag_count",
    "ece_flag_count",
    "fwd_header_length",
    "bwd_header_length",
}

NON_NEGATIVE_COLUMNS = {
    "duration",
    "src_port",
    "dst_port",
    "src_bytes",
    "dst_bytes",
    "total_packets_fwd",
    "total_packets_bwd",
    "packet_length_mean",
    "packet_length_std",
    "packet_length_min",
    "packet_length_max",
    "flow_bytes_per_sec",
    "flow_packets_per_sec",
    "fwd_iat_mean",
    "bwd_iat_mean",
    "fwd_iat_std",
    "bwd_iat_std",
    "fin_flag_count",
    "syn_flag_count",
    "rst_flag_count",
    "psh_flag_count",
    "ack_flag_count",
    "urg_flag_count",
    "cwe_flag_count",
    "ece_flag_count",
    "fwd_header_length",
    "bwd_header_length",
    "active_mean",
    "active_std",
    "idle_mean",
    "idle_std",
    "bytes_per_packet_ratio",
    "down_up_ratio",
    "avg_fwd_segment_size",
    "avg_bwd_segment_size",
    "subflow_fwd_bytes",
    "subflow_bwd_bytes",
    "bulk_rate_fwd",
    "bulk_rate_bwd",
}


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    features = [c for c in df.columns if c not in META_COLUMNS]
    numeric = df[features].select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in features if c not in numeric]
    return features, numeric, categorical


def allocate_counts(
    category_counts: pd.Series,
    total_rows: int,
    mode: str,
    minimum_per_category: int,
    label_by_category: dict[str, int] | None = None,
) -> dict[str, int]:
    categories = category_counts.index.tolist()
    if not categories:
        return {}

    if mode == "label_balanced" and label_by_category:
        normal_categories = [cat for cat in categories if int(label_by_category.get(str(cat), 1)) == 0]
        attack_categories = [cat for cat in categories if int(label_by_category.get(str(cat), 1)) == 1]
        if normal_categories and attack_categories:
            normal_rows = total_rows // 2
            attack_rows = total_rows - normal_rows
            normal_counts = allocate_counts(
                category_counts.loc[normal_categories],
                normal_rows,
                "proportional",
                minimum_per_category,
            )
            attack_counts = allocate_counts(
                category_counts.loc[attack_categories],
                attack_rows,
                "balanced",
                minimum_per_category,
            )
            merged = {**normal_counts, **attack_counts}
            diff = total_rows - sum(merged.values())
            if diff:
                target = max(merged, key=merged.get)
                merged[target] += diff
            return {key: int(value) for key, value in merged.items() if int(value) > 0}

    if mode == "balanced":
        base = pd.Series(1.0, index=categories)
    else:
        base = category_counts.astype(float)

    effective_minimum = int(minimum_per_category)
    if effective_minimum * len(categories) > total_rows:
        effective_minimum = max(0, total_rows // len(categories))

    weights = base / base.sum()
    raw = weights * total_rows
    counts = np.floor(raw).astype(int)

    for category in categories:
        if category_counts[category] > 0:
            counts[category] = max(int(counts[category]), effective_minimum)

    overflow_guard = 0
    while counts.sum() > total_rows and overflow_guard < 100000:
        idx = counts.idxmax()
        if counts[idx] > effective_minimum:
            counts[idx] -= 1
        else:
            break
        overflow_guard += 1

    remainder = total_rows - int(counts.sum())
    if remainder > 0:
        fractional = (raw - np.floor(raw)).sort_values(ascending=False)
        order = fractional.index.tolist() or categories
        for i in range(remainder):
            counts[order[i % len(order)]] += 1

    return {str(k): int(v) for k, v in counts.items() if int(v) > 0}


def safe_mode(series: pd.Series, default: object) -> object:
    clean = series.dropna()
    if clean.empty:
        return default
    return clean.mode(dropna=True).iloc[0]


def fill_missing_for_generation(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> pd.DataFrame:
    filled = df.copy()
    for col in numeric_cols:
        median = filled[col].median(skipna=True)
        if pd.isna(median):
            median = 0.0
        filled[col] = pd.to_numeric(filled[col], errors="coerce").fillna(median)
    for col in categorical_cols:
        filled[col] = filled[col].astype("string").fillna("")
    return filled


def jitter_numeric(
    values: pd.DataFrame,
    source_group: pd.DataFrame,
    numeric_cols: list[str],
    rng: np.random.Generator,
    noise_scale: float,
    clip_quantile: float,
) -> pd.DataFrame:
    generated = values.copy()
    if noise_scale <= 0:
        return generated

    clip_quantile = min(max(float(clip_quantile), 0.0), 0.1)
    stats = source_group[numeric_cols].agg(["std"]).iloc[0].fillna(0.0)
    if clip_quantile > 0:
        lower = source_group[numeric_cols].quantile(clip_quantile, numeric_only=True).fillna(0.0)
        upper = source_group[numeric_cols].quantile(1.0 - clip_quantile, numeric_only=True).fillna(0.0)
    else:
        lower = source_group[numeric_cols].min(numeric_only=True).fillna(0.0)
        upper = source_group[numeric_cols].max(numeric_only=True).fillna(0.0)

    for col in numeric_cols:
        std = float(stats.get(col, 0.0))
        if std > 0 and not math.isnan(std):
            noise = rng.normal(loc=0.0, scale=std * noise_scale, size=len(generated))
            generated[col] = pd.to_numeric(generated[col], errors="coerce") + noise

        if col in NON_NEGATIVE_COLUMNS:
            generated[col] = generated[col].clip(lower=0)

        lo = float(lower.get(col, 0.0))
        hi = float(upper.get(col, 0.0))
        if hi > lo:
            generated[col] = generated[col].clip(lower=lo, upper=hi)

        if col in INTEGER_LIKE_COLUMNS:
            generated[col] = generated[col].round()

    if "src_port" in generated:
        generated["src_port"] = generated["src_port"].clip(lower=0, upper=65535).round()
    if "dst_port" in generated:
        generated["dst_port"] = generated["dst_port"].clip(lower=0, upper=65535).round()

    return generated


def recompute_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    generated = frame.copy()

    packet_cols = {"total_packets_fwd", "total_packets_bwd", "src_bytes", "dst_bytes"}
    if packet_cols.issubset(generated.columns):
        packets = generated["total_packets_fwd"].fillna(0) + generated["total_packets_bwd"].fillna(0)
        total_bytes = generated["src_bytes"].fillna(0) + generated["dst_bytes"].fillna(0)
        valid = packets.gt(0)
        generated.loc[valid, "bytes_per_packet_ratio"] = total_bytes[valid] / packets[valid]

    if {"src_bytes", "dst_bytes", "duration", "flow_bytes_per_sec"}.issubset(generated.columns):
        duration = generated["duration"].replace(0, np.nan)
        total_bytes = generated["src_bytes"].fillna(0) + generated["dst_bytes"].fillna(0)
        valid = duration.notna() & duration.gt(0)
        generated.loc[valid, "flow_bytes_per_sec"] = total_bytes[valid] / duration[valid]

    if {"total_packets_fwd", "total_packets_bwd", "duration", "flow_packets_per_sec"}.issubset(generated.columns):
        duration = generated["duration"].replace(0, np.nan)
        packets = generated["total_packets_fwd"].fillna(0) + generated["total_packets_bwd"].fillna(0)
        valid = duration.notna() & duration.gt(0)
        generated.loc[valid, "flow_packets_per_sec"] = packets[valid] / duration[valid]

    if {"total_packets_fwd", "total_packets_bwd", "down_up_ratio"}.issubset(generated.columns):
        fwd = generated["total_packets_fwd"].replace(0, np.nan)
        valid = fwd.notna() & fwd.gt(0)
        generated.loc[valid, "down_up_ratio"] = generated.loc[valid, "total_packets_bwd"].fillna(0) / fwd[valid]

    generated.replace([np.inf, -np.inf], np.nan, inplace=True)
    return generated


def generate_for_category(
    category: str,
    source_group: pd.DataFrame,
    rows: int,
    numeric_cols: list[str],
    categorical_cols: list[str],
    rng: np.random.Generator,
    noise_scale: float,
    clip_quantile: float,
) -> pd.DataFrame:
    sampled_idx = rng.integers(0, len(source_group), size=rows)
    sampled = source_group.iloc[sampled_idx].reset_index(drop=True)

    generated = sampled.copy()
    generated[numeric_cols] = jitter_numeric(
        generated[numeric_cols],
        source_group,
        numeric_cols,
        rng,
        noise_scale,
        clip_quantile,
    )

    for col in categorical_cols:
        values = source_group[col].dropna().astype("string")
        if values.empty:
            generated[col] = ""
        else:
            generated[col] = rng.choice(values.to_numpy(), size=rows, replace=True)

    generated = recompute_derived_columns(generated)

    generated["dataset_source"] = "Synthetic-CyberSynth"
    generated["source_file"] = f"generated:{category}"
    generated["attack_category"] = category
    generated["label"] = 0 if category.lower() == "normal" else 1
    generated["environment_type"] = safe_mode(source_group["environment_type"], "Synthetic")
    subcategories = source_group["attack_subcategory"].dropna().astype("string").to_numpy()
    if len(subcategories):
        generated["attack_subcategory"] = rng.choice(subcategories, size=rows, replace=True)
    else:
        generated["attack_subcategory"] = category
    generated.loc[generated["label"].eq(0), "attack_subcategory"] = "Normal"
    return generated


def filter_real_data(
    real: pd.DataFrame,
    categories: set[str] | None,
    subcategories: set[str] | None,
    dataset_sources: set[str] | None,
    environment_types: set[str] | None,
    labels: set[int] | None,
) -> tuple[pd.DataFrame, dict[str, list[str] | list[int]]]:
    filtered = real.copy()
    applied: dict[str, list[str] | list[int]] = {}

    if categories:
        filtered = filtered[filtered["attack_category"].astype(str).isin(categories)].copy()
        applied["attack_categories"] = sorted(categories)
    if subcategories:
        filtered = filtered[filtered["attack_subcategory"].astype(str).isin(subcategories)].copy()
        applied["attack_subcategories"] = sorted(subcategories)
    if dataset_sources:
        filtered = filtered[filtered["dataset_source"].astype(str).isin(dataset_sources)].copy()
        applied["dataset_sources"] = sorted(dataset_sources)
    if environment_types:
        filtered = filtered[filtered["environment_type"].astype(str).isin(environment_types)].copy()
        applied["environment_types"] = sorted(environment_types)
    if labels:
        filtered = filtered[filtered["label"].astype(int).isin(labels)].copy()
        applied["labels"] = sorted(labels)

    return filtered, applied


def exact_match_mask(synthetic: pd.DataFrame, real: pd.DataFrame, columns: list[str]) -> pd.Series:
    safe_columns = [col for col in columns if col in synthetic.columns and col in real.columns]
    if not safe_columns:
        return pd.Series(False, index=synthetic.index)
    real_hash = pd.util.hash_pandas_object(real[safe_columns].astype("string"), index=False)
    synthetic_hash = pd.util.hash_pandas_object(synthetic[safe_columns].astype("string"), index=False)
    return synthetic_hash.isin(set(real_hash.to_numpy()))


def reduce_exact_matches(
    synthetic: pd.DataFrame,
    real: pd.DataFrame,
    numeric_cols: list[str],
    rng: np.random.Generator,
    noise_scale: float,
    clip_quantile: float,
) -> pd.DataFrame:
    compare_cols = [c for c in synthetic.columns if c not in {"dataset_source", "source_file"}]
    mask = exact_match_mask(synthetic, real, compare_cols)
    if not bool(mask.any()) or not numeric_cols:
        return synthetic

    adjusted = synthetic.copy()
    for category, index in adjusted.loc[mask].groupby("attack_category").groups.items():
        source_group = real[real["attack_category"].astype(str).eq(str(category))]
        if source_group.empty:
            source_group = real
        adjusted.loc[index, numeric_cols] = jitter_numeric(
            adjusted.loc[index, numeric_cols],
            source_group,
            numeric_cols,
            rng,
            max(noise_scale, 0.03),
            clip_quantile,
        )
    return recompute_derived_columns(adjusted)


def quality_report(
    synthetic: pd.DataFrame,
    real: pd.DataFrame,
    feature_columns: list[str],
    applied_filters: dict[str, list[str] | list[int]],
) -> dict[str, object]:
    compare_cols = [c for c in synthetic.columns if c not in {"dataset_source", "source_file"}]
    exact_matches = exact_match_mask(synthetic, real, compare_cols)
    non_negative_violations = {
        col: int((synthetic[col] < 0).sum())
        for col in sorted(NON_NEGATIVE_COLUMNS)
        if col in synthetic.columns and pd.api.types.is_numeric_dtype(synthetic[col])
    }
    non_negative_violations = {key: value for key, value in non_negative_violations.items() if value}

    port_range_violations = {}
    for col in ["src_port", "dst_port"]:
        if col in synthetic.columns:
            count = int(((synthetic[col] < 0) | (synthetic[col] > 65535)).sum())
            if count:
                port_range_violations[col] = count

    return {
        "schema_match": list(synthetic.columns) == list(real.columns),
        "missing_cells": int(synthetic.isna().sum().sum()),
        "duplicate_rows": int(synthetic.duplicated().sum()),
        "exact_real_row_matches": int(exact_matches.sum()),
        "non_negative_violations": non_negative_violations,
        "port_range_violations": port_range_violations,
        "selected_filters": applied_filters,
        "feature_columns": feature_columns,
    }


def generate_synthetic(
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    rows: int,
    mode: str,
    minimum_per_category: int,
    noise_scale: float,
    random_state: int,
    categories: set[str] | None,
    subcategories: set[str] | None = None,
    dataset_sources: set[str] | None = None,
    environment_types: set[str] | None = None,
    labels: set[int] | None = None,
    clip_quantile: float = 0.005,
    drop_exact_matches: bool = True,
) -> dict:
    real = pd.read_parquet(input_path)
    real, applied_filters = filter_real_data(
        real,
        categories=categories,
        subcategories=subcategories,
        dataset_sources=dataset_sources,
        environment_types=environment_types,
        labels=labels,
    )
    if real.empty:
        raise SystemExit("No rows available for generation")

    _, numeric_cols, categorical_cols = feature_columns(real)
    real = fill_missing_for_generation(real, numeric_cols, categorical_cols)
    real["attack_category"] = real["attack_category"].astype("string").fillna("Unknown")
    real["attack_subcategory"] = real["attack_subcategory"].astype("string").fillna("Unknown")

    category_counts = real["attack_category"].value_counts()
    label_by_category = (
        real.groupby("attack_category")["label"]
        .agg(lambda values: int(pd.Series(values).astype(int).mode().iloc[0]))
        .to_dict()
    )
    allocation = allocate_counts(
        category_counts,
        rows,
        mode,
        minimum_per_category,
        label_by_category={str(key): int(value) for key, value in label_by_category.items()},
    )

    rng = np.random.default_rng(random_state)
    generated_parts = []
    for category, count in allocation.items():
        group = real[real["attack_category"].astype(str).eq(category)].copy()
        if group.empty or count <= 0:
            continue
        print(f"[generate] {category:<18} {count:,} rows from {len(group):,} real rows")
        generated_parts.append(
            generate_for_category(
                category=category,
                source_group=group,
                rows=count,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                rng=rng,
                noise_scale=noise_scale,
                clip_quantile=clip_quantile,
            )
        )

    synthetic = pd.concat(generated_parts, ignore_index=True)
    synthetic = synthetic[real.columns]
    if drop_exact_matches:
        synthetic = reduce_exact_matches(
            synthetic=synthetic,
            real=real,
            numeric_cols=numeric_cols,
            rng=rng,
            noise_scale=noise_scale,
            clip_quantile=clip_quantile,
        )
    synthetic = synthetic.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_parquet(output_path, index=False)

    summary = {
        "generator": "conditioned_statistical_bootstrap",
        "input": str(input_path.relative_to(ROOT)),
        "output": str(output_path.relative_to(ROOT)),
        "rows_requested": rows,
        "rows_generated": int(len(synthetic)),
        "mode": mode,
        "noise_scale": noise_scale,
        "clip_quantile": clip_quantile,
        "random_state": random_state,
        "filters": applied_filters,
        "category_counts": synthetic["attack_category"].value_counts().to_dict(),
        "label_counts": synthetic["label"].value_counts().sort_index().to_dict(),
        "feature_columns": [c for c in synthetic.columns if c not in META_COLUMNS],
        "quality": quality_report(
            synthetic=synthetic,
            real=real,
            feature_columns=[c for c in synthetic.columns if c not in META_COLUMNS],
            applied_filters=applied_filters,
        ),
        "notes": [
            "Conditioned statistical generator with category/source/environment/subcategory filters.",
            "Numeric columns are jittered, clipped to observed quantile bounds, and rounded where needed.",
            "Exact real-row match reduction is enabled unless explicitly disabled.",
            "CTGAN/TVAE can be added later through the same input/output contract.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic cybersecurity dataset")
    parser.add_argument("--input", default="data/processed/train.parquet")
    parser.add_argument("--output", default="data/synthetic/synthetic_dataset.parquet")
    parser.add_argument("--summary-output", default="data/synthetic/synthetic_summary.json")
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument(
        "--mode",
        choices=["proportional", "balanced", "label_balanced"],
        default="proportional",
    )
    parser.add_argument("--minimum-per-category", type=int, default=25)
    parser.add_argument("--noise-scale", type=float, default=0.08)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--categories", nargs="+", help="Optional attack categories to generate")
    parser.add_argument("--subcategories", nargs="+", help="Optional attack subcategories to generate")
    parser.add_argument("--dataset-sources", nargs="+", help="Optional source datasets to generate from")
    parser.add_argument("--environment-types", nargs="+", help="Optional environments to generate from")
    parser.add_argument("--labels", nargs="+", type=int, choices=[0, 1], help="Optional labels: 0 normal, 1 attack")
    parser.add_argument("--clip-quantile", type=float, default=0.005)
    parser.add_argument("--keep-exact-matches", action="store_true")
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    summary_path = resolve_path(args.summary_output)
    categories = set(args.categories) if args.categories else None

    print(f"Input  : {input_path.relative_to(ROOT)}")
    print(f"Output : {output_path.relative_to(ROOT)}")
    print(f"Rows   : {args.rows:,}")
    summary = generate_synthetic(
        input_path=input_path,
        output_path=output_path,
        summary_path=summary_path,
        rows=args.rows,
        mode=args.mode,
        minimum_per_category=args.minimum_per_category,
        noise_scale=args.noise_scale,
        random_state=args.random_state,
        categories=categories,
        subcategories=set(args.subcategories) if args.subcategories else None,
        dataset_sources=set(args.dataset_sources) if args.dataset_sources else None,
        environment_types=set(args.environment_types) if args.environment_types else None,
        labels=set(args.labels) if args.labels else None,
        clip_quantile=args.clip_quantile,
        drop_exact_matches=not args.keep_exact_matches,
    )
    print()
    print(f"Done. Generated {summary['rows_generated']:,} rows")
    print(f"Summary: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
