"""
Build a unified cybersecurity dataset from the locally downloaded raw datasets.

The first production-safe output is a normalized Parquet table with shared
metadata, binary labels, attack categories, and a practical network-flow schema.
By default the script limits each dataset to 50,000 rows so you can verify the
pipeline quickly. Use --full when you are ready to process every available row.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[2]

KDD_FEATURE_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

META_COLUMNS = [
    "dataset_source",
    "source_file",
    "environment_type",
    "attack_category",
    "attack_subcategory",
    "label",
]

STRING_FEATURES = [
    "protocol",
    "service",
]

NUMERIC_FEATURES = [
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
]

OUTPUT_COLUMNS = META_COLUMNS + STRING_FEATURES + NUMERIC_FEATURES

ALIASES = {
    "duration": ["duration", "dur", "flow duration", "flow_duration", "tcp.time_delta"],
    "protocol": ["protocol", "proto", "protocol_type", "mqtt.protoname"],
    "service": ["service"],
    "src_port": ["src_port", "sport", "src port", "source port", "originp", "srcport"],
    "dst_port": ["dst_port", "dport", "dst port", "destination port", "responp", "dstport"],
    "src_bytes": [
        "src_bytes",
        "sbytes",
        "fwd packets length total",
        "totlen fwd pkts",
        "total length of fwd packets",
        "fwd_bytes",
        "tcp.len",
    ],
    "dst_bytes": [
        "dst_bytes",
        "dbytes",
        "bwd packets length total",
        "totlen bwd pkts",
        "total length of bwd packets",
        "bwd_bytes",
    ],
    "total_packets_fwd": [
        "total fwd packets",
        "tot fwd pkts",
        "spkts",
        "fwd_pkts_tot",
        "fwd packets",
    ],
    "total_packets_bwd": [
        "total backward packets",
        "tot bwd pkts",
        "dpkts",
        "bwd_pkts_tot",
        "bwd packets",
    ],
    "packet_length_mean": ["packet length mean", "pkt_len_avg", "flow pktl avg"],
    "packet_length_std": ["packet length std", "pkt_len_std", "flow pktl std"],
    "packet_length_min": ["packet length min", "pkt_len_min", "flow pktl min"],
    "packet_length_max": ["packet length max", "pkt_len_max", "flow pktl max"],
    "flow_bytes_per_sec": ["flow bytes/s", "flow byts/s", "sload", "dload", "rate"],
    "flow_packets_per_sec": ["flow packets/s", "flow pkts/s"],
    "fwd_iat_mean": ["fwd iat mean", "fwd_iat_mean"],
    "bwd_iat_mean": ["bwd iat mean", "bwd_iat_mean"],
    "fwd_iat_std": ["fwd iat std", "fwd_iat_std"],
    "bwd_iat_std": ["bwd iat std", "bwd_iat_std"],
    "fin_flag_count": ["fin flag count", "fin_flag_cnt"],
    "syn_flag_count": ["syn flag count", "syn_flag_cnt"],
    "rst_flag_count": ["rst flag count", "rst_flag_cnt"],
    "psh_flag_count": ["psh flag count", "psh_flag_cnt"],
    "ack_flag_count": ["ack flag count", "ack_flag_cnt"],
    "urg_flag_count": ["urg flag count", "urg_flag_cnt"],
    "cwe_flag_count": ["cwe flag count", "cwe_flag_count"],
    "ece_flag_count": ["ece flag count", "ece_flag_count"],
    "fwd_header_length": ["fwd header length", "fwd header len", "fwd_header_size"],
    "bwd_header_length": ["bwd header length", "bwd header len", "bwd_header_size"],
    "active_mean": ["active mean", "active.avg", "active_mean"],
    "active_std": ["active std", "active.std", "active_std"],
    "idle_mean": ["idle mean", "idle.avg", "idle_mean"],
    "idle_std": ["idle std", "idle.std", "idle_std"],
    "bytes_per_packet_ratio": ["bytes_per_packet_ratio"],
    "down_up_ratio": ["down/up ratio", "down_up_ratio"],
    "avg_fwd_segment_size": ["avg fwd segment size", "fwd seg size avg"],
    "avg_bwd_segment_size": ["avg bwd segment size", "bwd seg size avg"],
    "subflow_fwd_bytes": ["subflow fwd bytes"],
    "subflow_bwd_bytes": ["subflow bwd bytes"],
    "bulk_rate_fwd": ["fwd bulk rate avg", "bulk_rate_fwd"],
    "bulk_rate_bwd": ["bwd bulk rate avg", "bulk_rate_bwd"],
}

NORMAL_VALUES = {
    "0",
    "benign",
    "normal",
    "legitimate",
    "background",
    "none",
    "no",
    "false",
}

KDD_ATTACK_FAMILIES = {
    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "satan": "Probe",
    "ftp_write": "R2L",
    "guess_passwd": "R2L",
    "imap": "R2L",
    "multihop": "R2L",
    "phf": "R2L",
    "spy": "R2L",
    "warezclient": "R2L",
    "warezmaster": "R2L",
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "rootkit": "U2R",
}


def clean_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def clean_label(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip().strip(".")
    return re.sub(r"\s+", "_", text)


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get("datasets", {})


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def discover_files(cfg: dict) -> list[Path]:
    base = resolve_path(cfg["local_dir"])
    files: list[Path] = []
    for pattern in cfg.get("file_patterns", []):
        files.extend(p for p in base.glob(pattern) if p.is_file())
    return sorted(set(files))


def column_lookup(columns: Iterable[object]) -> dict[str, str]:
    return {clean_key(col): str(col) for col in columns}


def resolve_column(df: pd.DataFrame, candidates: Iterable[str] | str | None) -> str | None:
    if not candidates:
        return None
    if isinstance(candidates, str):
        candidates = [candidates]
    lookup = column_lookup(df.columns)
    for candidate in candidates:
        exact = str(candidate)
        if exact in df.columns:
            return exact
        key = clean_key(candidate)
        if key in lookup:
            return lookup[key]
    return None


def numeric_series(df: pd.DataFrame, aliases: list[str]) -> pd.Series:
    col = resolve_column(df, aliases)
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").astype("float64")


def string_series(df: pd.DataFrame, aliases: list[str], default: str | None = None) -> pd.Series:
    col = resolve_column(df, aliases)
    fill_value = "" if default is None else default
    if col is None:
        return pd.Series(fill_value, index=df.index, dtype="string")
    return df[col].astype("string").fillna(fill_value)


def infer_binary_label(raw: pd.Series) -> pd.Series:
    text = raw.map(clean_label).str.lower()
    numeric = pd.to_numeric(raw, errors="coerce")
    from_numeric = numeric.fillna(0).ne(0)
    from_text = ~text.isin(NORMAL_VALUES) & text.ne("")
    return (from_numeric.where(numeric.notna(), from_text)).astype("int8")


def attack_family(raw_value: object, binary_label: int) -> str:
    raw = clean_label(raw_value)
    low = raw.lower()
    if binary_label == 0 or low in NORMAL_VALUES:
        return "Normal"
    if low in KDD_ATTACK_FAMILIES:
        return KDD_ATTACK_FAMILIES[low]
    if low in {
        "dns",
        "ldap",
        "mssql",
        "netbios",
        "ntp",
        "portmap",
        "snmp",
        "syn",
        "tftp",
        "udp",
        "udplag",
        "udp-lag",
    }:
        return "DDoS"
    if "ddos" in low or "drdos" in low:
        return "DDoS"
    if "dos" in low:
        return "DoS"
    if "bruteforce" in low or "brute" in low or low in {"bfa"}:
        return "BruteForce"
    if "bot" in low or "mirai" in low or "gafgyt" in low or "bashlite" in low:
        return "Botnet"
    if "scan" in low or "recon" in low or "probe" in low:
        return "Reconnaissance"
    if "infil" in low:
        return "Infiltration"
    if "web" in low or "xss" in low or "sql" in low or "injection" in low:
        return "WebAttack"
    if "ransom" in low:
        return "Ransomware"
    if "backdoor" in low:
        return "Backdoor"
    if "mitm" in low:
        return "MITM"
    if "malware" in low or "evil" in low or "sus" in low:
        return "Anomaly"
    return raw or "Attack"


def normalize_chunk(df: pd.DataFrame, key: str, cfg: dict, source_file: Path) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["dataset_source"] = cfg.get("name", key)
    out["source_file"] = str(source_file.relative_to(ROOT))
    out["environment_type"] = cfg.get("environment", "Unknown")

    label_col = resolve_column(df, cfg.get("label_column"))
    attack_col = resolve_column(df, cfg.get("attack_column"))

    if key == "beth":
        label_bits = []
        for candidate in [cfg.get("label_column"), *cfg.get("auxiliary_label_columns", [])]:
            col = resolve_column(df, candidate)
            if col is not None:
                label_bits.append(pd.to_numeric(df[col], errors="coerce").fillna(0).ne(0))
        if label_bits:
            label = pd.concat(label_bits, axis=1).any(axis=1).astype("int8")
        else:
            label = pd.Series(0, index=df.index, dtype="int8")
    elif label_col is not None:
        label = infer_binary_label(df[label_col])
    elif attack_col is not None:
        label = infer_binary_label(df[attack_col])
    else:
        label = pd.Series(0, index=df.index, dtype="int8")

    if attack_col is not None:
        attack_raw = df[attack_col].astype("object").map(clean_label)
    elif label_col is not None:
        attack_raw = df[label_col].astype("object").map(clean_label)
    else:
        attack_raw = pd.Series(np.where(label == 0, "Normal", "Attack"), index=df.index)

    attack_raw = attack_raw.mask(label.eq(0), "Normal")
    out["attack_subcategory"] = attack_raw.astype("string").fillna("Unknown")
    if key == "beth":
        out["attack_category"] = np.where(label.eq(0), "Normal", "Anomaly")
    else:
        out["attack_category"] = [
            attack_family(raw, int(bin_label))
            for raw, bin_label in zip(out["attack_subcategory"], label, strict=False)
        ]
    out["label"] = label.astype("int8")

    for feature in STRING_FEATURES:
        out[feature] = string_series(df, ALIASES[feature])

    if key == "mqttset":
        out["protocol"] = out["protocol"].fillna("MQTT").replace({"0": "MQTT"})

    for feature in NUMERIC_FEATURES:
        out[feature] = numeric_series(df, ALIASES[feature])

    packets = out["total_packets_fwd"].fillna(0) + out["total_packets_bwd"].fillna(0)
    bytes_total = out["src_bytes"].fillna(0) + out["dst_bytes"].fillna(0)
    missing_ratio = out["bytes_per_packet_ratio"].isna() & packets.gt(0)
    out.loc[missing_ratio, "bytes_per_packet_ratio"] = bytes_total[missing_ratio] / packets[missing_ratio]

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out[OUTPUT_COLUMNS]


def read_csv_chunks(path: Path, cfg: dict, chunk_size: int) -> Iterable[pd.DataFrame]:
    fmt = cfg.get("format")
    read_kwargs = {
        "chunksize": chunk_size,
        "low_memory": False,
        "encoding": "utf-8",
        "encoding_errors": "replace",
        "on_bad_lines": "skip",
    }
    if fmt == "kdd":
        read_kwargs.update({"header": None, "names": [*KDD_FEATURE_COLUMNS, "attack"]})
    elif fmt == "nsl_kdd":
        read_kwargs.update({"header": None, "names": [*KDD_FEATURE_COLUMNS, "attack", "difficulty"]})
    yield from pd.read_csv(path, **read_kwargs)


def read_parquet_chunks(path: Path, chunk_size: int) -> Iterable[pd.DataFrame]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=chunk_size):
        yield batch.to_pandas()


def read_chunks(path: Path, cfg: dict, chunk_size: int) -> Iterable[pd.DataFrame]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        yield from read_parquet_chunks(path, chunk_size)
    else:
        yield from read_csv_chunks(path, cfg, chunk_size)


def table_from_frame(frame: pd.DataFrame) -> pa.Table:
    for col in META_COLUMNS + STRING_FEATURES:
        if col != "label":
            frame[col] = frame[col].astype("string")
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce").fillna(0).astype("int8")
    for col in NUMERIC_FEATURES:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    return pa.Table.from_pandas(frame[OUTPUT_COLUMNS], preserve_index=False)


def write_unified_dataset(
    datasets: dict,
    output_path: Path,
    chunk_size: int,
    max_rows_per_dataset: int | None,
    selected: set[str] | None,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    summary = {
        "output": str(output_path.relative_to(ROOT)),
        "rows": 0,
        "datasets": {},
        "skipped": {},
    }

    try:
        for key, cfg in datasets.items():
            if selected and key not in selected:
                continue

            files = discover_files(cfg)
            if not files:
                summary["skipped"][key] = "No matching files found"
                print(f"[skip] {key}: no files matched")
                continue

            print(f"[dataset] {key}: {len(files)} file(s)")
            rows_for_dataset = 0
            label_counts: Counter[str] = Counter()
            attack_counts: Counter[str] = Counter()
            processed_files: list[str] = []
            file_row_limit = (
                math.ceil(max_rows_per_dataset / len(files))
                if max_rows_per_dataset
                else None
            )

            for path in files:
                if max_rows_per_dataset and rows_for_dataset >= max_rows_per_dataset:
                    break

                print(f"  [read] {path.relative_to(ROOT)}")
                processed_files.append(str(path.relative_to(ROOT)))
                rows_for_file = 0
                for raw_chunk in read_chunks(path, cfg, chunk_size):
                    if raw_chunk.empty:
                        continue
                    if max_rows_per_dataset:
                        remaining = max_rows_per_dataset - rows_for_dataset
                        if file_row_limit:
                            remaining = min(remaining, file_row_limit - rows_for_file)
                        if remaining <= 0:
                            break
                        raw_chunk = raw_chunk.head(remaining)

                    normalized = normalize_chunk(raw_chunk, key, cfg, path)
                    if normalized.empty:
                        continue

                    table = table_from_frame(normalized)
                    if writer is None:
                        writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
                    writer.write_table(table)

                    chunk_rows = len(normalized)
                    rows_for_dataset += chunk_rows
                    rows_for_file += chunk_rows
                    summary["rows"] += chunk_rows
                    label_counts.update(normalized["label"].astype(str))
                    attack_counts.update(normalized["attack_category"].astype(str))

            summary["datasets"][key] = {
                "name": cfg.get("name", key),
                "rows": rows_for_dataset,
                "files": processed_files,
                "label_counts": dict(label_counts),
                "top_attack_categories": dict(attack_counts.most_common(20)),
            }
            print(f"  [ok] {rows_for_dataset:,} rows")
    finally:
        if writer is not None:
            writer.close()

    if summary["rows"] == 0 and output_path.exists():
        output_path.unlink()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified cybersecurity dataset")
    parser.add_argument(
        "--config",
        default="configs/active_datasets.yaml",
        help="YAML config containing the active datasets",
    )
    parser.add_argument(
        "--output",
        default="data/processed/unified_dataset.parquet",
        help="Output Parquet path",
    )
    parser.add_argument(
        "--summary-output",
        default="data/processed/preprocessing_summary.json",
        help="Output JSON summary path",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="Rows per read/write batch",
    )
    parser.add_argument(
        "--max-rows-per-dataset",
        type=int,
        default=50_000,
        help="Limit rows per dataset. Use 0 for no limit.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Process all rows from all selected datasets",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Optional dataset keys to process from the config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    output_path = resolve_path(args.output)
    summary_path = resolve_path(args.summary_output)
    max_rows = None if args.full or args.max_rows_per_dataset == 0 else args.max_rows_per_dataset
    selected = set(args.datasets) if args.datasets else None

    datasets = load_config(config_path)
    if not datasets:
        raise SystemExit(f"No datasets found in {config_path}")

    if selected:
        missing = selected - set(datasets)
        if missing:
            raise SystemExit(f"Unknown dataset key(s): {', '.join(sorted(missing))}")

    print(f"Config : {config_path.relative_to(ROOT)}")
    print(f"Output : {output_path.relative_to(ROOT)}")
    print(f"Rows   : {'full' if max_rows is None else f'max {max_rows:,} per dataset'}")

    summary = write_unified_dataset(
        datasets=datasets,
        output_path=output_path,
        chunk_size=args.chunk_size,
        max_rows_per_dataset=max_rows,
        selected=selected,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"Done. Wrote {summary['rows']:,} rows to {output_path.relative_to(ROOT)}")
    print(f"Summary written to {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
