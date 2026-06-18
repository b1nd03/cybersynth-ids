# -*- coding: utf-8 -*-
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
"""
CyberSynth - Master Dataset Downloader
=======================================
Downloads all 25 cybersecurity IDS datasets defined in configs/datasets.yaml.

Usage:
    python download_datasets.py                  # Download all datasets
    python download_datasets.py --tier 1         # Download Tier 1 only
    python download_datasets.py --dataset nsl_kdd cic_ids2017
    python download_datasets.py --list           # List all datasets
    python download_datasets.py --check          # Check what's already downloaded
"""

import os
import sys
import gzip
import shutil
import hashlib
import argparse
import tarfile
import zipfile
import subprocess
from pathlib import Path
from datetime import datetime

import requests
import yaml
from tqdm import tqdm

#  Colour helpers 
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):    print(f"{GREEN}  [OK]  {msg}{RESET}")
def warn(msg):  print(f"{YELLOW}  [!!]  {msg}{RESET}")
def err(msg):   print(f"{RED}  [ERR] {msg}{RESET}")
def info(msg):  print(f"{CYAN}  [>>]  {msg}{RESET}")
def head(msg):  print(f"\n{BOLD}{CYAN}{'='*60}\n  {msg}\n{'='*60}{RESET}")

ROOT = Path(__file__).parent.resolve()
CONFIG_PATH = ROOT / "configs" / "datasets.yaml"
LOG_PATH = ROOT / "logs" / "download_log.txt"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

#  Load Config 
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["datasets"]

#  Logging 
def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")

#  Progress-bar download 
def download_file(url: str, dest_path: Path, session: requests.Session, timeout=120) -> bool:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = session.get(url, stream=True, timeout=timeout,
                           headers={"User-Agent": "CyberSynth-Downloader/1.0"})
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True,
            desc=f"    {dest_path.name[:45]}", leave=False
        ) as bar:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        return True
    except Exception as e:
        err(f"Download failed [{dest_path.name}]: {e}")
        log(f"FAIL direct {url} -> {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False

#  Extract archives 
def extract(filepath: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = filepath.name.lower()
    try:
        if name.endswith(".zip"):
            with zipfile.ZipFile(filepath, "r") as zf:
                zf.extractall(dest_dir)
            ok(f"Extracted ZIP -> {dest_dir.name}/")
        elif name.endswith(".tar.gz") or name.endswith(".tgz"):
            with tarfile.open(filepath, "r:gz") as tf:
                tf.extractall(dest_dir)
            ok(f"Extracted TAR.GZ -> {dest_dir.name}/")
        elif name.endswith(".gz"):
            out_path = dest_dir / filepath.stem
            with gzip.open(filepath, "rb") as gz, open(out_path, "wb") as out:
                shutil.copyfileobj(gz, out)
            ok(f"Decompressed GZ -> {out_path.name}")
        else:
            info(f"No extraction needed for {filepath.name}")
    except Exception as e:
        warn(f"Extraction warning [{filepath.name}]: {e}")

#  Download methods 

def download_direct(key, cfg, session):
    """Download one or more direct HTTP/FTP URLs."""
    dest_dir = ROOT / cfg["local_dir"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    urls = cfg.get("urls", [])
    success_count = 0
    for url in urls:
        fname = url.split("/")[-1].split("?")[0] or f"{key}_data.bin"
        dest_path = dest_dir / fname
        if dest_path.exists() and dest_path.stat().st_size > 1000:
            ok(f"Already exists: {fname}")
            success_count += 1
            continue
        info(f"Fetching: {url}")
        if download_file(url, dest_path, session):
            ok(f"Downloaded: {fname}")
            log(f"OK direct {url}")
            extract(dest_path, dest_dir)
            success_count += 1
    return success_count == len(urls)


def download_kaggle(key, cfg):
    """Download via Kaggle API (requires ~/.kaggle/kaggle.json)."""
    dest_dir = ROOT / cfg["local_dir"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dataset = cfg["kaggle_dataset"]

    # Check if already downloaded
    existing = list(dest_dir.glob("*.csv")) + list(dest_dir.glob("*.zip")) + list(dest_dir.glob("*.parquet"))
    if existing:
        ok(f"Already downloaded ({len(existing)} files): {dest_dir.name}/")
        return True

    info(f"Kaggle download: {dataset}")
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset,
             "-p", str(dest_dir), "--unzip"],
            capture_output=True, text=True, timeout=1800
        )
        if result.returncode == 0:
            ok(f"Kaggle OK {dataset}")
            log(f"OK kaggle {dataset}")
            return True
        else:
            err(f"Kaggle API error [{dataset}]:\n{result.stderr.strip()}")
            log(f"FAIL kaggle {dataset} -> {result.stderr.strip()}")
            _kaggle_manual_instructions(dataset, dest_dir)
            return False
    except FileNotFoundError:
        err("Kaggle CLI not installed. Run: pip install kaggle")
        _kaggle_manual_instructions(dataset, dest_dir)
        return False
    except Exception as e:
        err(f"Kaggle exception [{dataset}]: {e}")
        log(f"FAIL kaggle {dataset} -> {e}")
        return False


def download_github(key, cfg, session):
    """Download specific files from a GitHub repository."""
    dest_dir = ROOT / cfg["local_dir"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    repo = cfg["github_repo"]
    files = cfg.get("github_files", [])
    base_url = f"https://raw.githubusercontent.com/{repo}/master"
    success_count = 0
    for fname in files:
        dest_path = dest_dir / fname
        if dest_path.exists() and dest_path.stat().st_size > 100:
            ok(f"Already exists: {fname}")
            success_count += 1
            continue
        url = f"{base_url}/{fname}"
        info(f"GitHub file: {fname}")
        if download_file(url, dest_path, session):
            ok(f"Downloaded: {fname}")
            log(f"OK github {url}")
            success_count += 1
    return success_count > 0


def _kaggle_manual_instructions(dataset, dest_dir):
    """Print manual download instructions when Kaggle API fails."""
    warn(f"Manual download needed:")
    print(f"""
    {YELLOW}1. Go to: https://www.kaggle.com/datasets/{dataset}
    2. Click 'Download' (requires Kaggle account)
    3. Extract ZIP contents into: {dest_dir}

    OR setup Kaggle API:
      a) Go to https://www.kaggle.com/settings -> API -> Create New Token
      b) Save kaggle.json to C:\\Users\\<you>\\.kaggle\\kaggle.json
      c) Run this script again{RESET}
    """)

#  CIC manual note 
def _cic_manual_note(name, url, dest_dir):
    warn(f"CIC dataset may require form acceptance at source.")
    print(f"""
    {YELLOW}If download fails:
    1. Go to: {url}
    2. Accept usage agreement if requested
    3. Extract to: {dest_dir}{RESET}
    """)

#  Per-dataset dispatcher 
def download_dataset(key: str, cfg: dict, session: requests.Session) -> bool:
    method = cfg.get("download_method", "direct")
    name   = cfg.get("name", key)
    size   = cfg.get("size_gb", 0)
    head(f"{name}  [{method.upper()}]  ~{size:.1f} GB")

    if method == "direct":
        return download_direct(key, cfg, session)
    elif method == "kaggle":
        return download_kaggle(key, cfg)
    elif method == "github":
        return download_github(key, cfg, session)
    else:
        warn(f"Unknown method '{method}' for {key}")
        return False

#  Check status 
def check_status(datasets: dict):
    head("Dataset Download Status")
    total_size = 0
    for key, cfg in datasets.items():
        dest_dir = ROOT / cfg["local_dir"]
        files = list(dest_dir.rglob("*")) if dest_dir.exists() else []
        data_files = [f for f in files if f.is_file() and f.suffix in
                      (".csv", ".parquet", ".txt", ".json", ".gz", ".pcap", ".log")]
        size_mb = sum(f.stat().st_size for f in data_files) / (1024**2)
        total_size += size_mb
        status = f"{GREEN}{RESET}" if data_files else f"{RED}{RESET}"
        tier   = cfg.get("tier", "?")
        print(f"  {status}  T{tier}  {cfg['name']:<35} {len(data_files):>4} files  {size_mb:>8.1f} MB")
    print(f"\n  Total downloaded: {total_size/1024:.2f} GB")

#  List datasets 
def list_datasets(datasets: dict):
    head("All Registered Datasets (25 total)")
    by_tier = {}
    for key, cfg in datasets.items():
        t = cfg.get("tier", 4)
        by_tier.setdefault(t, []).append((key, cfg))

    tier_names = {
        1: "Tier 1 - Primary / Most Cited",
        2: "Tier 2 - IoT & Edge-Specific",
        3: "Tier 3 - SDN, Cloud & Application",
        4: "Tier 4 - Kaggle Community"
    }
    for t in sorted(by_tier):
        print(f"\n  {BOLD}{tier_names[t]}{RESET}")
        for key, cfg in by_tier[t]:
            method = cfg.get("download_method","?")
            attacks = ", ".join(cfg.get("attack_types",[])[:4])
            print(f"    {key:<30} [{method:<8}]  ~{cfg.get('size_gb',0):.1f}GB  Attacks: {attacks}...")

#  Main 
def main():
    parser = argparse.ArgumentParser(
        description="CyberSynth - Dataset Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--list",    action="store_true", help="List all datasets")
    parser.add_argument("--check",   action="store_true", help="Check download status")
    parser.add_argument("--tier",    type=int,  help="Download specific tier (1-4)")
    parser.add_argument("--dataset", nargs="+", help="Download specific dataset keys")
    parser.add_argument("--skip-large", action="store_true",
                        help="Skip datasets > 2 GB (useful for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded without downloading")
    args = parser.parse_args()

    datasets = load_config()

    if args.list:
        list_datasets(datasets)
        return

    if args.check:
        check_status(datasets)
        return

    #  Filter datasets 
    targets = datasets
    if args.dataset:
        targets = {k: v for k, v in datasets.items() if k in args.dataset}
        if not targets:
            err(f"No datasets matched: {args.dataset}")
            sys.exit(1)
    elif args.tier:
        targets = {k: v for k, v in datasets.items() if v.get("tier") == args.tier}

    if args.skip_large:
        large = {k for k, v in targets.items() if v.get("size_gb", 0) > 2.0}
        if large:
            warn(f"Skipping large datasets (>2GB): {large}")
        targets = {k: v for k, v in targets.items() if v.get("size_gb", 0) <= 2.0}

    total_gb = sum(v.get("size_gb", 0) for v in targets.values())
    print(f"\n{BOLD}CyberSynth Dataset Downloader{RESET}")
    print(f"  Datasets to download : {len(targets)}")
    print(f"  Estimated total size : ~{total_gb:.1f} GB")
    print(f"  Output root          : {ROOT / 'data' / 'raw'}")
    print(f"  Log file             : {LOG_PATH}")

    if args.dry_run:
        head("DRY RUN - would download:")
        for key, cfg in targets.items():
            print(f"  {cfg['name']:<35} [{cfg.get('download_method','?')}]  ~{cfg.get('size_gb',0):.1f}GB")
        return

    print(f"\n  Starting download at {datetime.now().strftime('%H:%M:%S')} ...\n")
    log(f"=== Download session started. Targets: {list(targets.keys())} ===")

    #  Run downloads 
    session = requests.Session()
    session.headers.update({"Accept-Encoding": "gzip, deflate"})

    results = {}
    for key, cfg in targets.items():
        try:
            results[key] = download_dataset(key, cfg, session)
        except KeyboardInterrupt:
            warn("Interrupted by user.")
            break
        except Exception as e:
            err(f"Unexpected error [{key}]: {e}")
            log(f"ERROR {key} -> {e}")
            results[key] = False

    #  Summary 
    head("Download Summary")
    succeeded = [k for k, v in results.items() if v]
    failed    = [k for k, v in results.items() if not v]

    for k in succeeded:
        ok(datasets[k]["name"])
    for k in failed:
        err(f"{datasets[k]['name']}  (see logs/download_log.txt)")

    print(f"\n  Succeeded : {len(succeeded)}/{len(results)}")
    if failed:
        print(f"  Failed    : {len(failed)}/{len(results)}")
        print(f"\n  For failed datasets, see instructions above or check:")
        print(f"  {LOG_PATH}")

    print(f"\n  Next step -> run the preprocessing pipeline:")
    print(f"  python src/ingestion/preprocessor.py --config configs/datasets.yaml\n")

    log(f"=== Session complete. OK={len(succeeded)} FAIL={len(failed)} ===")


if __name__ == "__main__":
    main()
