"""
Regenerate Markdown and LaTeX reports from an evaluation JSON file.

The main evaluation script already writes reports. This wrapper exists so the
project also has a separate report-generator entry point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def write_markdown(report: dict, output: Path) -> None:
    utility = report["utility"]
    privacy = report["privacy"]
    summary = "\n".join(f"- {line}" for line in report["executive_summary"])
    text = f"""# CyberSynth Evaluation Report

## Executive Summary

{summary}

## Validation Snapshot

| Protocol | F1 | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Real only | {utility['real_only']['f1']:.4f} | {utility['real_only']['precision']:.4f} | {utility['real_only']['recall']:.4f} | {utility['real_only']['roc_auc']:.4f} |
| Real + synthetic | {utility['augmented']['f1']:.4f} | {utility['augmented']['precision']:.4f} | {utility['augmented']['recall']:.4f} | {utility['augmented']['roc_auc']:.4f} |

## Privacy

- Exact real row matches: {privacy['exact_real_row_matches']:,}

## Recommendation

{report['recommendation']}
"""
    output.write_text(text, encoding="utf-8")


def write_latex(report: dict, output: Path) -> None:
    utility = report["utility"]
    text = rf"""\documentclass{{article}}
\begin{{document}}
\section*{{CyberSynth Evaluation Report}}
Baseline F1: {utility['real_only']['f1']:.4f}

Real plus generated-data F1: {utility['augmented']['f1']:.4f}

Recommendation: {report['recommendation'].replace('_', r'\_')}
\end{{document}}
"""
    output.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CyberSynth reports")
    parser.add_argument("--results", default="outputs/reports/evaluation_report.json")
    parser.add_argument("--format", default="markdown,latex")
    parser.add_argument("--output", default="outputs/reports")
    args = parser.parse_args()

    results_path = resolve_path(args.results)
    output_dir = resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(results_path.read_text(encoding="utf-8"))

    formats = {item.strip().lower() for item in args.format.split(",")}
    if "markdown" in formats or "md" in formats:
        write_markdown(report, output_dir / "evaluation_report.md")
    if "latex" in formats or "tex" in formats:
        write_latex(report, output_dir / "evaluation_report.tex")


if __name__ == "__main__":
    main()
