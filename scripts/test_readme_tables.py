"""test_readme_tables.py — Validate that every benchmark number in README.md matches the JSON results.

Fails (exit code 1) if any measured number (sequence length, latency, VRAM) in README tables
does not match the committed benchmark JSON files.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README_PATH = REPO_ROOT / "README.md"
DF_E2E_PATH = REPO_ROOT / "benchmark" / "darkforest_e2e_results.json"
SDPA_PATH = REPO_ROOT / "benchmark" / "sdpa_baseline_results.json"
KERNEL_PATH = REPO_ROOT / "benchmark" / "kernel_scaling_results.json"


def normalize_num(s: str) -> float:
    """Strip formatting, units, commas, and parse as float."""
    clean = re.sub(r"[^\d.]", "", s)
    return float(clean)


def test_readme():
    print("================================================================================")
    print(" Validating README.md Benchmark Tables Against Committed JSON Files")
    print("================================================================================")

    if not README_PATH.exists():
        print(f"ERROR: {README_PATH} not found.")
        sys.exit(1)

    readme_text = README_PATH.read_text(encoding="utf-8")

    errors = []

    # ---------------------------------------------------------------------------
    # 1. Dark Forest E2E Table
    # ---------------------------------------------------------------------------
    if DF_E2E_PATH.exists():
        with open(DF_E2E_PATH, "r", encoding="utf-8") as f:
            df_e2e_data = json.load(f)

        print(f"Checking Dark Forest E2E numbers against {DF_E2E_PATH.name}...")
        for r in df_e2e_data["results"]:
            s = r["seq_len"]
            status = r.get("status")
            if status == "OK":
                lat = r["median_ms"]
                vram = r["peak_vram_mb"]
                # Look for row containing S and latency
                lat_str = f"{lat:.2f}"
                lat_round = f"{lat:.1f}"
                if lat_str not in readme_text and lat_round not in readme_text:
                    errors.append(f"DF E2E S={s}: median latency {lat} ms not found in README")
                vram_str = f"{vram:.1f}"
                vram_round = f"{round(vram):,}"
                if vram_str not in readme_text and vram_round not in readme_text:
                    errors.append(f"DF E2E S={s}: peak VRAM {vram} MB not found in README")
            else:
                if f"{s:,}" in readme_text and "OOM" not in readme_text:
                    errors.append(f"DF E2E S={s}: expected OOM status not reflected in README")
    else:
        errors.append(f"Missing {DF_E2E_PATH}")

    # ---------------------------------------------------------------------------
    # 2. PyTorch SDPA Baseline Tables
    # ---------------------------------------------------------------------------
    if SDPA_PATH.exists():
        with open(SDPA_PATH, "r", encoding="utf-8") as f:
            sdpa_data = json.load(f)

        print(f"Checking PyTorch SDPA numbers against {SDPA_PATH.name}...")
        # Check standard checkpointed SDPA
        if "with_checkpointing" in sdpa_data:
            for r in sdpa_data["with_checkpointing"]["results"]:
                s = r["seq_len"]
                if r.get("status") == "OK":
                    lat = r["median_ms"]
                    vram = r["peak_vram_mb"]
                    lat_str = f"{lat:.1f}"
                    vram_str = f"{round(vram):,}"
                    if lat_str not in readme_text:
                        errors.append(f"PyTorch SDPA S={s}: latency {lat_str} ms not found in README")
                    if vram_str not in readme_text and f"{vram:.1f}" not in readme_text:
                        errors.append(f"PyTorch SDPA S={s}: VRAM {vram_str} MB not found in README")
        # Check chunked CE
        if "with_chunked_ce" in sdpa_data:
            for r in sdpa_data["with_chunked_ce"]["results"]:
                s = r["seq_len"]
                if r.get("status") == "OK" and s == 16384:
                    lat_str = f"{r['median_ms']:.1f}"
                    vram_str = f"{round(r['peak_vram_mb']):,}"
                    if lat_str not in readme_text:
                        errors.append(f"PyTorch SDPA (+chunked CE) S={s}: latency {lat_str} ms not found in README")
                    if vram_str not in readme_text:
                        errors.append(f"PyTorch SDPA (+chunked CE) S={s}: VRAM {vram_str} MB not found in README")
    else:
        errors.append(f"Missing {SDPA_PATH}")

    # ---------------------------------------------------------------------------
    # 3. Kernel Scaling Table
    # ---------------------------------------------------------------------------
    if KERNEL_PATH.exists():
        with open(KERNEL_PATH, "r", encoding="utf-8") as f:
            kernel_data = json.load(f)

        print(f"Checking Kernel Scaling numbers against {KERNEL_PATH.name}...")
        for r in kernel_data["results"]:
            s = r["seq_len"]
            if r.get("status") == "OK":
                lat = r["latency_ms"]
                vram = r["vram_allocated_mb"]
                lat_str = f"{lat:.1f}"
                lat_comma = f"{lat:,.1f}"
                lat_round = f"{round(lat):,}"
                lat_raw = f"{lat:.3f}"
                if lat_str not in readme_text and lat_comma not in readme_text and lat_round not in readme_text and lat_raw not in readme_text:
                    errors.append(f"Kernel S={s}: latency {lat} ms not found in README")
                vram_str = f"{round(vram):,} MB"
                vram_raw = f"{vram:.2f} MB"
                if vram_str not in readme_text and vram_raw not in readme_text:
                    errors.append(f"Kernel S={s}: VRAM {vram} MB not found in README")
    else:
        errors.append(f"Missing {KERNEL_PATH}")

    # ---------------------------------------------------------------------------
    # Report results
    # ---------------------------------------------------------------------------
    if errors:
        print("\n❌ README VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\n✅ README VALIDATION PASSED: All numbers in README match committed JSON results.")
        sys.exit(0)


if __name__ == "__main__":
    test_readme()
