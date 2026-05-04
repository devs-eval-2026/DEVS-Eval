"""
Ablation: sample-size stability of pass@1.

Takes an existing results CSV (with up to N_SAMPLES_PER_TASK samples per task)
and computes mean pass@1 across all tasks using only the first n samples
for n in {1, 2, 3, 5, 8, 10, 15, 20}.

This shows how many samples are required before the pass@1 estimate stabilises,
informing the cost-vs-accuracy trade-off in future evaluations.

Usage:
    python ablation_sample_size.py --results path/to/results.csv [--output out.csv]
    python ablation_sample_size.py --results-dir path/to/results/ [--output out.csv]
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_SIZES = [1, 2, 3, 5, 8, 10, 15, 20]
N_SAMPLES_MAX = 20


def _pass_at_k(n: int, c: int, k: int = 1) -> float:
    if k > n:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod([(n - c - i) / (n - i) for i in range(k)]))


def _correct_flags(row: pd.Series, n_samples: int) -> list[bool]:
    """Return list of per-sample correctness flags (up to n_samples)."""
    flags = []
    for i in range(n_samples):
        output = row.get(f"LLM Output #{i}", "")
        if isinstance(output, float) and math.isnan(output):
            continue
        if str(output).strip().lower() in ("", "nan", "none"):
            continue
        result = str(row.get(f"LLM DEVS Correct? #{i}", "Failure")).strip().lower()
        flags.append(result == "success")
    return flags


def _score_csv(csv_path: Path, max_samples: int) -> list[dict]:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [skip] {csv_path}: {e}")
        return []

    rows = []
    for idx, row in df.iterrows():
        if isinstance(row.get("prompt"), float) and math.isnan(row.get("prompt", 0)):
            continue
        flags = _correct_flags(row, max_samples)
        if not flags:
            continue
        rows.append({"task_id": row.get("task_id", idx), "flags": flags})
    return rows


def _stability_table(records: list[dict], label: str) -> pd.DataFrame:
    result_rows = []
    for n in SAMPLE_SIZES:
        pass1_vals = []
        for r in records:
            subset = r["flags"][:n]
            actual_n = len(subset)
            if actual_n == 0:
                continue
            c = sum(subset)
            pass1_vals.append(_pass_at_k(actual_n, c, 1))
        mean_pass1 = float(np.mean(pass1_vals)) if pass1_vals else float("nan")
        std_pass1  = float(np.std(pass1_vals))  if pass1_vals else float("nan")
        result_rows.append({
            "group":       label,
            "n_samples":   n,
            "mean_pass@1": round(mean_pass1, 4),
            "std_pass@1":  round(std_pass1, 4),
            "tasks":       len(pass1_vals),
        })
    return pd.DataFrame(result_rows)


def run_single(results_csv: str, output_csv: str | None):
    path    = Path(results_csv)
    records = _score_csv(path, N_SAMPLES_MAX)
    if not records:
        print("No valid rows found.")
        return

    table = _stability_table(records, path.stem)
    print(f"\n── Sample-size stability ({path.name}) ──────────────────────────────")
    print(table.to_string(index=False))

    if output_csv:
        table.to_csv(output_csv, index=False)
        print(f"\nSaved to {output_csv}")


def run_dir(results_dir: str, output_csv: str | None):
    root       = Path(results_dir)
    model_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not model_dirs:
        model_dirs = [root]

    all_tables = []
    for mdir in model_dirs:
        model_name = mdir.name
        all_records: list[dict] = []
        for csv_path in sorted(mdir.rglob("*.csv")):
            all_records.extend(_score_csv(csv_path, N_SAMPLES_MAX))
        if not all_records:
            continue
        t = _stability_table(all_records, model_name)
        all_tables.append(t)
        print(f"\n── {model_name} ───────────────────────────────────────────────────")
        print(t[["n_samples", "mean_pass@1", "std_pass@1"]].to_string(index=False))

    if all_tables:
        full = pd.concat(all_tables, ignore_index=True)
        if output_csv:
            full.to_csv(output_csv, index=False)
            print(f"\nSaved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Ablation: sample-size stability")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--results",     help="Single results CSV")
    group.add_argument("--results-dir", help="Directory tree of results CSVs")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.results:
        run_single(args.results, args.output)
    else:
        run_dir(args.results_dir, args.output)


if __name__ == "__main__":
    main()
