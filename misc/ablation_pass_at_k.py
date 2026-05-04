"""
Ablation: pass@k for k = 1, 2, 5, 10, 20.

Uses the unbiased estimator from Chen et al. (2021):
    pass@k = 1 - C(n-c, k) / C(n, k)

where n = number of samples per task and c = number of correct samples.

Usage:
    python ablation_pass_at_k.py --results path/to/results.csv [--output out.csv]
    python ablation_pass_at_k.py --results-dir path/to/results/ [--output out.csv]

When --results-dir is given, every CSV in the tree is processed and results
are grouped by model (inferred from the parent directory name).
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

K_VALUES = [1, 2, 5, 10, 20]
N_SAMPLES_DEFAULT = 20


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al. 2021)."""
    if k > n:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod([(n - c - i) / (n - i) for i in range(k)]))


def _score_file(csv_path: Path, n_samples: int) -> list[dict]:
    """Return one dict per task row with pass@k for each k."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [skip] {csv_path}: {e}")
        return []

    rows = []
    for idx, row in df.iterrows():
        if isinstance(row.get("prompt"), float) and math.isnan(row.get("prompt", 0)):
            continue

        # Count valid samples and correct samples
        n = 0
        c = 0
        for i in range(n_samples):
            output = row.get(f"LLM Output #{i}", "")
            if isinstance(output, float) and math.isnan(output):
                continue
            if str(output).strip().lower() in ("", "nan", "none"):
                continue
            n += 1
            result = str(row.get(f"LLM DEVS Correct? #{i}", "Failure")).strip().lower()
            if result == "success":
                c += 1

        if n == 0:
            continue

        entry = {
            "file":    csv_path.stem,
            "task_id": row.get("task_id", idx),
            "n":       n,
            "c":       c,
        }
        for k in K_VALUES:
            entry[f"pass@{k}"] = round(_pass_at_k(n, c, k), 4)
        rows.append(entry)
    return rows


def _aggregate(records: list[dict], label: str) -> pd.DataFrame:
    df = pd.DataFrame(records)
    agg = {f"pass@{k}": "mean" for k in K_VALUES}
    agg["n"] = "mean"
    agg["c"] = "sum"
    summary = df.agg(agg).to_frame().T
    summary.insert(0, "group", label)
    summary["tasks"] = len(df)
    return summary


def run_single(results_csv: str, n_samples: int, output_csv: str | None):
    path = Path(results_csv)
    records = _score_file(path, n_samples)
    if not records:
        print("No valid rows found.")
        return

    per_task = pd.DataFrame(records)
    summary  = _aggregate(records, path.stem)

    print(f"\n── pass@k per task ({path.name}) ─────────────────────────────────")
    print(per_task[[c for c in per_task.columns if c != "file"]].to_string(index=False))
    print(f"\n── Aggregate ─────────────────────────────────────────────────────")
    print(summary.to_string(index=False))

    if output_csv:
        per_task.to_csv(output_csv, index=False)
        summary.to_csv(output_csv.replace(".csv", "_aggregate.csv"), index=False)
        print(f"\nSaved to {output_csv}")


def run_dir(results_dir: str, n_samples: int, output_csv: str | None):
    root = Path(results_dir)
    model_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not model_dirs:
        model_dirs = [root]

    all_summaries = []
    for mdir in model_dirs:
        model_name = mdir.name
        all_records = []
        for csv_path in sorted(mdir.rglob("*.csv")):
            all_records.extend(_score_file(csv_path, n_samples))
        if not all_records:
            continue
        s = _aggregate(all_records, model_name)
        all_summaries.append(s)
        print(f"[{model_name}] tasks={s['tasks'].item()}  "
              + "  ".join(f"pass@{k}={s[f'pass@{k}'].item():.3f}" for k in K_VALUES))

    if all_summaries:
        full = pd.concat(all_summaries, ignore_index=True)
        print("\n── Full comparison ────────────────────────────────────────────────")
        print(full.to_string(index=False))
        if output_csv:
            full.to_csv(output_csv, index=False)
            print(f"\nSaved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Ablation: pass@k analysis")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--results",     help="Single results CSV")
    group.add_argument("--results-dir", help="Directory tree of results CSVs")
    parser.add_argument("--samples", type=int, default=N_SAMPLES_DEFAULT)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.results:
        run_single(args.results, args.samples, args.output)
    else:
        run_dir(args.results_dir, args.samples, args.output)


if __name__ == "__main__":
    main()
