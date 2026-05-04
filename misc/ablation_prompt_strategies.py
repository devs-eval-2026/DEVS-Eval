"""
Ablation: prompt enhancement strategy comparison.

Loads one results CSV per strategy (baseline, CoT, FSP, RAG, multi-turn)
from a results directory and reports pass@1, plannable rate, and correct rate
side-by-side for every strategy × model combination found.

Usage:
    python ablation_prompt_strategies.py --results-dir path/to/results/ [--output out.csv]

Directory layout expected (produced by eval.py):
    results/
        <model>/
            evaluation-dataset-for-<task>.csv          ← baseline (no suffix)
            evaluation-dataset-for-<task>-COT.csv
            evaluation-dataset-for-<task>-FSP.csv
            evaluation-dataset-for-<task>-RAG.csv
            evaluation-dataset-for-<task>-multi-turn.csv
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

STRATEGIES = ["", "COT", "FSP", "RAG", "multi-turn"]
STRATEGY_LABELS = {
    "":           "baseline",
    "COT":        "COT",
    "FSP":        "FSP",
    "RAG":        "RAG",
    "multi-turn": "multi-turn",
}
N_SAMPLES_DEFAULT = 20


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod([(n - c - i) / (n - i) for i in range(k)]))


def _csv_suffix(strategy: str) -> str:
    return f"-{strategy}" if strategy else ""


def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _score_df(df: pd.DataFrame, n_samples: int) -> dict:
    """Return aggregate metrics for a single results CSV."""
    total = plannable = correct = 0
    for _, row in df.iterrows():
        if isinstance(row.get("prompt"), float) and math.isnan(row["prompt"]):
            continue
        for i in range(n_samples):
            plan  = str(row.get(f"LLM Plannable? #{i}", "False")).strip() == "True"
            corr  = str(row.get(f"LLM DEVS Correct? #{i}", "Failure")).strip().lower() == "success"
            total    += 1
            plannable += int(plan)
            correct   += int(corr)
    n_tasks = len(df)
    pass1_vals = []
    for _, row in df.iterrows():
        n = sum(
            1 for i in range(n_samples)
            if not (isinstance(row.get(f"LLM Output #{i}"), float)
                    and math.isnan(row.get(f"LLM Output #{i}", float("nan"))))
        )
        c = sum(
            1 for i in range(n_samples)
            if str(row.get(f"LLM DEVS Correct? #{i}", "Failure")).strip().lower() == "success"
        )
        if n > 0:
            pass1_vals.append(_pass_at_k(n, c, 1))
    return {
        "tasks":          n_tasks,
        "samples":        total,
        "plannable_rate": round(plannable / total, 4) if total else 0.0,
        "correct_rate":   round(correct / total, 4) if total else 0.0,
        "pass@1":         round(float(np.mean(pass1_vals)), 4) if pass1_vals else 0.0,
    }


def run(results_dir: str, n_samples: int, output_csv: str | None):
    root = Path(results_dir)
    models = sorted([d.name for d in root.iterdir() if d.is_dir()])
    print(f"Models found: {models}")

    rows = []
    for model in models:
        model_dir = root / model
        # Collect all base CSV names (without strategy suffix)
        base_files = set()
        for f in model_dir.rglob("*.csv"):
            stem = f.stem
            for strat in STRATEGIES:
                if strat and stem.endswith(f"-{strat}"):
                    base_files.add(stem[: -(len(strat) + 1)])
                    break
            else:
                base_files.add(stem)

        for base in sorted(base_files):
            for strat in STRATEGIES:
                suffix = _csv_suffix(strat)
                csv_path = model_dir / f"{base}{suffix}.csv"
                # Also search subdirectories
                if not csv_path.exists():
                    matches = list(model_dir.rglob(f"{base}{suffix}.csv"))
                    csv_path = matches[0] if matches else csv_path

                df = _load_csv(csv_path)
                if df is None:
                    continue
                metrics = _score_df(df, n_samples)
                rows.append({
                    "model":          model,
                    "task_file":      base,
                    "strategy":       STRATEGY_LABELS[strat],
                    **metrics,
                })

    if not rows:
        print("No CSV files found. Check --results-dir path.")
        return

    summary = pd.DataFrame(rows)

    # Pivot: one row per (model, task), columns per strategy
    pivot = summary.pivot_table(
        index=["model", "task_file"],
        columns="strategy",
        values="pass@1",
        aggfunc="mean",
    ).reset_index()

    print("\n── pass@1 by strategy ───────────────────────────────────────────────")
    print(pivot.to_string(index=False))

    print("\n── Full metrics table ───────────────────────────────────────────────")
    print(summary.to_string(index=False))

    if output_csv:
        summary.to_csv(output_csv, index=False)
        pivot.to_csv(output_csv.replace(".csv", "_pivot.csv"), index=False)
        print(f"\nSaved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Ablation: prompt enhancement strategies")
    parser.add_argument("--results-dir", required=True, help="Root results directory")
    parser.add_argument("--samples", type=int, default=N_SAMPLES_DEFAULT)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(args.results_dir, args.samples, args.output)


if __name__ == "__main__":
    main()
