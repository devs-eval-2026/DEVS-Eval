"""
Ablation: canonical matching functions compared independently.

For each row in a results CSV, re-applies every matcher from validation.py
against the canonical solution and reports per-matcher accuracy, along with
the full-pipeline result as baseline.

Usage:
    python ablation_matchers.py --results path/to/results.csv [--output out.csv]

Expected CSV columns (produced by eval.py / eval_open.py):
    canonical_solution, LLM Output #0 … #N, LLM DEVS Correct? #0 … #N,
    LLM Plannable? #0 … #N
"""
import argparse
import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np

# Allow importing from ../eval
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from validation import (
    match_exact,
    match_token_set,
    match_jaccard,
    match_line_set,
    match_statements,
    CANONICAL_MATCHERS,
)

MATCHERS = {name: fn for name, fn in CANONICAL_MATCHERS}

N_SAMPLES_DEFAULT = 20


def _iter_samples(df: pd.DataFrame, n_samples: int):
    """Yield (row_idx, canonical, generated, plannable, pipeline_correct) per sample."""
    for idx, row in df.iterrows():
        canonical = str(row.get("canonical_solution", "") or "")
        for i in range(n_samples):
            generated  = str(row.get(f"LLM Output #{i}", "") or "")
            plannable  = str(row.get(f"LLM Plannable? #{i}", "False")).strip() == "True"
            full_result = str(row.get(f"LLM DEVS Correct? #{i}", "Failure")).strip().lower()
            pipeline_ok = full_result == "success"
            if not generated or generated.lower() in ("nan", "none"):
                continue
            yield idx, canonical, generated, plannable, pipeline_ok


def run(results_csv: str, n_samples: int, output_csv: str | None):
    df = pd.read_csv(results_csv)
    print(f"Loaded {len(df)} tasks from {results_csv}")

    counters: dict[str, dict] = {
        name: {"match": 0, "total": 0} for name in MATCHERS
    }
    counters["pipeline_full"] = {"match": 0, "total": 0}

    for _, canonical, generated, plannable, pipeline_ok in _iter_samples(df, n_samples):
        for name, fn in MATCHERS.items():
            if not plannable:
                # Syntax failure — none of the matchers can fire either
                counters[name]["total"] += 1
                continue
            ok, _ = fn(generated, canonical)
            counters[name]["total"] += 1
            if ok:
                counters[name]["match"] += 1

        counters["pipeline_full"]["total"] += 1
        if pipeline_ok:
            counters["pipeline_full"]["match"] += 1

    rows = []
    for name, c in counters.items():
        total = c["total"]
        match = c["match"]
        acc = match / total if total else 0.0
        rows.append({"matcher": name, "correct": match, "total": total, "accuracy": round(acc, 4)})

    summary = pd.DataFrame(rows).sort_values("accuracy", ascending=False)

    print("\n── Matcher Accuracy ─────────────────────────────────")
    print(summary.to_string(index=False))

    if output_csv:
        summary.to_csv(output_csv, index=False)
        print(f"\nSaved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Ablation: matcher functions")
    parser.add_argument("--results", required=True, help="Path to results CSV")
    parser.add_argument("--samples", type=int, default=N_SAMPLES_DEFAULT)
    parser.add_argument("--output", default=None, help="Save summary to CSV")
    args = parser.parse_args()
    run(args.results, args.samples, args.output)


if __name__ == "__main__":
    main()
