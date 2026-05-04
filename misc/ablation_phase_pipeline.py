"""
Ablation: pipeline phase comparison.

Simulates three pipeline configurations from a results CSV and compares
their effective correct rates:

  Config A — Phase 1 only
      A sample is "correct" if it compiled (LLM Plannable? == True).
      This is the trivial upper-bound for syntax-only evaluation.

  Config B — Phase 1 + Phase 2a (canonical matching, no LLM judge)
      Correct if it compiled AND at least one canonical matcher fires.
      Re-runs all matchers from validation.py post-hoc.

  Config C — Full pipeline (Phase 1 + 2a + 2b)
      Uses the LLM DEVS Correct? column recorded during evaluation.
      This is the ground truth for the complete pipeline.

Usage:
    python ablation_phase_pipeline.py --results path/to/results.csv [--output out.csv]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from validation import canonical_match

N_SAMPLES_DEFAULT = 20


def _pass_at_k(n: int, c: int, k: int = 1) -> float:
    if k > n:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod([(n - c - i) / (n - i) for i in range(k)]))


CONFIGS = ["phase1_only", "phase1+2a", "full_pipeline"]


def run(results_csv: str, n_samples: int, output_csv: str | None):
    df = pd.read_csv(results_csv)
    print(f"Loaded {len(df)} tasks from {results_csv}")

    per_task_rows = []

    for idx, row in df.iterrows():
        if isinstance(row.get("prompt"), float) and math.isnan(row.get("prompt", 0)):
            continue

        canonical = str(row.get("canonical_solution", "") or "")
        counts = {cfg: 0 for cfg in CONFIGS}
        n_valid = 0

        for i in range(n_samples):
            generated  = str(row.get(f"LLM Output #{i}", "") or "")
            if not generated or generated.lower() in ("nan", "none"):
                continue
            n_valid += 1

            plannable  = str(row.get(f"LLM Plannable? #{i}", "False")).strip() == "True"
            full_ok    = str(row.get(f"LLM DEVS Correct? #{i}", "Failure")).strip().lower() == "success"

            # Config A: Phase 1 only
            if plannable:
                counts["phase1_only"] += 1

            # Config B: Phase 1 + 2a
            if plannable:
                any_match, _ = canonical_match(generated, canonical)
                if any_match:
                    counts["phase1+2a"] += 1

            # Config C: full pipeline
            if full_ok:
                counts["full_pipeline"] += 1

        if n_valid == 0:
            continue

        entry = {
            "task_id": row.get("task_id", idx),
            "n":       n_valid,
        }
        for cfg in CONFIGS:
            entry[f"c_{cfg}"]       = counts[cfg]
            entry[f"pass@1_{cfg}"]  = round(_pass_at_k(n_valid, counts[cfg], 1), 4)
        per_task_rows.append(entry)

    if not per_task_rows:
        print("No valid rows found.")
        return

    per_task = pd.DataFrame(per_task_rows)

    # Aggregate
    agg_rows = []
    for cfg in CONFIGS:
        c_col    = f"c_{cfg}"
        pass_col = f"pass@1_{cfg}"
        agg_rows.append({
            "config":          cfg,
            "total_correct":   int(per_task[c_col].sum()),
            "mean_pass@1":     round(per_task[pass_col].mean(), 4),
            "correct_rate":    round(per_task[c_col].sum() / per_task["n"].sum(), 4),
        })
    agg = pd.DataFrame(agg_rows)

    print("\n── Phase Ablation — per task (pass@1) ───────────────────────────────")
    cols = ["task_id", "n"] + [f"pass@1_{c}" for c in CONFIGS]
    print(per_task[cols].to_string(index=False))

    print("\n── Aggregate ────────────────────────────────────────────────────────")
    print(agg.to_string(index=False))

    if output_csv:
        per_task.to_csv(output_csv, index=False)
        agg.to_csv(output_csv.replace(".csv", "_aggregate.csv"), index=False)
        print(f"\nSaved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Ablation: pipeline phase comparison")
    parser.add_argument("--results", required=True, help="Path to results CSV")
    parser.add_argument("--samples", type=int, default=N_SAMPLES_DEFAULT)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(args.results, args.samples, args.output)


if __name__ == "__main__":
    main()
