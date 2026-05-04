"""
Ablation: Jaccard similarity threshold sweep.

Re-runs the Jaccard matcher across a range of threshold values on an existing
results CSV and reports how many samples would be classified as correct at each
threshold, as well as the precision/recall relative to the full-pipeline label.

Usage:
    python ablation_jaccard_threshold.py --results path/to/results.csv [--output out.csv]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from validation import _tokenize  # reuse the same tokeniser

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
N_SAMPLES_DEFAULT = 20


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    union = ta | tb
    return len(ta & tb) / len(union) if union else 0.0


def run(results_csv: str, n_samples: int, output_csv: str | None):
    df = pd.read_csv(results_csv)
    print(f"Loaded {len(df)} tasks from {results_csv}")

    # Collect (jaccard_score, pipeline_correct, plannable) per sample
    records = []
    for _, row in df.iterrows():
        canonical = str(row.get("canonical_solution", "") or "")
        for i in range(n_samples):
            generated   = str(row.get(f"LLM Output #{i}", "") or "")
            plannable   = str(row.get(f"LLM Plannable? #{i}", "False")).strip() == "True"
            full_result = str(row.get(f"LLM DEVS Correct? #{i}", "Failure")).strip().lower()
            pipeline_ok = full_result == "success"
            if not generated or generated.lower() in ("nan", "none"):
                continue
            score = _jaccard(generated, canonical) if plannable else 0.0
            records.append({
                "jaccard":      score,
                "plannable":    plannable,
                "pipeline_ok":  pipeline_ok,
            })

    total = len(records)
    pipeline_correct = sum(r["pipeline_ok"] for r in records)
    print(f"\nTotal samples: {total}  |  Pipeline-correct: {pipeline_correct}")

    rows = []
    for t in THRESHOLDS:
        predicted = [r["jaccard"] >= t for r in records]
        tp = sum(p and r["pipeline_ok"] for p, r in zip(predicted, records))
        fp = sum(p and not r["pipeline_ok"] for p, r in zip(predicted, records))
        fn = sum(not p and r["pipeline_ok"] for p, r in zip(predicted, records))
        n_predicted = sum(predicted)
        precision = tp / n_predicted if n_predicted else 0.0
        recall    = tp / pipeline_correct if pipeline_correct else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({
            "threshold":    t,
            "predicted_ok": n_predicted,
            "precision":    round(precision, 4),
            "recall":       round(recall, 4),
            "f1":           round(f1, 4),
        })

    summary = pd.DataFrame(rows)
    print("\n── Jaccard Threshold Sweep ──────────────────────────────────────────")
    print(summary.to_string(index=False))

    if output_csv:
        summary.to_csv(output_csv, index=False)
        print(f"\nSaved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Ablation: Jaccard threshold sweep")
    parser.add_argument("--results", required=True, help="Path to results CSV")
    parser.add_argument("--samples", type=int, default=N_SAMPLES_DEFAULT)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(args.results, args.samples, args.output)


if __name__ == "__main__":
    main()
