"""
Ablation: LLM judge analysis — four independent modes.

  agreement   — post-hoc: how often does the judge agree with canonical matchers?
                No API calls required; works from existing results CSVs.

  models      — compare GPT-5.4 vs GPT-5.3-codex vs Gemini as judge on a
                sample of rows from a results CSV.

  templates   — compare judge WITHOUT reference (llm_as_judge_single) vs
                judge WITH reference (llm_as_judge_reference) from metrics.py.

  consistency — call the same judge model N times on the same sample and
                measure verdict variance (Fleiss κ proxy).

Usage:
    python ablation_judge.py agreement  --results path/to/results.csv [--output out.csv]
    python ablation_judge.py models     --results path/to/results.csv --rows 30 [--output out.csv]
    python ablation_judge.py templates  --results path/to/results.csv --rows 30 [--output out.csv]
    python ablation_judge.py consistency --results path/to/results.csv --rows 20 --repeats 5 [--output out.csv]
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

# Allow importing from ../eval
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from validation import canonical_match, SEMANTIC_JUDGE_SYSTEM
import metrics as metrics_mod

N_SAMPLES_DEFAULT = 20


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_verdict(text: str) -> str:
    """Return 'Correct', 'Incorrect', or 'Unknown' from judge response."""
    import re
    m = re.search(r"Rating:\s*(Correct|Incorrect)", text, re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    low = text.lower()
    if "rating: correct" in low:
        return "Correct"
    if "rating: incorrect" in low:
        return "Incorrect"
    return "Unknown"


def _iter_valid_samples(df: pd.DataFrame, n_samples: int, max_rows: int | None = None):
    """Yield dicts with task metadata and per-sample fields."""
    count = 0
    for idx, row in df.iterrows():
        if isinstance(row.get("prompt"), float) and math.isnan(row.get("prompt", 0)):
            continue
        canonical = str(row.get("canonical_solution", "") or "")
        prompt    = str(row.get("prompt", "") or "")
        for i in range(n_samples):
            generated  = str(row.get(f"LLM Output #{i}", "") or "")
            if not generated or generated.lower() in ("nan", "none"):
                continue
            plannable  = str(row.get(f"LLM Plannable? #{i}", "False")).strip() == "True"
            full_ok    = str(row.get(f"LLM DEVS Correct? #{i}", "Failure")).strip().lower() == "success"
            yield {
                "task_id":   row.get("task_id", idx),
                "sample":    i,
                "prompt":    prompt,
                "canonical": canonical,
                "generated": generated,
                "plannable": plannable,
                "full_ok":   full_ok,
            }
            count += 1
            if max_rows and count >= max_rows:
                return


def _build_judge_client():
    """Return an OpenAI client, prompting for key if needed."""
    from openai import OpenAI
    if "OPENAI_API_KEY" not in os.environ:
        import getpass
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key: ")
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _call_openai(client, model: str, system: str, user: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=512,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(15 * (attempt + 1))
                continue
            return f"[error: {e}]"
    return "[error: max retries]"


def _call_gemini(system: str, user: str) -> str:
    import google.generativeai as genai
    if "GOOGLE_API_KEY" not in os.environ:
        import getpass
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Google API Key: ")
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-2.5-flash")
    try:
        resp = model.generate_content(system + "\n\n" + user)
        return resp.candidates[0].content.parts[0].text
    except Exception as e:
        return f"[error: {e}]"


# ── Mode A: agreement ─────────────────────────────────────────────────────────

def mode_agreement(results_csv: str, n_samples: int, output_csv: str | None):
    """
    Post-hoc agreement between judge result (LLM DEVS Correct?) and
    canonical matchers, for every plannable sample in the CSV.
    """
    df = pd.read_csv(results_csv)
    print(f"Loaded {len(df)} tasks from {results_csv}")

    rows = []
    for s in _iter_valid_samples(df, n_samples):
        if not s["plannable"]:
            continue
        any_match, matcher_results = canonical_match(s["generated"], s["canonical"])
        rows.append({
            "task_id":          s["task_id"],
            "sample":           s["sample"],
            "canonical_match":  any_match,
            "judge_correct":    s["full_ok"],
            "agree":            any_match == s["full_ok"],
            **{f"m_{k}": v["match"] for k, v in matcher_results.items()},
        })

    if not rows:
        print("No plannable samples found.")
        return

    result = pd.DataFrame(rows)
    total  = len(result)
    agree  = result["agree"].sum()

    # Agreement breakdown
    tp = ((result["canonical_match"]) & (result["judge_correct"])).sum()
    tn = ((~result["canonical_match"]) & (~result["judge_correct"])).sum()
    fp = ((result["canonical_match"]) & (~result["judge_correct"])).sum()
    fn = ((~result["canonical_match"]) & (result["judge_correct"])).sum()

    print(f"\n── Judge ↔ Canonical Matcher Agreement ────────────────────────────")
    print(f"  Total plannable samples : {total}")
    print(f"  Agreement               : {agree}/{total}  ({agree/total*100:.1f}%)")
    print(f"  True positive  (both OK): {tp}")
    print(f"  True negative  (both KO): {tn}")
    print(f"  False positive (match only): {fp}  ← judge says wrong, matcher says ok")
    print(f"  False negative (judge only): {fn}  ← judge says ok, matcher says wrong")

    # Per-matcher agreement with judge
    print("\n── Per-matcher agreement with judge ────────────────────────────────")
    for col in [c for c in result.columns if c.startswith("m_")]:
        name = col[2:]
        a = (result[col] == result["judge_correct"]).mean()
        print(f"  {name:<14} {a*100:.1f}%")

    if output_csv:
        result.to_csv(output_csv, index=False)
        print(f"\nSaved to {output_csv}")


# ── Mode B: models ─────────────────────────────────────────────────────────────

JUDGE_MODELS = {
    "gpt-5.4":       ("openai", "gpt-5.4"),
    "gpt-5.3-codex": ("openai", "gpt-5.3-codex"),
    "gemini-2.5-flash": ("gemini", None),
}


def mode_models(results_csv: str, n_samples: int, max_rows: int, output_csv: str | None):
    """Compare GPT-5.4, GPT-5.3-codex and Gemini as judge."""
    df = pd.read_csv(results_csv)
    samples = list(_iter_valid_samples(df, n_samples, max_rows))
    print(f"Evaluating {len(samples)} samples with {len(JUDGE_MODELS)} judge models…")

    client = _build_judge_client()
    rows = []

    for i, s in enumerate(samples):
        row = {
            "task_id":   s["task_id"],
            "sample":    s["sample"],
            "ground_truth": s["full_ok"],
        }
        judge_prompt = metrics_mod.llm_as_judge_single(s["prompt"], s["generated"])
        for label, (backend, model_id) in JUDGE_MODELS.items():
            if backend == "openai":
                raw = _call_openai(client, model_id, SEMANTIC_JUDGE_SYSTEM, judge_prompt)
            else:
                raw = _call_gemini(SEMANTIC_JUDGE_SYSTEM, judge_prompt)
            verdict = _parse_verdict(raw)
            row[f"verdict_{label}"] = verdict
            row[f"correct_{label}"] = verdict == "Correct"
            time.sleep(0.3)
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(samples)} done")

    result = pd.DataFrame(rows)

    print("\n── Judge model accuracy vs ground truth ────────────────────────────")
    for label in JUDGE_MODELS:
        col = f"correct_{label}"
        if col not in result.columns:
            continue
        gt_match = (result[col] == result["ground_truth"]).mean()
        correct_rate = result[col].mean()
        print(f"  {label:<20}  accuracy={gt_match*100:.1f}%  predicted_correct={correct_rate*100:.1f}%")

    # Inter-judge agreement (pairwise)
    labels = list(JUDGE_MODELS.keys())
    print("\n── Pairwise inter-judge agreement ──────────────────────────────────")
    for i, a in enumerate(labels):
        for b in labels[i+1:]:
            ca, cb = f"correct_{a}", f"correct_{b}"
            if ca in result.columns and cb in result.columns:
                agree = (result[ca] == result[cb]).mean()
                print(f"  {a} ↔ {b}: {agree*100:.1f}%")

    if output_csv:
        result.to_csv(output_csv, index=False)
        print(f"\nSaved to {output_csv}")


# ── Mode C: templates ─────────────────────────────────────────────────────────

def mode_templates(results_csv: str, n_samples: int, max_rows: int, output_csv: str | None):
    """
    Compare judge WITHOUT reference (llm_as_judge_single) vs
    judge WITH reference (llm_as_judge_reference).
    Uses GPT-5.4 as the judge model for both.
    """
    df = pd.read_csv(results_csv)
    samples = list(_iter_valid_samples(df, n_samples, max_rows))
    print(f"Evaluating {len(samples)} samples (single vs reference template)…")

    client = _build_judge_client()
    rows = []

    for i, s in enumerate(samples):
        prompt_single = metrics_mod.llm_as_judge_single(s["prompt"], s["generated"])
        prompt_ref    = metrics_mod.llm_as_judge_reference(s["canonical"], s["generated"])

        raw_single = _call_openai(client, "gpt-5.4", "", prompt_single)
        raw_ref    = _call_openai(client, "gpt-5.4", "", prompt_ref)

        v_single = _parse_verdict(raw_single)
        v_ref    = _parse_verdict(raw_ref)

        rows.append({
            "task_id":       s["task_id"],
            "sample":        s["sample"],
            "ground_truth":  s["full_ok"],
            "verdict_single":  v_single,
            "verdict_ref":     v_ref,
            "correct_single":  v_single == "Correct",
            "correct_ref":     v_ref == "Correct",
            "agree_templates": v_single == v_ref,
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(samples)} done")
        time.sleep(0.3)

    result = pd.DataFrame(rows)

    gt = result["ground_truth"]
    acc_single = (result["correct_single"] == gt).mean()
    acc_ref    = (result["correct_ref"]    == gt).mean()
    agreement  = result["agree_templates"].mean()

    print("\n── Template comparison (GPT-5.4 judge) ─────────────────────────────")
    print(f"  single (no ref) accuracy : {acc_single*100:.1f}%")
    print(f"  reference accuracy       : {acc_ref*100:.1f}%")
    print(f"  inter-template agreement : {agreement*100:.1f}%")

    if output_csv:
        result.to_csv(output_csv, index=False)
        print(f"\nSaved to {output_csv}")


# ── Mode D: consistency ───────────────────────────────────────────────────────

def mode_consistency(results_csv: str, n_samples: int, max_rows: int,
                     repeats: int, output_csv: str | None):
    """
    Call the same judge (GPT-5.4, single template) `repeats` times on
    the same sample and measure verdict variance.
    """
    df = pd.read_csv(results_csv)
    samples = list(_iter_valid_samples(df, n_samples, max_rows))
    print(f"Testing judge consistency: {len(samples)} samples × {repeats} repeats…")

    client = _build_judge_client()
    rows = []

    for i, s in enumerate(samples):
        judge_prompt = metrics_mod.llm_as_judge_single(s["prompt"], s["generated"])
        verdicts = []
        for _ in range(repeats):
            raw = _call_openai(client, "gpt-5.4", "", judge_prompt)
            verdicts.append(_parse_verdict(raw))
            time.sleep(0.3)

        n_correct   = sum(v == "Correct" for v in verdicts)
        consistent  = len(set(verdicts)) == 1
        majority    = "Correct" if n_correct > repeats / 2 else "Incorrect"
        rows.append({
            "task_id":      s["task_id"],
            "sample":       s["sample"],
            "ground_truth": s["full_ok"],
            "verdicts":     "|".join(verdicts),
            "n_correct":    n_correct,
            "n_incorrect":  repeats - n_correct,
            "consistent":   consistent,
            "majority":     majority,
            "majority_ok":  majority == ("Correct" if s["full_ok"] else "Incorrect"),
        })
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(samples)} done")

    result = pd.DataFrame(rows)

    consistency_rate = result["consistent"].mean()
    majority_acc     = result["majority_ok"].mean()
    flip_rate        = 1 - consistency_rate

    print(f"\n── Judge consistency (GPT-5.4, {repeats} repeats) ──────────────────")
    print(f"  Samples tested         : {len(result)}")
    print(f"  Always-same verdict    : {consistency_rate*100:.1f}%")
    print(f"  Flip rate              : {flip_rate*100:.1f}%")
    print(f"  Majority-vote accuracy : {majority_acc*100:.1f}%")

    # Distribution of disagreements
    flip_df = result[~result["consistent"]]
    if not flip_df.empty:
        print(f"\n  Samples that flipped ({len(flip_df)}):")
        print(flip_df[["task_id", "sample", "verdicts", "ground_truth"]].to_string(index=False))

    if output_csv:
        result.to_csv(output_csv, index=False)
        print(f"\nSaved to {output_csv}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ablation: LLM judge analysis")
    sub = parser.add_subparsers(dest="mode", required=True)

    # agreement
    p_agree = sub.add_parser("agreement", help="Post-hoc judge vs canonical matcher agreement")
    p_agree.add_argument("--results",  required=True)
    p_agree.add_argument("--samples",  type=int, default=N_SAMPLES_DEFAULT)
    p_agree.add_argument("--output",   default=None)

    # models
    p_models = sub.add_parser("models", help="Compare GPT-5.4 / GPT-5.3-codex / Gemini as judge")
    p_models.add_argument("--results", required=True)
    p_models.add_argument("--samples", type=int, default=N_SAMPLES_DEFAULT)
    p_models.add_argument("--rows",    type=int, default=30, help="Max samples to evaluate")
    p_models.add_argument("--output",  default=None)

    # templates
    p_tmpl = sub.add_parser("templates", help="Single template vs reference template")
    p_tmpl.add_argument("--results",   required=True)
    p_tmpl.add_argument("--samples",   type=int, default=N_SAMPLES_DEFAULT)
    p_tmpl.add_argument("--rows",      type=int, default=30)
    p_tmpl.add_argument("--output",    default=None)

    # consistency
    p_cons = sub.add_parser("consistency", help="Verdict variance across repeated judge calls")
    p_cons.add_argument("--results",   required=True)
    p_cons.add_argument("--samples",   type=int, default=N_SAMPLES_DEFAULT)
    p_cons.add_argument("--rows",      type=int, default=20)
    p_cons.add_argument("--repeats",   type=int, default=5)
    p_cons.add_argument("--output",    default=None)

    args = parser.parse_args()

    if args.mode == "agreement":
        mode_agreement(args.results, args.samples, args.output)
    elif args.mode == "models":
        mode_models(args.results, args.samples, args.rows, args.output)
    elif args.mode == "templates":
        mode_templates(args.results, args.samples, args.rows, args.output)
    elif args.mode == "consistency":
        mode_consistency(args.results, args.samples, args.rows, args.repeats, args.output)


if __name__ == "__main__":
    main()
