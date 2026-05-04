"""
DEVS-Eval LLM-as-a-Judge
Uses GPT-5.4 (via OpenAI API) to evaluate CoT outputs.
Based on the llm_as_judge_reference template from DEVS-Eval (Fig 8, https://arxiv.org/pdf/2306.05685).

Usage:
    pip install openai pandas tqdm
    export OPENAI_API_KEY=your_key_here
    python llm_judge_devs_eval.py

Output:
    devs_eval_judge_results.csv  — per-sample verdicts
    devs_eval_judge_summary.csv  — per-task and overall summary
"""

import os
import re
import time
import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_CSV      = "evaluation-dataset-for-data-COT.csv"   # GPT-5.4 CoT data
OUTPUT_SAMPLES = "devs_eval_judge_results.csv"
OUTPUT_SUMMARY = "devs_eval_judge_summary.csv"
MODEL          = "gpt-5.4"
MAX_TOKENS     = 512
N_SAMPLES      = 20          # samples per task
SLEEP_BETWEEN  = 0.5         # seconds between API calls (rate limit)
RESUME         = True        # skip already-evaluated rows if output file exists

# ── Judge template (Fig 8 of https://arxiv.org/pdf/2306.05685, adapted) ──────
JUDGE_TEMPLATE = """Please act as an impartial judge and evaluate the correctness of the DNL/SES code provided by an AI assistant for a DEVS simulation modeling task. You will be given a reference answer (canonical solution) and the assistant's answer. Your job is to evaluate if the assistant's answer is correct or incorrect in comparison to the reference answer.

The code is written in MS4 Me's Declarative Natural Language (DNL) or SES syntax. A solution is correct if it is semantically equivalent to the reference, even if variable names or ordering differ slightly, as long as the DEVS model behavior is preserved.

Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Avoid any position biases. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.

After providing your explanation, output your final verdict by strictly following this format: "Rating: Correct" or "Rating: Incorrect".

[User Question]
{prompt}

[The Start of Reference Answer]
{reference}
[The End of Reference Answer]

[The Start of Assistant's Answer]
{candidate}
[The End of Assistant's Answer]
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'\r', '', str(s))).strip()

def parse_verdict(response_text: str) -> str:
    """Extract 'Correct' or 'Incorrect' from judge response."""
    match = re.search(r'Rating:\s*(Correct|Incorrect)', response_text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    # Fallback: search for explicit keywords
    text_lower = response_text.lower()
    if 'rating: correct' in text_lower:
        return 'Correct'
    if 'rating: incorrect' in text_lower:
        return 'Incorrect'
    return 'Unknown'

def call_judge(client: OpenAI, prompt: str, reference: str, candidate: str) -> dict:
    """Call GPT-5.4 as judge. Returns dict with verdict, explanation, raw response."""
    if not candidate or candidate.lower() in ('nan', 'none', ''):
        return {'verdict': 'Incorrect', 'explanation': 'Empty output', 'raw': ''}

    message_content = JUDGE_TEMPLATE.format(
        prompt=prompt,
        reference=reference,
        candidate=candidate,
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": message_content}],
        )
        raw = response.choices[0].message.content
        verdict = parse_verdict(raw)
        # Extract explanation (everything before "Rating:")
        explanation = re.sub(r'Rating:.*', '', raw, flags=re.IGNORECASE).strip()
        return {'verdict': verdict, 'explanation': explanation[:300], 'raw': raw}
    except Exception as e:
        return {'verdict': 'Error', 'explanation': str(e), 'raw': ''}

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=api_key)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} tasks from {INPUT_CSV}")

    # Load existing results for resuming
    existing = set()
    if RESUME and os.path.exists(OUTPUT_SAMPLES):
        df_existing = pd.read_csv(OUTPUT_SAMPLES)
        existing = set(zip(df_existing['task_id'], df_existing['sample_idx']))
        print(f"Resuming — {len(existing)} samples already evaluated.")

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Tasks"):
        tid      = row['task_id']
        cat      = row['category']
        prompt   = row['prompt']
        canon    = normalize(row['canonical_solution'])

        for i in range(N_SAMPLES):
            if (tid, i) in existing:
                continue

            output    = normalize(str(row.get(f'LLM Output #{i}', '')))
            plannable = str(row.get(f'LLM Plannable? #{i}', 'False')).strip() == 'True'
            correct   = str(row.get(f'LLM Correct? #{i}', 'Failure')).strip().lower() == 'success'

            # Only judge syntactically valid (plannable) outputs — others are trivially incorrect
            if not plannable or not output or output.lower() in ('nan', 'none', ''):
                judge_verdict     = 'Incorrect'
                judge_explanation = 'Not plannable / empty output — skipped judge call.'
                judge_raw         = ''
            else:
                result = call_judge(client, prompt, canon, output)
                judge_verdict     = result['verdict']
                judge_explanation = result['explanation']
                judge_raw         = result['raw']
                time.sleep(SLEEP_BETWEEN)

            results.append({
                'task_id':           tid,
                'category':          cat,
                'sample_idx':        i,
                'prompt':            prompt,
                'canonical':         canon,
                'output':            output[:200],  # truncate for CSV readability
                'ms4me_plannable':   plannable,
                'ms4me_correct':     correct,
                'judge_verdict':     judge_verdict,
                'judge_explanation': judge_explanation,
            })

        # Save incrementally every task
        if results:
            pd.DataFrame(results).to_csv(OUTPUT_SAMPLES, index=False)

    df_res = pd.read_csv(OUTPUT_SAMPLES) if os.path.exists(OUTPUT_SAMPLES) else pd.DataFrame(results)
    print(f"\nTotal sample evaluations: {len(df_res)}")

    # ── Per-task summary ──────────────────────────────────────────────────────
    import numpy as np

    def pass_at_k(n, c, k):
        if n - c < k: return 1.0
        return 1.0 - np.prod([(n - c - i) / (n - i) for i in range(k)])

    summary_rows = []
    for tid, grp in df_res.groupby('task_id'):
        n         = len(grp)
        n_correct_ms4me = grp['ms4me_correct'].sum()
        n_correct_judge = (grp['judge_verdict'] == 'Correct').sum()
        summary_rows.append({
            'task_id':            tid,
            'category':           grp['category'].iloc[0],
            'n_samples':          n,
            'n_correct_ms4me':    n_correct_ms4me,
            'n_correct_judge':    n_correct_judge,
            'pass1_ms4me':        pass_at_k(n, n_correct_ms4me, 1),
            'pass1_judge':        pass_at_k(n, n_correct_judge, 1),
            'pass5_judge':        pass_at_k(n, n_correct_judge, min(5, n)),
            'pass10_judge':       pass_at_k(n, n_correct_judge, min(10, n)),
            'agreement':          (grp['ms4me_correct'] == (grp['judge_verdict'] == 'Correct')).mean(),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUTPUT_SUMMARY, index=False)

    # ── Print final report ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DEVS-EVAL LLM-AS-A-JUDGE RESULTS")
    print("=" * 60)
    print(f"Tasks evaluated:       {len(df_summary)}")
    print(f"Pass@1 (MS4Me):        {df_summary['pass1_ms4me'].mean()*100:.2f}%")
    print(f"Pass@1 (Judge):        {df_summary['pass1_judge'].mean()*100:.2f}%")
    print(f"Pass@5 (Judge):        {df_summary['pass5_judge'].mean()*100:.2f}%")
    print(f"Pass@10 (Judge):       {df_summary['pass10_judge'].mean()*100:.2f}%")
    print(f"MS4Me/Judge Agreement: {df_summary['agreement'].mean()*100:.2f}%")

    print("\nBy category (Pass@k Judge):")
    for cat, g in df_summary.groupby('category'):
        print(f"  {cat:<20} {g['pass1_judge'].mean()*100:.1f}%  (n={len(g)})")

    print(f"\nSaved: {OUTPUT_SAMPLES}")
    print(f"Saved: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()