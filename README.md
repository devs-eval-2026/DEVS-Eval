# DEVS-Eval

A benchmark for evaluating the ability of large language models (LLMs) to generate
DEVS simulation models in the declarative natural language interface (DNL) of the
MS4 Me modeling environment.

## Repository Structure

```
DEVS-Eval/
├── data/                   # Dataset files (CSV)
├── eval/                   # Evaluation pipeline scripts
├── prompt_templates/       # Prompt templates for all strategies
│   ├── zero_shot.txt
│   ├── few_shot.txt
│   ├── cot.txt
│   └── multi_turn.txt
├── rag/                    # RAG setup and vector database configuration
├── misc/                   # Ablation — not necessary for validation pipeline
├── dataset.py              # Dataset loading script
└── README.md
```

## Dataset

The dataset is hosted on Hugging Face at:
[https://huggingface.co/datasets/devs-eval-2026/DEVS-Eval](https://huggingface.co/datasets/devs-eval-2026/DEVS-Eval)

It consists of 181 human-curated tasks across 10 DEVS categories, each containing:
- `task_id` — unique identifier
- `category` — DEVS construct category
- `sub_category` — fine-grained construct
- `prompt` — natural language task description
- `canonical_solution` — reference DNL or SES implementation

## Requirements

```bash
pip install -r requirements.txt
```

Required: Python 3.8+, OpenAI API key, Google API key (for Gemini), Replicate API key
(for open-source models), and MS4 Me installed locally for evaluation.

## Reproducing the Evaluation

### 1. Load the dataset

```python
from datasets import load_dataset

dataset = load_dataset("devs-eval-2026/DEVS-Eval")
df = dataset["train"].to_pandas()
```

Or load locally:

```python
import pandas as pd
df = pd.read_csv("data/devs_eval.csv")
```

### 2. Run generation

Set your API keys:

```bash
export OPENAI_API_KEY=your_key
export GOOGLE_API_KEY=your_key
export REPLICATE_API_TOKEN=your_key
```

Run a model with a prompting strategy:

```bash
python eval/run_eval.py \
    --model gpt-5.4 \
    --strategy cot \
    --output results/gpt54_cot.csv
```

Available `--model` options: `gpt-5.4`, `gpt-5.3-codex`, `gemini-2.5-flash`,
`codellama-7b`, `codellama-34b`, `wizardcoder-33b`

Available `--strategy` options: `zero_shot`, `few_shot_2`, `few_shot_4`, `cot`,
`multi_turn`, `rag`

### 3. Run RAG setup

```bash
python rag/llama_index_retriever.py
python eval/run_eval.py --model gpt-5.4 --strategy rag --output results/gpt54_rag.csv
```

### 4. Evaluate results

```bash
python eval/compute_metrics.py --results results/gpt54_cot.csv
```

This outputs pass@1, BLEU, and CodeBERTScore scores per model and category.

### 5. MS4Me 
Semantic evaluation (Phase 2) requires MS4 Me installed locally with a valid license. To set it up:

1. Download and run the MS4 Me `.exe` installer on Windows 7 or higher (64-bit).
2. Obtain a license certificate file from [ms4systems.com](https://ms4systems.com) or by contacting `support@ms4systems.com`.
3. On first launch, the Licensing Wizard will open — click **Next** and browse to your license file.
4. Click **Install License Certificate** and then **Next** to complete activation.
5. If a Model Store login window appears on restart, click **Cancel** — MS4 Me is ready to use.

## Prompting Strategies

All prompt templates are in `prompt_templates/`. See the paper appendix for the
full prompt text used in each strategy.

## License

CC BY 4.0 — see [LICENSE](LICENSE).

## Citation

Citation information will be provided upon acceptance.
