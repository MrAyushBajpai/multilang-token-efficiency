# multilang-token-efficiency

**Does Language Choice Affect Token Efficiency Across Task Types? A Cross-Task Study on Open-Weight LLMs**

A reproducible experiment measuring how prompt language (English, Chinese, Hindi, Arabic, Spanish, Turkish) affects **token consumption**, **task accuracy**, and **cost-efficiency** across three task types — math reasoning, commonsense reasoning, and code generation — using open-weight models served via the Groq API.

This is the code repository for our IEEE conference paper submission.

---

## Motivation

Prior work (EfficientXLang, 2025) showed that non-English languages reduce token usage in *math reasoning* by 20–40%. The Mythbuster paper (2025) showed this does **not** hold for *coding* tasks. No work has systematically compared the same models across multiple task types with a unified cost-efficiency metric.

We fill that gap.

---

## Models Tested

| Key | Model | Parameters |
|-----|-------|------------|
| `qwen3-32b` | `qwen/qwen3-32b` | 32B (MoE, reasoning) |
| `llama3.3-70b` | `llama-3.3-70b-versatile` | 70B |
| `llama4-scout` | `meta-llama/llama-4-scout-17b-16e-instruct` | 17B |

## Languages

English · Chinese (zh) · Hindi (hi) · Arabic (ar) · Spanish (es) · Turkish (tr)

## Tasks

| Task | Dataset | Metric |
|------|---------|--------|
| Math reasoning | MATH500 subset / fallback bank | Exact answer match |
| Commonsense | ARC-Easy subset / fallback bank | Multiple-choice accuracy |
| Code generation | HumanEval subset / fallback bank | Function presence (manual eval) |

---

## Repo Structure

```
multilang-token-efficiency/
├── run_experiment.py     # Main experiment loop
├── analyze.py            # Analysis, tables, and plots
├── data/
│   └── datasets.py       # Problem loaders (HF + fallback)
├── scripts/
│   ├── metrics.py        # Token ratios, accuracy, Ceff
│   └── logger.py         # JSON + CSV result saving
├── results/              # Auto-created; raw JSONs + summary.csv
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/multilang-token-efficiency
cd multilang-token-efficiency

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your Groq API key (free at https://console.groq.com)
```

---

## Running the Experiment

### Quick test (dry run — no API calls)
```bash
python run_experiment.py --dry_run
```

### Single model, single task, 2 languages (smoke test)
```bash
python run_experiment.py \
  --models qwen3-32b \
  --tasks math \
  --languages en zh \
  --n_samples 5
```

### Full experiment
```bash
python run_experiment.py --n_samples 50
```

> **Rate limits:** Groq's free tier allows ~30 req/min and 6,000 tokens/min.  
> The script adds a 1.5s delay between calls and uses exponential backoff.  
> A full run (3 models × 3 tasks × 6 languages × 50 samples = 2,700 calls) takes roughly **3–4 hours** on the free tier.

### Run a single model to stay within limits
```bash
python run_experiment.py --models llama3.3-70b --n_samples 50
```

---

## Analysis & Plots

After experiment runs are complete:

```bash
python analyze.py --results_dir results --plots_dir results/plots
```

This generates:
- `results/table_token_ratios.csv` — completion token ratios vs English
- `results/table_accuracy.csv` — accuracy per model/task/language
- `results/table_ceff_ratios.csv` — cost-efficiency ratios vs English
- `results/plots/token_bars_*.png` — grouped bar charts
- `results/plots/accuracy_vs_tokens.png` — scatter plot
- `results/plots/efficiency_heatmap.png` — heatmap of token ratios
- `results/plots/ceff_comparison.png` — cost-efficiency comparison

---

## Key Metrics

| Metric | Description |
|--------|-------------|
| **Completion token ratio** | Mean completion tokens in language X / mean in English |
| **Accuracy delta** | Accuracy in language X minus accuracy in English |
| **Ceff** | Expected cost per successful task = avg cost per attempt / resolution rate (from Mythbuster 2025) |
| **Ceff ratio** | Ceff in language X / Ceff in English |

---

## Reproducing with HuggingFace Datasets (optional)

```bash
pip install datasets
# datasets.py will auto-detect and load from HF if available
```

Without `datasets`, the experiment uses a built-in fallback bank of 20 problems per task which is sufficient for a proof-of-concept run.

---

## Notes on Groq API

- Token counts are available from `response.usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`)
- `Qwen3-32B` supports `reasoning_format="parsed"` to separate reasoning from response
- Temperature is fixed at **0.6** across all runs for reproducibility
- The `usage` object also returns `prompt_time` and `completion_time` in seconds

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{yourname2025multilang,
  title     = {Does Language Choice Affect Token Efficiency Across Task Types?},
  author    = {Your Name},
  booktitle = {Proceedings of [IEEE Conference]},
  year      = {2025}
}
```

---

## Related Work

- EfficientXLang (Ahuja et al., 2025) — multilingual math reasoning
- Mythbuster (Ren et al., 2026) — Chinese vs English for coding
- Petrov et al. (2023) — tokenizer fairness across languages
