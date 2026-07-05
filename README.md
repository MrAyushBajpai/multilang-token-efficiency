# multilang-token-efficiency

**Does Language Choice Affect Token Efficiency Across Task Types? A Cross-Task Study on Open-Weight LLMs**

A reproducible experiment measuring how prompt language (11 languages, see below) affects **token consumption**, **task accuracy**, and **cost-efficiency** across three task types — math reasoning, commonsense reasoning, and code generation — using open-weight models served via the Groq API.

This is the code repository for our research project.

---

## Motivation

Prior work (EfficientXLang, 2025) showed that non-English languages reduce token usage in *math reasoning* by 20–40%. The Mythbuster paper (Ren et al., 2026) showed this does **not** hold for *coding* tasks, comparing English and Chinese prompts on closed, proprietary models. Neither line of work has systematically compared the same set of *open-weight* models across multiple task types with a unified cost-efficiency metric — and neither covers more than one or two non-English languages in depth.

We fill that gap by testing five open-weight models across three task types and ten non-English languages chosen specifically to contrast high- and low-fertility tokenization.

---

## Models Tested

| Key | Model | Parameters |
|-----|-------|------------|
| `llama3.3-70b` | `llama-3.3-70b-versatile` | 70B |
| `llama4-scout` | `meta-llama/llama-4-scout-17b-16e-instruct` | 17B (newest architecture) |
| `llama3.1-8b` | `llama-3.1-8b-instant` | 8B (small/fast baseline) |
| `gpt-oss-20b` | `openai/gpt-oss-20b` | 20B (OpenAI open-weight) |
| `gpt-oss-120b` | `openai/gpt-oss-120b` | 120B (OpenAI open-weight) |

All five are non-reasoning models by default, which avoids the token-count/reasoning-overhead confound described below.

> **A note on Qwen3-32B:** Qwen3-32B is a reasoning model on Groq. Even with reasoning disabled, it buffers chain-of-thought tokens internally and routinely hits the free-tier token-per-minute limit, truncating at any completion cap below ~8k tokens. We exclude it from the default run to avoid confounding token counts with reasoning overhead. It can be re-enabled in `run_experiment.py` for anyone with a paid Groq key and a larger token budget.

## Languages

Eleven languages total, English plus ten chosen to span the range of tokenizer "fertility" (how many tokens a language costs relative to English):

**Higher-fertility / typologically distant from English** (expected to cost *more* tokens):
- Chinese (`zh`) — logographic script, high fertility in BPE tokenizers
- Arabic (`ar`) — root-and-pattern morphology
- Hindi (`hi`) — Devanagari script, moderate–high fertility
- Finnish (`fi`) — agglutinative, 15 grammatical cases
- Korean (`ko`) — agglutinative, syllabic Hangul
- Swahili (`sw`) — Bantu agglutinative, low-resource in most pre-training corpora

**Lower-fertility / typologically closer to English** (expected to cost ~the same):
- Spanish (`es`)
- Turkish (`tr`)
- German (`de`) — compound words inflate token counts slightly
- French (`fr`)

English (`en`) is the baseline for all ratios.

## Tasks

| Task | Dataset | Metric |
|------|---------|--------|
| Math reasoning | MATH500 subset / fallback bank | Exact numeric answer match |
| Commonsense | ARC-Easy subset / fallback bank | Multiple-choice accuracy |
| Code generation | MBPP subset / fallback bank | Real test-case execution (asserts run in an isolated subprocess); falls back to a heuristic (runs cleanly + entry-point name present) only for the hardcoded fallback bank, which predates the MBPP test cases |

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
# Edit .env and add your Groq API key(s — see .env.example for the expected
# variable names), free at https://console.groq.com
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
  --models llama3.3-70b \
  --tasks math \
  --languages en zh \
  --n_samples 5
```

### Full experiment
```bash
python run_experiment.py --n_samples 500
```

> **Rate limits:** Groq's free tier allows ~30 req/min and 6,000 tokens/min.
> A full run (5 models × 3 tasks × 11 languages × 500 samples = 82,500 calls)
> is a large-scale run and can take on the order of days on the free tier;
> plan to run it in stages. Runs are resumable — the script tracks progress
> in `results/run_state.json`, so `DONE` cells are skipped and
> partially-completed cells pick up where they left off if interrupted.

### Run a subset to stay within limits
```bash
python run_experiment.py --models llama3.1-8b gpt-oss-20b --n_samples 100
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
| **Ceff** | Expected cost per successful task = avg cost per attempt / resolution rate (from Mythbuster, 2026) |
| **Ceff ratio** | Ceff in language X / Ceff in English |

---

## Reproducing with HuggingFace Datasets (optional)

```bash
pip install datasets
# datasets.py will auto-detect and load from HF if available
```

Without `datasets`, the experiment uses a built-in fallback bank of 20 problems per task, which is sufficient for a proof-of-concept run.

---

## Notes on Groq API

- Token counts are available from `response.usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`)
- Temperature is fixed at **0.0** across all runs for reproducibility
- The `usage` object also returns `prompt_time`, `completion_time`, and `queue_time` in seconds

---

## Credit

If you use this code, please give credit to the author.

---

## Related Work

- EfficientXLang (Ahuja et al., 2025) — multilingual math reasoning, cross-lingual token efficiency
- Mythbuster (Ren et al., 2026) — Chinese vs. English token cost and problem-solving rate for "vibe coding" tasks on closed models (MiniMax-2.7, GPT-5.4-mini, GLM-5)
- PolyMath (2025) — multilingual mathematical reasoning benchmark; includes a preliminary analysis of how "thinking length" varies by language across reasoning and non-reasoning LLMs, noting multilingual reasoning efficiency remains underexplored
- Petrov et al. (2023) — tokenizer fairness across languages
