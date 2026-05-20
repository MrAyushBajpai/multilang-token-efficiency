"""
metrics.py
----------
Computes per-run and cross-run statistics for the paper's analysis.

Key metrics:
  - accuracy / pass rate
  - mean / median completion tokens
  - token efficiency ratio (vs English baseline)
  - expected cost per successful task (Ceff)
  - latency stats
"""

import json
import math
from pathlib import Path
from typing import List, Dict, Any


# Groq free-tier approximate pricing as of 2025 ($/million tokens)
# Update these if Groq changes pricing
TOKEN_PRICE_PER_M = {
    "qwen3-32b":    {"input": 0.29, "output": 0.39},
    "llama3.3-70b": {"input": 0.59, "output": 0.79},
    "llama4-scout": {"input": 0.11, "output": 0.34},
}


def compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary stats for one run (model x task x language)."""
    if not records:
        return {}

    n = len(records)
    correct = [r for r in records if r.get("correct") is True]
    n_correct = len(correct)

    comp_toks  = [r["completion_tokens"] for r in records]
    total_toks = [r["total_tokens"]      for r in records]
    latencies  = [r["latency_s"]         for r in records]

    accuracy = n_correct / n if n > 0 else 0.0

    model_key = records[0].get("model", "")
    prices    = TOKEN_PRICE_PER_M.get(model_key, {"input": 0.0, "output": 0.0})
    avg_cost_per_attempt = (
        (safe_mean([r["prompt_tokens"]     for r in records]) / 1e6) * prices["input"] +
        (safe_mean([r["completion_tokens"] for r in records]) / 1e6) * prices["output"]
    )
    # Ceff = avg cost per attempt / resolution rate  (from Mythbuster paper)
    ceff = avg_cost_per_attempt / accuracy if accuracy > 0 else float("inf")

    return {
        "n":                       n,
        "n_correct":               n_correct,
        "accuracy":                accuracy,
        "mean_completion_tokens":  safe_mean(comp_toks),
        "median_completion_tokens":safe_median(comp_toks),
        "std_completion_tokens":   safe_std(comp_toks),
        "mean_total_tokens":       safe_mean(total_toks),
        "median_total_tokens":     safe_median(total_toks),
        "mean_latency_s":          safe_mean(latencies),
        "avg_cost_per_attempt_usd":avg_cost_per_attempt,
        "ceff_usd":                ceff,
    }


def compute_efficiency_ratios(all_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Compute token efficiency ratios relative to English baseline.
    all_results keys: "<model>__<task>__<lang>"
    Returns nested dict: ratios[model][task][lang] = {ratio, accuracy_delta}
    """
    ratios = {}

    for run_key, records in all_results.items():
        parts = run_key.split("__")
        if len(parts) != 3:
            continue
        model, task, lang = parts

        m = compute_metrics(records)
        if not m:
            continue

        ratios.setdefault(model, {}).setdefault(task, {})[lang] = m

    # Compute deltas relative to English
    deltas = {}
    for model, tasks in ratios.items():
        deltas[model] = {}
        for task, langs in tasks.items():
            en_metrics = langs.get("en", {})
            if not en_metrics:
                continue
            en_comp   = en_metrics["mean_completion_tokens"]
            en_acc    = en_metrics["accuracy"]
            deltas[model][task] = {}
            for lang, metrics in langs.items():
                comp_ratio  = metrics["mean_completion_tokens"] / en_comp if en_comp else None
                acc_delta   = metrics["accuracy"] - en_acc
                ceff_ratio  = (metrics["ceff_usd"] / en_metrics["ceff_usd"]
                               if en_metrics.get("ceff_usd") and en_metrics["ceff_usd"] > 0
                               else None)
                deltas[model][task][lang] = {
                    **metrics,
                    "completion_token_ratio_vs_en": comp_ratio,
                    "accuracy_delta_vs_en":         acc_delta,
                    "ceff_ratio_vs_en":             ceff_ratio,
                }

    return deltas


def aggregate_results_dir(results_dir: str) -> Dict[str, List[Dict]]:
    """Load all run JSON files from results_dir into a single dict."""
    all_results = {}
    for path in sorted(Path(results_dir).glob("*.json")):
        run_key = path.stem
        with open(path) as f:
            all_results[run_key] = json.load(f)
    return all_results


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_mean(lst):
    return sum(lst) / len(lst) if lst else 0.0

def safe_median(lst):
    if not lst:
        return 0.0
    s = sorted(lst)
    n = len(s)
    return (s[n//2 - 1] + s[n//2]) / 2 if n % 2 == 0 else s[n//2]

def safe_std(lst):
    if len(lst) < 2:
        return 0.0
    m = safe_mean(lst)
    return math.sqrt(sum((x - m)**2 for x in lst) / (len(lst) - 1))