"""
metrics.py
----------
Computes per-run and cross-run statistics for the experiment.

Key metrics:
  - accuracy / pass rate
  - mean / median / std / p10 / p90 completion tokens
  - token efficiency ratio (vs English baseline)
  - expected cost per successful task (Ceff)
  - latency statistics
  - fertility proxy (tokens / whitespace-split words in response)
  - truncation rate (responses where finish_reason == 'length')
"""

import json
import math
from pathlib import Path
from typing import List, Dict, Any, Optional


# Groq pricing as of June 2026 (USD per million tokens)
# Source: https://groq.com/pricing — verified directly against Groq's
# official pricing page. Update if Groq changes pricing.
# NOTE: keys here MUST match the model_key strings used in run_experiment.py's
# MODELS registry (e.g. "gpt-oss-20b", not "openai/gpt-oss-20b"), since
# compute_metrics() looks up records[0]["model"] which stores the short key.
TOKEN_PRICE_PER_M: Dict[str, Dict[str, float]] = {
    "llama3.3-70b":  {"input": 0.59,  "output": 0.79},
    "llama4-scout":  {"input": 0.11,  "output": 0.34},
    "llama3.1-8b":   {"input": 0.05,  "output": 0.08},
    "gpt-oss-20b":   {"input": 0.075, "output": 0.30},
    "gpt-oss-120b":  {"input": 0.15,  "output": 0.60},
    "llama3.2-3b":   {"input": 0.06,  "output": 0.06},
    "gemma2-9b":     {"input": 0.20,  "output": 0.20},
    "mistral-saba":  {"input": 0.79,  "output": 0.79},
}


def compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute all summary stats for one run (model × task × language).
    Accepts per-question records as produced by run_experiment.py.
    """
    if not records:
        return {}

    n = len(records)
    correct    = [r for r in records if r.get("correct") is True]
    n_correct  = len(correct)
    accuracy   = n_correct / n if n > 0 else 0.0

    comp_toks   = [r["completion_tokens"]   for r in records if "completion_tokens"   in r]
    total_toks  = [r["total_tokens"]        for r in records if "total_tokens"        in r]
    prompt_toks = [r["prompt_tokens"]       for r in records if "prompt_tokens"       in r]
    latencies   = [r["latency_s"]           for r in records if "latency_s"           in r]
    resp_lens   = [r["response_length"]     for r in records if "response_length"     in r]

    truncated = sum(1 for r in records if r.get("finish_reason") == "length")

    # Fertility proxy: completion_tokens / word_count(response)
    fertility_vals = []
    for r in records:
        words = len(r.get("response", "").split())
        if words > 0 and "completion_tokens" in r:
            fertility_vals.append(r["completion_tokens"] / words)

    model_key  = records[0].get("model", "")
    prices     = TOKEN_PRICE_PER_M.get(model_key, {"input": 0.0, "output": 0.0})
    avg_prompt = safe_mean(prompt_toks)
    avg_comp   = safe_mean(comp_toks)

    avg_cost_per_attempt = (
        (avg_prompt / 1e6) * prices["input"] +
        (avg_comp   / 1e6) * prices["output"]
    )
    # Ceff = cost-per-attempt / accuracy  (from pass@1 expected-cost framing)
    ceff = avg_cost_per_attempt / accuracy if accuracy > 0 else float("inf")

    return {
        "n":                          n,
        "n_correct":                  n_correct,
        "accuracy":                   round(accuracy, 6),
        # completion tokens
        "mean_completion_tokens":     safe_mean(comp_toks),
        "median_completion_tokens":   safe_median(comp_toks),
        "std_completion_tokens":      safe_std(comp_toks),
        "p10_completion_tokens":      safe_percentile(comp_toks, 10),
        "p90_completion_tokens":      safe_percentile(comp_toks, 90),
        # total / prompt tokens
        "mean_total_tokens":          safe_mean(total_toks),
        "mean_prompt_tokens":         safe_mean(prompt_toks),
        # latency
        "mean_latency_s":             safe_mean(latencies),
        "median_latency_s":           safe_median(latencies),
        "p90_latency_s":              safe_percentile(latencies, 90),
        # response character length
        "mean_response_chars":        safe_mean(resp_lens),
        # fertility
        "mean_fertility":             safe_mean(fertility_vals),
        # truncation
        "n_truncated":                truncated,
        "truncation_rate":            round(truncated / n, 6) if n > 0 else 0.0,
        # cost
        "avg_cost_per_attempt_usd":   avg_cost_per_attempt,
        "ceff_usd":                   ceff,
    }


def compute_efficiency_ratios(
    all_results: Dict[str, List[Dict]]
) -> Dict[str, Any]:
    """
    Compute per-language ratios relative to the English baseline.
    all_results keys: "<model>__<task>__<lang>"
    Returns nested dict: deltas[model][task][lang]
    """
    raw: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for run_key, records in all_results.items():
        parts = run_key.split("__")
        if len(parts) != 3:
            continue
        model, task, lang = parts
        m = compute_metrics(records)
        if not m:
            continue
        raw.setdefault(model, {}).setdefault(task, {})[lang] = m

    deltas: Dict[str, Any] = {}
    for model, tasks in raw.items():
        deltas[model] = {}
        for task, langs in tasks.items():
            en = langs.get("en", {})
            if not en:
                continue
            en_comp  = en["mean_completion_tokens"]
            en_acc   = en["accuracy"]
            en_ceff  = en.get("ceff_usd", 0) or 0
            deltas[model][task] = {}
            for lang, m in langs.items():
                comp_ratio = m["mean_completion_tokens"] / en_comp if en_comp else None
                acc_delta  = m["accuracy"] - en_acc
                ceff_ratio = (m["ceff_usd"] / en_ceff if en_ceff > 0 else None)
                deltas[model][task][lang] = {
                    **m,
                    "completion_token_ratio_vs_en": comp_ratio,
                    "accuracy_delta_vs_en":         acc_delta,
                    "ceff_ratio_vs_en":             ceff_ratio,
                }

    return deltas


def aggregate_results_dir(results_dir: str) -> Dict[str, List[Dict]]:
    """Load all JSONL run files from results_dir."""
    all_results: Dict[str, List[Dict]] = {}
    for path in sorted(Path(results_dir).glob("*.jsonl")):
        run_key = path.stem
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        all_results[run_key] = records
    return all_results


# ── Helpers ────────────────────────────────────────────────────────────────────

def safe_mean(lst: List) -> float:
    return sum(lst) / len(lst) if lst else 0.0

def safe_median(lst: List) -> float:
    if not lst:
        return 0.0
    s = sorted(lst)
    n = len(s)
    return (s[n // 2 - 1] + s[n // 2]) / 2 if n % 2 == 0 else float(s[n // 2])

def safe_std(lst: List) -> float:
    if len(lst) < 2:
        return 0.0
    m = safe_mean(lst)
    return math.sqrt(sum((x - m) ** 2 for x in lst) / (len(lst) - 1))

def safe_percentile(lst: List, p: int) -> float:
    if not lst:
        return 0.0
    s = sorted(lst)
    idx = (p / 100) * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)