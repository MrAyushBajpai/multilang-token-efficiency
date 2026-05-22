"""
Cross-Task Multilingual Token Efficiency Study
================================================
Tests token efficiency across languages and task types using Groq API.
Models: Qwen3-32B, Llama-3.3-70B, Llama-4-Scout
Tasks:  Math reasoning, Commonsense reasoning, Code generation
Languages: English, Chinese, Hindi, Arabic, Spanish, Turkish
"""

import os
import json
import time
import random
import argparse
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq, RateLimitError, APIError

from data.datasets import get_dataset
from scripts.metrics import compute_metrics
from scripts.logger import ExperimentLogger

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")

MODELS = {
    "qwen3-32b":      "qwen/qwen3-32b",
    "llama3.3-70b":   "llama-3.3-70b-versatile",
    "llama4-scout":   "meta-llama/llama-4-scout-17b-16e-instruct",
}

LANGUAGES = {
    "en": "English",
    "zh": "Chinese",
    "hi": "Hindi",
    "ar": "Arabic",
    "es": "Spanish",
    "tr": "Turkish",
}

TASKS = ["math", "commonsense", "code"]

# Groq model IDs that support reasoning_format
REASONING_MODELS = {"qwen/qwen3-32b"}

# How many problems per task per language per model
N_SAMPLES = 50

# Temperature fixed across all runs for reproducibility
TEMPERATURE = 0.6

# Retry settings
MAX_RETRIES = 5
RETRY_BASE_DELAY = 10  # seconds


# ── Prompt templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "math": (
        "You are a math problem solver. "
        "You must think and respond entirely in {language} — including all reasoning steps. "
        "Solve the given problem step by step in {language}. "
        "Give your final numerical answer on the last line prefixed with 'Answer:'. "
        "Do NOT translate the final answer — keep it as a number."
    ),
    "commonsense": (
        "You are a reasoning assistant. "
        "You must think and respond entirely in {language} — including all reasoning steps. "
        "Answer the multiple-choice question below. "
        "Think step by step in {language}, then give your final answer as a single letter (A/B/C/D) "
        "on the last line prefixed with 'Answer:'."
    ),
    "code": (
        "You are an expert programmer. "
        "You must write all explanations and comments entirely in {language}. "
        "Explain your approach and reasoning in {language}. "
        "Write the actual code solution in Python inside ```python ... ``` blocks. "
        "Code, variable names, and function names must remain in English — only natural language text should be in {language}."
    ),
}

# ── Core API call ─────────────────────────────────────────────────────────────

def call_groq(client, model_id, system_prompt, user_prompt, use_reasoning=False):
    """
    Single Groq API call with retry on rate limit.
    Returns dict with response text, token counts, latency, and reasoning tokens.
    """
    kwargs = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": TEMPERATURE,
        "max_completion_tokens": 2048,
    }

    # For Qwen3, request parsed reasoning so we can measure reasoning tokens
    if use_reasoning and model_id in REASONING_MODELS:
        kwargs["reasoning_format"] = "parsed"

    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            response = client.chat.completions.create(**kwargs)
            latency = time.perf_counter() - t0

            usage = response.usage
            content = response.choices[0].message.content or ""

            # Reasoning tokens only present for Qwen3 with parsed format
            reasoning_tokens = 0
            reasoning_text = ""
            if hasattr(response.choices[0].message, "reasoning") and \
               response.choices[0].message.reasoning:
                reasoning_text = response.choices[0].message.reasoning
                # Groq doesn't expose reasoning token count separately yet,
                # so we estimate from the reasoning text length as proxy
                reasoning_tokens = len(reasoning_text.split())

            return {
                "content": content,
                "reasoning_text": reasoning_text,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "reasoning_tokens_est": reasoning_tokens,
                "latency_s": round(latency, 3),
                "completion_time": getattr(usage, "completion_time", None),
                "prompt_time": getattr(usage, "prompt_time", None),
                "error": None,
            }

        except RateLimitError as e:
            wait = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 2)
            print(f"  [Rate limit] attempt {attempt+1}/{MAX_RETRIES}, waiting {wait:.1f}s...")
            time.sleep(wait)

        except APIError as e:
            print(f"  [API error] {e}")
            return {"error": str(e), "content": "", "prompt_tokens": 0,
                    "completion_tokens": 0, "total_tokens": 0,
                    "reasoning_tokens_est": 0, "latency_s": 0}

    return {"error": "Max retries exceeded", "content": "", "prompt_tokens": 0,
            "completion_tokens": 0, "total_tokens": 0,
            "reasoning_tokens_est": 0, "latency_s": 0}


# ── Correctness checkers ──────────────────────────────────────────────────────

def check_math(response_text, expected_answer):
    """Extract 'Answer: <number>' and compare."""
    for line in reversed(response_text.strip().splitlines()):
        if line.strip().lower().startswith("answer:"):
            candidate = line.split(":", 1)[1].strip().replace(",", "")
            try:
                return abs(float(candidate) - float(expected_answer)) < 1e-3
            except ValueError:
                pass
    return False


def check_commonsense(response_text, expected_answer):
    """Extract last 'Answer: X' line."""
    for line in reversed(response_text.strip().splitlines()):
        if line.strip().lower().startswith("answer:"):
            letter = line.split(":", 1)[1].strip().upper()
            return letter == expected_answer.upper()
    return False


def check_code(response_text, expected_answer):
    """
    Simple check: look for the expected function/output in code block.
    For a proper eval you'd run the code — left as a stub here.
    """
    import re
    code_blocks = re.findall(r"```python(.*?)```", response_text, re.DOTALL)
    if not code_blocks:
        return False
    # Stub: mark as 'needs_eval' for post-processing
    return "NEEDS_EVAL"


CHECKERS = {
    "math":        check_math,
    "commonsense": check_commonsense,
    "code":        check_code,
}


# ── Main experiment loop ──────────────────────────────────────────────────────

def run_experiment(models=None, tasks=None, languages=None, n_samples=N_SAMPLES,
                   results_dir="results", dry_run=False):

    client = Groq(api_key=GROQ_API_KEY)
    logger = ExperimentLogger(results_dir)

    models    = models    or list(MODELS.keys())
    tasks     = tasks     or TASKS
    languages = languages or list(LANGUAGES.keys())

    total_calls = len(models) * len(tasks) * len(languages) * n_samples
    print(f"\n{'='*60}")
    print(f"Experiment config:")
    print(f"  Models:    {models}")
    print(f"  Tasks:     {tasks}")
    print(f"  Languages: {languages}")
    print(f"  Samples:   {n_samples} per cell")
    print(f"  Total API calls: {total_calls}")
    print(f"{'='*60}\n")

    if dry_run:
        print("[DRY RUN] No API calls will be made.")
        return

    for model_key in models:
        model_id = MODELS[model_key]
        use_reasoning = model_id in REASONING_MODELS

        for task in tasks:
            dataset = get_dataset(task, n_samples=n_samples)
            checker = CHECKERS[task]

            for lang_code in languages:
                lang_name = LANGUAGES[lang_code]
                run_key = f"{model_key}__{task}__{lang_code}"
                print(f"\n[{run_key}]")

                run_results = []

                for i, item in enumerate(dataset):
                    sys_prompt = SYSTEM_PROMPTS[task].format(language=lang_name)
                    user_prompt = item["question"]

                    result = call_groq(
                        client, model_id, sys_prompt, user_prompt,
                        use_reasoning=use_reasoning
                    )

                    if result["error"]:
                        print(f"  [{i+1}/{n_samples}] ERROR: {result['error']}")
                        continue

                    is_correct = checker(result["content"], item["answer"])

                    record = {
                        "idx":               i,
                        "model":             model_key,
                        "task":              task,
                        "language":          lang_code,
                        "question_id":       item.get("id", i),
                        # --- input/output detail ---
                        "system_prompt":     sys_prompt,
                        "user_prompt":       user_prompt,
                        "response":          result["content"],
                        "reasoning":         result["reasoning_text"],
                        "expected_answer":   item["answer"],
                        "correct":           is_correct,
                        # --- token/perf stats ---
                        "prompt_tokens":     result["prompt_tokens"],
                        "completion_tokens": result["completion_tokens"],
                        "total_tokens":      result["total_tokens"],
                        "reasoning_tokens":  result["reasoning_tokens_est"],
                        "latency_s":         result["latency_s"],
                        "response_length":   len(result["content"]),
                    }
                    run_results.append(record)

                    status = "✓" if is_correct else "✗"
                    print(f"  [{i+1:2d}/{n_samples}] {status} "
                          f"comp_tok={result['completion_tokens']:4d} "
                          f"total_tok={result['total_tokens']:5d} "
                          f"lat={result['latency_s']:.2f}s")

                    # Polite delay to avoid rate limits (6000 tok/min free tier)
                    time.sleep(1.5)

                logger.save_run(run_key, run_results)
                summary = compute_metrics(run_results)
                print(f"  → accuracy={summary['accuracy']:.2%}  "
                      f"mean_completion_tokens={summary['mean_completion_tokens']:.1f}  "
                      f"mean_total_tokens={summary['mean_total_tokens']:.1f}")

    print("\n[Done] All results saved to:", results_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multilingual token efficiency experiment")
    parser.add_argument("--models",    nargs="+", choices=list(MODELS.keys()),   default=None)
    parser.add_argument("--tasks",     nargs="+", choices=TASKS,                 default=None)
    parser.add_argument("--languages", nargs="+", choices=list(LANGUAGES.keys()),default=None)
    parser.add_argument("--n_samples", type=int,  default=N_SAMPLES)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--dry_run",   action="store_true")
    args = parser.parse_args()

    run_experiment(
        models=args.models,
        tasks=args.tasks,
        languages=args.languages,
        n_samples=args.n_samples,
        results_dir=args.results_dir,
        dry_run=args.dry_run,
    )