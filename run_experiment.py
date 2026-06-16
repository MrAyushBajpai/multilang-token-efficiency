"""
Cross-Task Multilingual Token Efficiency Study
================================================
Tests token efficiency across languages and task types using Groq API.

Models (non-reasoning only, avoids ceiling problems):
  - llama3.3-70b  : Llama 3.3 70B Versatile  — strongest general model
  - llama4-scout  : Llama 4 Scout 17B         — newest architecture
  - llama3.1-8b   : Llama 3.1 8B Instant      — small / fast baseline
  - gpt-oss-20b   : OpenAI GPT OSS 20B        — fast OpenAI open model
  - gpt-oss-120b  : OpenAI GPT OSS 120B       — large OpenAI open model

NOTE ON QWEN3-32B:
  Qwen3-32B is a reasoning model on Groq. Even with reasoning disabled it
  buffers chain-of-thought tokens internally, routinely hitting the 6 k/min
  TPM free-tier limit and truncating at any token cap below ~8 k. We exclude
  it here to avoid confounding token counts with reasoning overhead. If you
  have a paid Groq key and want to include it, set INCLUDE_QWEN=True below.

Languages (10 total, chosen for maximal tokenization-efficiency contrast):
  High-fertility / morphologically complex (expected MORE tokens than English):
    zh  Chinese   — logographic, high fertility in BPE tokenizers
    ar  Arabic    — morphologically very rich (root-and-pattern)
    hi  Hindi     — Devanagari; moderate–high fertility
    fi  Finnish   — agglutinative (15 grammatical cases)
    ko  Korean    — agglutinative + syllabic Hangul
    sw  Swahili   — Bantu agglutinative, low-resource pre-training signal
  Low-fertility / typologically close to English (expected ~same tokens):
    es  Spanish   — close to English in BPE training data
    tr  Turkish   — agglutinative but well-represented in training
    de  German    — compound words inflate slightly
    fr  French    — same script family as English

  English (en) is the baseline for all ratios.

Tasks: math, commonsense, code
N_SAMPLES: 500 per cell
MAX_COMPLETION_TOKENS: 5000

Resume behaviour:
  - Run state is tracked in results/run_state.json
  - DONE runs are skipped on re-run
  - RUNNING / FAILED runs are resumed from the last completed question
  - Each question is written atomically to disk immediately after the API
    call returns, so Ctrl-C only loses the in-flight question
"""

import os
import sys
import json
import time
import signal
import random
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── PYTHONPATH: make data/ and scripts/ importable ──────────────────────────
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from data.datasets      import get_dataset
from scripts.metrics    import compute_metrics
from scripts.logger     import ExperimentLogger

try:
    from groq import Groq, RateLimitError, APIError
except ImportError:
    raise SystemExit("groq package not installed. Run: pip install groq")

# ── Toggle ────────────────────────────────────────────────────────────────────
INCLUDE_QWEN = False   # Set True only with a paid key & 8k+ token budget

# ── Model registry ────────────────────────────────────────────────────────────
MODELS: dict[str, str] = {
    "llama3.3-70b": "llama-3.3-70b-versatile",
    "llama4-scout": "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama3.1-8b":  "llama-3.1-8b-instant",
    "gpt-oss-20b":  "openai/gpt-oss-20b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
}

if INCLUDE_QWEN:
    MODELS["qwen3-32b"] = "qwen/qwen3-32b"

# ── Language registry (ordered for display) ───────────────────────────────────
LANGUAGES: dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "ar": "Arabic",
    "hi": "Hindi",
    "fi": "Finnish",
    "ko": "Korean",
    "sw": "Swahili",
    "es": "Spanish",
    "tr": "Turkish",
    "de": "German",
    "fr": "French",
}

TASKS = ["math", "commonsense", "code"]

# ── Experiment hyperparameters ─────────────────────────────────────────────────
N_SAMPLES              = 500
TEMPERATURE            = 0.0
MAX_COMPLETION_TOKENS  = 5000
MAX_RETRIES            = 6
RETRY_BASE_DELAY       = 15   # seconds (doubles each attempt)
INTER_QUESTION_DELAY   = 1.2  # seconds between questions (rate-limit courtesy)


# ── Prompt templates ──────────────────────────────────────────────────────────
SYSTEM_PROMPTS = {
    "math": (
        "You are a math problem solver. "
        "You MUST think and respond ENTIRELY in {language} — including ALL reasoning steps. "
        "Do NOT write any part of your solution in English unless {language} IS English. "
        "Solve the problem step by step in {language}. "
        "Give your final numerical answer on the LAST line, prefixed with 'Answer:' "
        "(the prefix 'Answer:' may stay in English; the number must follow it)."
    ),
    "commonsense": (
        "You are a reasoning assistant. "
        "You MUST think and respond ENTIRELY in {language} — including ALL reasoning steps. "
        "Do NOT write any part of your reasoning in English unless {language} IS English. "
        "Answer the multiple-choice question. Think step by step in {language}, "
        "then give your final answer as a single letter (A/B/C/D) on the LAST line "
        "prefixed with 'Answer:'."
    ),
    "code": (
        "You are an expert programmer. "
        "Write ALL natural-language explanations, comments, and reasoning in {language}. "
        "The code itself (variable names, function names, keywords) must remain in English "
        "and be placed inside ```python ... ``` blocks. "
        "Do NOT mix {language} text into the code block."
    ),
}


# ── API call ───────────────────────────────────────────────────────────────────

def call_groq(
    client: "Groq",
    model_id: str,
    system_prompt: str,
    user_prompt: str,
) -> dict:
    """
    Single Groq API call with exponential-backoff retry on rate limits.
    Returns a dict with all fields needed for the record, plus an 'error' key.
    """
    kwargs = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature":          TEMPERATURE,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
    }

    for attempt in range(MAX_RETRIES):
        try:
            t0       = time.perf_counter()
            response = client.chat.completions.create(**kwargs)
            latency  = time.perf_counter() - t0

            usage   = response.usage
            choice  = response.choices[0]
            content = choice.message.content or ""

            return {
                "content":          content,
                "finish_reason":    choice.finish_reason,
                "prompt_tokens":    usage.prompt_tokens,
                "completion_tokens":usage.completion_tokens,
                "total_tokens":     usage.total_tokens,
                "latency_s":        round(latency, 4),
                # Groq sometimes exposes these; keep whatever is available
                "completion_time":  getattr(usage, "completion_time", None),
                "prompt_time":      getattr(usage, "prompt_time", None),
                "queue_time":       getattr(usage, "queue_time", None),
                "error":            None,
            }

        except RateLimitError:
            wait = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 3)
            print(f"    [rate-limit] attempt {attempt+1}/{MAX_RETRIES}, "
                  f"waiting {wait:.1f}s …")
            time.sleep(wait)

        except APIError as e:
            print(f"    [api-error] {e}")
            return _error_result(str(e))

        except Exception as e:
            print(f"    [unexpected] {e}")
            return _error_result(str(e))

    return _error_result("max_retries_exceeded")


def _error_result(msg: str) -> dict:
    return {
        "content": "", "finish_reason": "error",
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        "latency_s": 0.0, "completion_time": None,
        "prompt_time": None, "queue_time": None,
        "error": msg,
    }


# ── Correctness checkers ───────────────────────────────────────────────────────

def check_math(response_text: str, expected: str) -> bool:
    for line in reversed(response_text.strip().splitlines()):
        if line.strip().lower().startswith("answer:"):
            candidate = line.split(":", 1)[1].strip().replace(",", "").replace(" ", "")
            try:
                return abs(float(candidate) - float(expected)) < 1e-3
            except ValueError:
                pass
    return False


def check_commonsense(response_text: str, expected: str) -> bool:
    for line in reversed(response_text.strip().splitlines()):
        if line.strip().lower().startswith("answer:"):
            letter = line.split(":", 1)[1].strip().upper()[:1]
            return letter == expected.upper()
    return False


def check_code(response_text: str, expected: str) -> bool:
    import re, subprocess
    blocks = re.findall(r"```python(.*?)```", response_text, re.DOTALL)
    if not blocks:
        return False
    try:
        result = subprocess.run(
            [sys.executable, "-c", blocks[0]],
            timeout=10, capture_output=True, text=True,
        )
        return expected in result.stdout or result.returncode == 0
    except Exception:
        return False


CHECKERS = {
    "math":        check_math,
    "commonsense": check_commonsense,
    "code":        check_code,
}


# ── Main experiment loop ────────────────────────────────────────────────────────

def run_experiment(
    models:     list[str] | None = None,
    tasks:      list[str] | None = None,
    languages:  list[str] | None = None,
    n_samples:  int               = N_SAMPLES,
    results_dir: str              = "results",
    dry_run:    bool              = False,
    force_rerun: bool             = False,
) -> None:

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key or api_key == "YOUR_GROQ_API_KEY_HERE":
        raise SystemExit(
            "GROQ_API_KEY not set. "
            "Add it to your .env file or set it as an environment variable."
        )

    client  = Groq(api_key=api_key)
    logger  = ExperimentLogger(results_dir)

    models    = models    or list(MODELS.keys())
    tasks     = tasks     or TASKS
    languages = languages or list(LANGUAGES.keys())

    total_runs  = len(models) * len(tasks) * len(languages)
    total_calls = total_runs * n_samples
    print(f"\n{'='*70}")
    print(f"Experiment configuration")
    print(f"  Models    : {models}")
    print(f"  Tasks     : {tasks}")
    print(f"  Languages : {languages}")
    print(f"  Samples   : {n_samples} per cell")
    print(f"  Max tokens: {MAX_COMPLETION_TOKENS}")
    print(f"  Total runs: {total_runs}  |  Total API calls: {total_calls}")
    print(f"  Results   : {results_dir}/")
    print(f"{'='*70}\n")

    if dry_run:
        print("[DRY RUN] Datasets will be downloaded/verified only.")
        for task in tasks:
            dataset = get_dataset(task, n_samples=n_samples)
            print(f"  {task}: {len(dataset)} questions ready.")
        return

    # Pre-download all datasets ONCE before any API calls
    print("Verifying / downloading datasets …")
    datasets: dict[str, list] = {}
    for task in tasks:
        datasets[task] = get_dataset(task, n_samples=n_samples)
        print(f"  {task}: {len(datasets[task])} questions ✓")
    print()

    run_num = 0
    for model_key in models:
        model_id = MODELS[model_key]

        for task in tasks:
            dataset = datasets[task]
            checker = CHECKERS[task]

            for lang_code in languages:
                run_num += 1
                lang_name = LANGUAGES[lang_code]
                run_key   = f"{model_key}__{task}__{lang_code}"

                # ── Resume / skip logic ───────────────────────────────────
                status = logger.get_status(run_key)
                if status == "DONE" and not force_rerun:
                    print(f"[{run_num}/{total_runs}] SKIP (already DONE): {run_key}")
                    continue

                done_indices = logger.completed_indices(run_key)
                if done_indices and not force_rerun:
                    print(f"[{run_num}/{total_runs}] RESUME ({len(done_indices)} done): {run_key}")
                else:
                    print(f"[{run_num}/{total_runs}] START: {run_key}")

                logger.mark_running(run_key)
                n_errors = 0

                for i, item in enumerate(dataset):
                    if i in done_indices and not force_rerun:
                        continue   # already saved

                    sys_prompt  = SYSTEM_PROMPTS[task].format(language=lang_name)
                    user_prompt = item["question"]

                    result = call_groq(
                        client, model_id, sys_prompt, user_prompt
                    )

                    if result["error"]:
                        n_errors += 1
                        print(f"  [{i+1:3d}/{n_samples}] ✗ ERROR: {result['error']}")
                        if n_errors >= 10:
                            logger.mark_failed(run_key, f"too_many_errors:{n_errors}")
                            print(f"  Too many consecutive errors — marking run FAILED.")
                            break
                        continue

                    n_errors = 0  # reset on success
                    is_correct = checker(result["content"], item["answer"])

                    record = {
                        "idx":               i,
                        "model":             model_key,
                        "task":              task,
                        "language":          lang_code,
                        "question_id":       item.get("id", str(i)),
                        # ── input / output ──
                        "system_prompt":     sys_prompt,
                        "user_prompt":       user_prompt,
                        "response":          result["content"],
                        "expected_answer":   str(item["answer"]),
                        "correct":           is_correct,
                        # ── token counts ──
                        "prompt_tokens":     result["prompt_tokens"],
                        "completion_tokens": result["completion_tokens"],
                        "total_tokens":      result["total_tokens"],
                        # ── timing ──
                        "latency_s":         result["latency_s"],
                        "completion_time":   result["completion_time"],
                        "prompt_time":       result["prompt_time"],
                        "queue_time":        result["queue_time"],
                        # ── meta ──
                        "finish_reason":     result["finish_reason"],
                        "response_length":   len(result["content"]),
                        "model_id":          model_id,
                        "max_tokens_cap":    MAX_COMPLETION_TOKENS,
                        "temperature":       TEMPERATURE,
                        "timestamp_utc":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }

                    # Write immediately — safe against Ctrl-C
                    logger.append_record(run_key, record)

                    status_char = "✓" if is_correct else "✗"
                    trunc_flag  = " [TRUNC]" if result["finish_reason"] == "length" else ""
                    print(
                        f"  [{i+1:3d}/{n_samples}] {status_char} "
                        f"comp={result['completion_tokens']:4d} "
                        f"total={result['total_tokens']:5d} "
                        f"lat={result['latency_s']:.2f}s"
                        f"{trunc_flag}"
                    )

                    time.sleep(INTER_QUESTION_DELAY)

                else:
                    # Loop completed without break — finalize
                    logger.finalize_run(run_key)
                    # Quick accuracy printout
                    done_recs = logger.completed_indices(run_key)
                    print(f"  → Run DONE. {len(done_recs)}/{n_samples} questions saved.")

    print(f"\n[Done] All results in: {results_dir}/")
    print(f"       State file    : {results_dir}/run_state.json")
    print(f"       Summary CSV   : {results_dir}/summary.csv")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multilingual token efficiency experiment"
    )
    parser.add_argument(
        "--models",     nargs="+", choices=list(MODELS.keys()), default=None,
        help="Subset of models to run (default: all)"
    )
    parser.add_argument(
        "--tasks",      nargs="+", choices=TASKS, default=None,
        help="Subset of tasks to run (default: all)"
    )
    parser.add_argument(
        "--languages",  nargs="+", choices=list(LANGUAGES.keys()), default=None,
        help="Subset of languages to run (default: all)"
    )
    parser.add_argument(
        "--n_samples",  type=int, default=N_SAMPLES,
        help=f"Questions per cell (default: {N_SAMPLES})"
    )
    parser.add_argument(
        "--results_dir", default="results",
        help="Directory for output files (default: results)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Download datasets only, make no API calls"
    )
    parser.add_argument(
        "--force_rerun", action="store_true",
        help="Re-run even cells already marked DONE (overwrites data)"
    )
    args = parser.parse_args()

    run_experiment(
        models      = args.models,
        tasks       = args.tasks,
        languages   = args.languages,
        n_samples   = args.n_samples,
        results_dir = args.results_dir,
        dry_run     = args.dry_run,
        force_rerun = args.force_rerun,
    )