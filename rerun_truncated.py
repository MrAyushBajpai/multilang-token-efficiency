"""
rerun_truncated.py
------------------
Finds every record with finish_reason == "length" across all JSONL run files,
reruns those specific questions with a doubling token limit until the response
finishes cleanly or Groq rejects the request, then updates the JSONL files and
summary.csv in-place.

Usage
-----
    python rerun_truncated.py [--results_dir results] [--max_cap 80000] [--dry_run]

Algorithm per truncated record
-------------------------------
1. Start at cap = max(record["max_tokens_cap"] * 2, INITIAL_DOUBLE_CAP).
2. Call Groq with that cap.
3. If finish_reason == "length": double cap and retry (up to MAX_DOUBLINGS).
4. If Groq raises a context-window / token-limit APIError: stop for this record
   (the model/API cannot handle the size; keep the old record unchanged).
5. If a clean finish is received: replace the old record in the JSONL and
   recompute the summary.csv row for that run.

JSONL update strategy
----------------------
- Read the entire JSONL into memory.
- Replace the matching record (by idx).
- Write back atomically (temp file → rename) so a crash never corrupts data.

summary.csv update strategy
-----------------------------
- Read the entire CSV into memory.
- Replace the row whose run_key matches (or append if not present yet).
- Write back atomically.
"""

import os
import sys
import csv
import json
import time
import random
import argparse
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# ── Make project importable ───────────────────────────────────────────────────
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

try:
    from groq import Groq, RateLimitError, APIError
except ImportError:
    raise SystemExit("groq package not installed.  Run: pip install groq")

# ── Constants ─────────────────────────────────────────────────────────────────
INITIAL_DOUBLE_CAP  = 10_000   # minimum starting cap when rerunning
MAX_DOUBLINGS       = 4        # 10k → 20k → 40k → 80k → 160k (5 attempts max)
TEMPERATURE         = 0.0
RETRY_BASE_DELAY    = 15       # seconds, doubles on rate-limit
MAX_RETRIES         = 6
INTER_QUESTION_DELAY = 1.2     # courtesy delay between API calls

# Groq context-window error substrings (adjust if Groq changes wording)
CONTEXT_WINDOW_ERRORS = (
    "context_length_exceeded",
    "maximum context length",
    "token limit",
    "reduce the length",
    "too many tokens",
)

# ── Model ID map (must match run_experiment.py) ───────────────────────────────
MODELS: dict[str, str] = {
    "llama3.3-70b": "llama-3.3-70b-versatile",
    "llama4-scout": "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama3.1-8b":  "llama-3.1-8b-instant",
    "gpt-oss-20b":  "openai/gpt-oss-20b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
}

CHECKERS: dict = {}  # populated lazily from run_experiment imports


def _load_checkers():
    """Import correctness checkers from run_experiment without running it."""
    global CHECKERS
    if CHECKERS:
        return
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_experiment", HERE / "run_experiment.py"
        )
        mod = importlib.util.load_from_spec(spec)  # type: ignore[attr-defined]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        CHECKERS.update(mod.CHECKERS)
    except Exception as e:
        # Fallback: inline copies so this script stays self-contained
        print(f"  [warn] Could not import run_experiment.py checkers ({e}); using inline fallbacks.")
        import re, subprocess

        def check_math(text, expected):
            for line in reversed(text.strip().splitlines()):
                if line.strip().lower().startswith("answer:"):
                    cand = line.split(":", 1)[1].strip().replace(",", "").replace(" ", "")
                    try:
                        return abs(float(cand) - float(expected)) < 1e-3
                    except ValueError:
                        pass
            return False

        def check_commonsense(text, expected):
            for line in reversed(text.strip().splitlines()):
                if line.strip().lower().startswith("answer:"):
                    letter = line.split(":", 1)[1].strip().upper()[:1]
                    return letter == expected.upper()
            return False

        def check_code(text, expected):
            import subprocess
            blocks = re.findall(r"```python(.*?)```", text, re.DOTALL)
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

        CHECKERS.update({"math": check_math, "commonsense": check_commonsense, "code": check_code})


# ── Groq call with doubling-cap loop ─────────────────────────────────────────

def call_groq_with_doubling(
    client: "Groq",
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    start_cap: int,
    max_doublings: int = MAX_DOUBLINGS,
) -> tuple[dict | None, int]:
    """
    Try the API call starting at `start_cap` tokens, doubling on truncation.

    Returns (result_dict, final_cap_used).
    result_dict is None if a hard context-window error was hit (give up).
    result_dict["finish_reason"] == "length" means we exhausted all doublings.
    """
    cap = start_cap
    for attempt_num in range(max_doublings + 1):
        result = _call_once(client, model_id, system_prompt, user_prompt, cap)

        if result is None:
            # Hard context-window error — cannot go further
            return None, cap

        if result["finish_reason"] != "length":
            # Clean finish (or other non-truncation finish)
            return result, cap

        # Still truncated — double and retry (unless we've exhausted doublings)
        if attempt_num < max_doublings:
            cap *= 2
            print(f"    [trunc] still truncated at cap={cap//2}; retrying with cap={cap} …")
            time.sleep(INTER_QUESTION_DELAY)
        # else: fall through and return the last truncated result

    return result, cap  # type: ignore[return-value]  # truncated after all doublings


def _call_once(
    client: "Groq",
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    cap: int,
) -> dict | None:
    """
    One API call with rate-limit retry.
    Returns None on a hard context-window error (caller should stop doubling).
    Returns a result dict otherwise (may contain error/finish_reason).
    """
    kwargs = {
        "model":              model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature":          TEMPERATURE,
        "max_completion_tokens": cap,
    }

    for attempt in range(MAX_RETRIES):
        try:
            t0       = time.perf_counter()
            response = client.chat.completions.create(**kwargs)
            latency  = time.perf_counter() - t0

            usage  = response.usage
            choice = response.choices[0]
            return {
                "content":           choice.message.content or "",
                "finish_reason":     choice.finish_reason,
                "prompt_tokens":     usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens":      usage.total_tokens,
                "latency_s":         round(latency, 4),
                "completion_time":   getattr(usage, "completion_time", None),
                "prompt_time":       getattr(usage, "prompt_time", None),
                "queue_time":        getattr(usage, "queue_time", None),
                "error":             None,
            }

        except RateLimitError:
            wait = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 3)
            print(f"    [rate-limit] attempt {attempt+1}/{MAX_RETRIES}, waiting {wait:.1f}s …")
            time.sleep(wait)

        except APIError as e:
            err_str = str(e).lower()
            if any(kw in err_str for kw in CONTEXT_WINDOW_ERRORS):
                print(f"    [context-limit] cap={cap}: {e}")
                return None   # signal caller to stop doubling
            print(f"    [api-error] {e}")
            # Transient API error — return an error dict (don't retry here)
            return {
                "content": "", "finish_reason": "error",
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                "latency_s": 0.0, "completion_time": None,
                "prompt_time": None, "queue_time": None,
                "error": str(e),
            }

        except Exception as e:
            print(f"    [unexpected] {e}")
            return {
                "content": "", "finish_reason": "error",
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                "latency_s": 0.0, "completion_time": None,
                "prompt_time": None, "queue_time": None,
                "error": str(e),
            }

    return {
        "content": "", "finish_reason": "error",
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        "latency_s": 0.0, "completion_time": None,
        "prompt_time": None, "queue_time": None,
        "error": "max_retries_exceeded",
    }


# ── JSONL helpers ─────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def save_jsonl_atomic(path: Path, records: list[dict]) -> None:
    """Write to a temp file then rename — crash-safe."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


# ── summary.csv helpers ───────────────────────────────────────────────────────

def compute_metrics_inline(records: list[dict]) -> dict:
    """Minimal local copy of metrics.compute_metrics to avoid import issues."""
    import math

    TOKEN_PRICE_PER_M = {
        "llama3.3-70b": {"input": 0.59, "output": 0.79},
        "llama4-scout":  {"input": 0.11, "output": 0.34},
        "llama3.1-8b":   {"input": 0.05, "output": 0.08},
        "llama3.2-3b":   {"input": 0.06, "output": 0.06},
        "gemma2-9b":     {"input": 0.20, "output": 0.20},
        "mistral-saba":  {"input": 0.79, "output": 0.79},
    }

    def _mean(lst):   return sum(lst)/len(lst) if lst else 0.0
    def _median(lst):
        if not lst: return 0.0
        s = sorted(lst); n = len(s)
        return (s[n//2-1]+s[n//2])/2 if n%2==0 else float(s[n//2])
    def _std(lst):
        if len(lst)<2: return 0.0
        m = _mean(lst)
        return math.sqrt(sum((x-m)**2 for x in lst)/(len(lst)-1))
    def _pct(lst, p):
        if not lst: return 0.0
        s = sorted(lst); idx = (p/100)*(len(s)-1)
        lo, hi = int(idx), min(int(idx)+1, len(s)-1)
        return s[lo]+(s[hi]-s[lo])*(idx-lo)

    n          = len(records)
    n_correct  = sum(1 for r in records if r.get("correct") is True)
    accuracy   = n_correct/n if n else 0.0

    comp_toks   = [r["completion_tokens"]   for r in records if "completion_tokens"   in r]
    total_toks  = [r["total_tokens"]        for r in records if "total_tokens"        in r]
    prompt_toks = [r["prompt_tokens"]       for r in records if "prompt_tokens"       in r]
    latencies   = [r["latency_s"]           for r in records if "latency_s"           in r]
    resp_lens   = [r["response_length"]     for r in records if "response_length"     in r]

    truncated = sum(1 for r in records if r.get("finish_reason") == "length")

    fertility_vals = []
    for r in records:
        words = len(r.get("response","").split())
        if words > 0 and "completion_tokens" in r:
            fertility_vals.append(r["completion_tokens"]/words)

    model_key = records[0].get("model","") if records else ""
    prices    = TOKEN_PRICE_PER_M.get(model_key, {"input":0.0,"output":0.0})
    avg_prompt = _mean(prompt_toks)
    avg_comp   = _mean(comp_toks)
    avg_cost   = (avg_prompt/1e6)*prices["input"] + (avg_comp/1e6)*prices["output"]
    ceff       = avg_cost/accuracy if accuracy>0 else float("inf")

    return {
        "n": n, "n_correct": n_correct, "accuracy": round(accuracy,6),
        "mean_completion_tokens":   _mean(comp_toks),
        "median_completion_tokens": _median(comp_toks),
        "std_completion_tokens":    _std(comp_toks),
        "p10_completion_tokens":    _pct(comp_toks,10),
        "p90_completion_tokens":    _pct(comp_toks,90),
        "mean_total_tokens":        _mean(total_toks),
        "mean_prompt_tokens":       _mean(prompt_toks),
        "mean_latency_s":           _mean(latencies),
        "median_latency_s":         _median(latencies),
        "p90_latency_s":            _pct(latencies,90),
        "mean_response_chars":      _mean(resp_lens),
        "mean_fertility":           _mean(fertility_vals),
        "n_truncated":              truncated,
        "truncation_rate":          round(truncated/n,6) if n else 0.0,
        "avg_cost_per_attempt_usd": avg_cost,
        "ceff_usd":                 ceff,
    }


CSV_HEADER = [
    "timestamp","run_key","model","task","language",
    "n","n_correct","accuracy",
    "mean_completion_tokens","median_completion_tokens",
    "std_completion_tokens","p10_completion_tokens","p90_completion_tokens",
    "mean_total_tokens","mean_prompt_tokens",
    "mean_latency_s","median_latency_s","p90_latency_s",
    "mean_response_chars","mean_fertility",
    "n_truncated","truncation_rate",
    "avg_cost_per_attempt_usd","ceff_usd",
]


def update_summary_csv(summary_path: Path, run_key: str, records: list[dict]) -> None:
    """Recompute the row for run_key and write it back atomically."""
    parts = run_key.split("__")
    model, task, lang = parts if len(parts)==3 else ("?","?","?")

    m = compute_metrics_inline(records)
    new_row = [
        datetime.now(timezone.utc).isoformat(),
        run_key, model, task, lang,
        m["n"], m["n_correct"], round(m["accuracy"],6),
        round(m["mean_completion_tokens"],2),
        round(m["median_completion_tokens"],2),
        round(m["std_completion_tokens"],2),
        round(m["p10_completion_tokens"],2),
        round(m["p90_completion_tokens"],2),
        round(m["mean_total_tokens"],2),
        round(m["mean_prompt_tokens"],2),
        round(m["mean_latency_s"],3),
        round(m["median_latency_s"],3),
        round(m["p90_latency_s"],3),
        round(m["mean_response_chars"],1),
        round(m["mean_fertility"],4),
        m["n_truncated"],
        round(m["truncation_rate"],6),
        round(m["avg_cost_per_attempt_usd"],8),
        round(m["ceff_usd"],8),
    ]

    # Read existing CSV (preserve header + all other rows)
    existing_rows: list[list] = []
    header = CSV_HEADER
    replaced = False

    if summary_path.exists():
        with open(summary_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if rows:
            header = rows[0]
            for row in rows[1:]:
                # run_key is column index 1
                if len(row) > 1 and row[1] == run_key:
                    existing_rows.append(new_row)
                    replaced = True
                else:
                    existing_rows.append(row)

    if not replaced:
        existing_rows.append(new_row)

    # Atomic write
    tmp = summary_path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(existing_rows)
    tmp.replace(summary_path)
    print(f"    [csv] summary.csv {'updated' if replaced else 'appended'} for {run_key}")


# ── Main ──────────────────────────────────────────────────────────────────────

def rerun_truncated(
    results_dir: str = "results",
    max_cap: int = 160_000,
    dry_run: bool = False,
) -> None:
    _load_checkers()

    api_key = os.environ.get("GROQ_API_KEY","")
    if not api_key or api_key == "YOUR_GROQ_API_KEY_HERE":
        raise SystemExit(
            "GROQ_API_KEY not set. Add it to your .env file or set it as an environment variable."
        )

    client       = Groq(api_key=api_key)
    results_path = Path(results_dir)
    summary_path = results_path / "summary.csv"

    jsonl_files = sorted(results_path.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {results_dir}/")
        return

    total_truncated = 0
    total_fixed     = 0
    total_gave_up   = 0

    for jsonl_path in jsonl_files:
        run_key = jsonl_path.stem
        records = load_jsonl(jsonl_path)

        truncated_indices = [
            (i, r) for i, r in enumerate(records)
            if r.get("finish_reason") == "length"
        ]

        if not truncated_indices:
            continue

        print(f"\n{'─'*70}")
        print(f"Run: {run_key}  |  truncated: {len(truncated_indices)}/{len(records)}")

        total_truncated += len(truncated_indices)
        run_modified = False

        for list_pos, (rec_idx, old_record) in enumerate(truncated_indices, 1):
            idx        = old_record.get("idx", rec_idx)
            model_key  = old_record.get("model","")
            task       = old_record.get("task","")
            model_id   = MODELS.get(model_key, model_key)
            checker    = CHECKERS.get(task)

            old_cap    = old_record.get("max_tokens_cap", 5000)
            start_cap  = max(old_cap * 2, INITIAL_DOUBLE_CAP)

            print(f"  [{list_pos}/{len(truncated_indices)}] idx={idx}  "
                  f"old_cap={old_cap}  start_cap={start_cap}")

            if dry_run:
                print(f"    [dry-run] would rerun with start_cap={start_cap}")
                continue

            if start_cap > max_cap:
                print(f"    [skip] start_cap={start_cap} already exceeds max_cap={max_cap}")
                total_gave_up += 1
                continue

            # Compute how many doublings are allowed before hitting max_cap
            allowed_doublings = 0
            cap = start_cap
            while cap * 2 <= max_cap and allowed_doublings < MAX_DOUBLINGS:
                cap *= 2
                allowed_doublings += 1

            result, final_cap = call_groq_with_doubling(
                client,
                model_id,
                old_record.get("system_prompt",""),
                old_record.get("user_prompt",""),
                start_cap=start_cap,
                max_doublings=allowed_doublings,
            )

            if result is None:
                print(f"    [give-up] context-window error at cap={final_cap}; keeping old record.")
                total_gave_up += 1
                continue

            if result.get("finish_reason") == "length":
                print(f"    [give-up] still truncated after cap={final_cap}; keeping old record.")
                total_gave_up += 1
                continue

            if result.get("error"):
                print(f"    [error] {result['error']}; keeping old record.")
                total_gave_up += 1
                continue

            # Re-evaluate correctness with the new (longer) response
            is_correct = False
            if checker:
                is_correct = checker(result["content"], old_record.get("expected_answer",""))

            new_record = deepcopy(old_record)
            new_record.update({
                "response":          result["content"],
                "correct":           is_correct,
                "prompt_tokens":     result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens":      result["total_tokens"],
                "latency_s":         result["latency_s"],
                "completion_time":   result["completion_time"],
                "prompt_time":       result["prompt_time"],
                "queue_time":        result["queue_time"],
                "finish_reason":     result["finish_reason"],
                "response_length":   len(result["content"]),
                "max_tokens_cap":    final_cap,
                "timestamp_utc":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                # Provenance fields
                "rerun":             True,
                "rerun_prev_cap":    old_cap,
                "rerun_final_cap":   final_cap,
            })

            records[rec_idx] = new_record
            run_modified = True
            total_fixed += 1

            status = "✓" if is_correct else "✗"
            print(f"    {status} finish_reason={result['finish_reason']}  "
                  f"comp={result['completion_tokens']}  cap={final_cap}")

            time.sleep(INTER_QUESTION_DELAY)

        if run_modified and not dry_run:
            save_jsonl_atomic(jsonl_path, records)
            print(f"  [saved] {jsonl_path.name}")
            update_summary_csv(summary_path, run_key, records)

    print(f"\n{'='*70}")
    print(f"Rerun complete.")
    print(f"  Truncated found : {total_truncated}")
    print(f"  Fixed (clean)   : {total_fixed}")
    print(f"  Gave up         : {total_gave_up}")
    if dry_run:
        print("  (DRY RUN — no files were modified)")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rerun truncated responses with doubling token limits."
    )
    parser.add_argument(
        "--results_dir", default="results",
        help="Directory containing JSONL files and summary.csv (default: results)"
    )
    parser.add_argument(
        "--max_cap", type=int, default=160_000,
        help="Hard upper bound on max_completion_tokens (default: 160000)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Scan and report truncated records without making any API calls or writes"
    )
    args = parser.parse_args()

    rerun_truncated(
        results_dir=args.results_dir,
        max_cap=args.max_cap,
        dry_run=args.dry_run,
    )