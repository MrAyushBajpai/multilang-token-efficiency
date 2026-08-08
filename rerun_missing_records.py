"""
rerun_missing_records.py
=========================
Scans every *.jsonl in results_dir for gaps in idx 0..n_samples-1.
For any missing record, makes the Groq API call, builds the record in the
EXACT same shape as run_experiment.py's current schema, appends it to the
JSONL, and updates the corresponding row in summary.csv.

Changes from original:
  - n_samples default bumped 100 → 500 (experiment now uses 500)
  - Removed all REASONING_MODELS / use_reasoning references (no reasoning
    models in the current model registry)
  - GROQ_API_KEY is now read from env directly (it's not a module-level var
    in run_experiment.py)
  - call_groq() signature no longer takes use_reasoning
  - Record schema aligned with run_experiment.py's current fields
  - SYSTEM_PROMPTS.format() now passes both `language` and
    `lang_self_instruction` kwargs (required by the template)
  - user_prompt mirrors run_experiment.py: prepend native-lang instruction
    for non-English conditions

Run from the experiment root (same dir as run_experiment.py):

    python rerun_missing_records.py --results_dir results
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import run_experiment as exp  # reuses call_groq, SYSTEM_PROMPTS, CHECKERS, MODELS, LANGUAGES
from data.datasets import get_dataset
from scripts.metrics import compute_metrics


def find_gaps(records, n_samples):
    present = {r["idx"] for r in records}
    return [i for i in range(n_samples) if i not in present]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument(
        "--n_samples", type=int, default=500,
        help="Expected number of records per file (default: 500)",
    )
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    summary_path = results_path / "summary.csv"

    # Read API key from env (not a module-level var in run_experiment.py)
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key or api_key == "YOUR_GROQ_API_KEY_HERE":
        raise SystemExit(
            "GROQ_API_KEY not set. "
            "Add it to your .env file or set it as an environment variable."
        )

    found_any_gap = False

    for f in sorted(results_path.glob("*.jsonl")):
        records = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        if not records:
            continue

        gaps = find_gaps(records, args.n_samples)
        if not gaps:
            continue

        found_any_gap = True
        model = records[0]["model"]
        task  = records[0]["task"]
        lang  = records[0]["language"]

        lang_name = exp.LANGUAGES[lang]
        model_id  = exp.MODELS[model]
        checker   = exp.CHECKERS[task]

        print(f"{f.name}: missing idx {gaps} — fetching {len(gaps)} record(s)")

        dataset = get_dataset(task, n_samples=args.n_samples)
        client  = exp.Groq(api_key=api_key)

        for i in gaps:
            item = dataset[i]

            # Mirror run_experiment.py's prompt construction exactly
            lang_prefix = exp.LANG_SELF_INSTRUCTION.get(lang, "")
            sys_prompt  = exp.SYSTEM_PROMPTS[task].format(
                language=lang_name,
                lang_self_instruction=lang_prefix,
            )
            user_prompt = (
                f"{lang_prefix}\n\n{item['question']}"
                if lang_prefix and lang != "en"
                else item["question"]
            )

            result = exp.call_groq(client, model_id, sys_prompt, user_prompt)

            if result["error"]:
                print(f"  idx={i}: ERROR: {result['error']} — skipping, try again later")
                continue

            is_correct = checker(result["content"], item["answer"])

            # Schema matches run_experiment.py's current record layout
            record = {
                "idx":               i,
                "model":             model,
                "task":              task,
                "language":          lang,
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
                "max_tokens_cap":    exp.MAX_COMPLETION_TOKENS,
                "temperature":       exp.TEMPERATURE,
                "timestamp_utc":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            records.append(record)
            status = "correct" if is_correct else "incorrect"
            print(f"  idx={i}: fetched, comp_tok={result['completion_tokens']}, {status}")

        records.sort(key=lambda r: r["idx"])
        with open(f, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Update summary.csv row
        m = compute_metrics(records)
        if summary_path.exists():
            update_summary_row(summary_path, model, task, lang, m)
        print(f"  → {f.name} now has {len(records)} records; summary updated.\n")

    if not found_any_gap:
        print("No gaps found — all files already have the expected number of records.")


def update_summary_row(summary_path, model, task, lang, m):
    rows = []
    with open(summary_path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows.append(header)
        for row in reader:
            # Match by (model, task, lang) columns regardless of column position
            if len(row) >= 5 and (row[2], row[3], row[4]) == (model, task, lang):
                row = [
                    datetime.now(timezone.utc).isoformat(),
                    row[1], model, task, lang,
                    m["n"], m["n_correct"], round(m["accuracy"], 4),
                    round(m["mean_completion_tokens"], 2),
                    round(m["median_completion_tokens"], 2),
                    round(m["std_completion_tokens"], 2),
                    round(m["mean_total_tokens"], 2),
                    round(m["mean_latency_s"], 3),
                    round(m["avg_cost_per_attempt_usd"], 6),
                    round(m["ceff_usd"], 6),
                ]
            rows.append(row)

    with open(summary_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)


if __name__ == "__main__":
    main()