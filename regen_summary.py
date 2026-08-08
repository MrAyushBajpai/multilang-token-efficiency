#!/usr/bin/env python3
"""
regen_summary.py
-----------------
summary.csv had drifted from the raw jsonl (2 duplicate rows for
llama4-scout/code/es and llama4-scout/commonsense/ko, plus stale
accuracy after fix_commonsense_checker.py / fix_numeric_answer_labels.py
edits) because it was being hand-patched/appended in place instead of
regenerated. This rebuilds it from scratch, one row per (model, task,
language), directly from the jsonl files -- the only source of truth.

Usage:
    python regen_summary.py [--results_dir results]
"""
import argparse
import csv
import json
from pathlib import Path

from scripts.metrics import compute_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    files = sorted(results_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No .jsonl files in {results_dir}")

    rows = []
    seen = set()
    for path in files:
        model, task, lang = path.stem.split("__")
        key = (model, task, lang)
        if key in seen:
            raise SystemExit(f"Duplicate result file for {key}: {path}")
        seen.add(key)

        records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        m = compute_metrics(records)
        if not m:
            continue
        rows.append({
            "timestamp": max((r.get("timestamp_utc", "") for r in records), default=""),
            "run_key": f"{model}__{task}__{lang}",
            "model": model, "task": task, "language": lang,
            **m,
        })

    out_path = results_dir / "summary.csv"
    fieldnames = ["timestamp", "run_key", "model", "task", "language",
                  "n", "n_correct", "accuracy",
                  "mean_completion_tokens", "median_completion_tokens", "std_completion_tokens",
                  "p10_completion_tokens", "p90_completion_tokens",
                  "mean_total_tokens", "mean_prompt_tokens",
                  "mean_latency_s", "median_latency_s", "p90_latency_s",
                  "mean_response_chars", "mean_fertility",
                  "n_truncated", "truncation_rate",
                  "avg_cost_per_attempt_usd", "ceff_usd"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows ({len(files)} jsonl files found) -> {out_path}")


if __name__ == "__main__":
    main()
