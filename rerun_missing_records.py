"""
rerun_missing_records.py
=========================
llama3_3-70b__commonsense__tr.jsonl has 99/100 records -- idx=99 is
missing (almost certainly dropped by the `continue` on API error in
run_experiment.py's main loop).

This script makes the ONE missing Groq API call, builds the record in
the exact same shape as run_experiment.py, appends it to the JSONL,
and updates the corresponding row in summary.csv in place (n: 99->100,
recomputed accuracy / token stats / ceff).

Run this from the SAME directory as your run_experiment.py (it imports
from it directly), with your .env / GROQ_API_KEY set up as before:

    python3 rerun_missing_records.py --results_dir results

It's generalized to scan every *.jsonl in results_dir for gaps in
idx 0..n_samples-1, not just this one file -- if everything else is
already complete it'll just report "no gaps found" for those.
"""

import argparse
import csv
import json
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
    parser.add_argument("--n_samples", type=int, default=100)
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    summary_path = results_path / "summary.csv"

    for f in sorted(results_path.glob("*.jsonl")):
        records = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        if not records:
            continue

        gaps = find_gaps(records, args.n_samples)
        if not gaps:
            continue

        model = records[0]["model"]
        task = records[0]["task"]
        lang = records[0]["language"]
        lang_name = exp.LANGUAGES[lang]
        model_id = exp.MODELS[model]
        use_reasoning = model_id in exp.REASONING_MODELS
        checker = exp.CHECKERS[task]

        print(f"{f.name}: missing idx {gaps} -- fetching {len(gaps)} record(s)")

        dataset = get_dataset(task, n_samples=args.n_samples)
        client = exp.Groq(api_key=exp.GROQ_API_KEY)

        for i in gaps:
            item = dataset[i]
            sys_prompt = exp.SYSTEM_PROMPTS[task].format(language=lang_name)
            user_prompt = item["question"]

            result = exp.call_groq(
                client, model_id, sys_prompt, user_prompt, use_reasoning=use_reasoning
            )
            if result["error"]:
                print(f"  idx={i}: ERROR: {result['error']} -- skipping, try again later")
                continue

            is_correct = checker(result["content"], item["answer"])
            record = {
                "idx": i, "model": model, "task": task, "language": lang,
                "question_id": item.get("id", i),
                "system_prompt": sys_prompt, "user_prompt": user_prompt,
                "response": result["content"], "reasoning": result["reasoning_text"],
                "expected_answer": item["answer"], "correct": is_correct,
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
                "reasoning_tokens": result["reasoning_tokens_est"],
                "latency_s": result["latency_s"],
                "response_length": len(result["content"]),
            }
            records.append(record)
            status = "correct" if is_correct else "incorrect"
            print(f"  idx={i}: fetched, comp_tok={result['completion_tokens']}, {status}")

        records.sort(key=lambda r: r["idx"])
        with open(f, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        # update summary.csv in place
        m = compute_metrics(records)
        update_summary_row(summary_path, model, task, lang, m)
        print(f"  -> {f.name} now has {len(records)} records; summary.csv updated.\n")


def update_summary_row(summary_path, model, task, lang, m):
    rows = []
    with open(summary_path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows.append(header)
        for row in reader:
            if row[1:5] == [f"{model}__{task}__{lang}", model, task, lang] or \
               (row[2], row[3], row[4]) == (model, task, lang):
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