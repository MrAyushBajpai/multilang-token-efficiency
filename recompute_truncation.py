"""
recompute_truncation.py
========================
The 2048-token completion cap clips ~3% of qwen3-32b/math responses
(and 1 qwen3-32b/commonsense response) before the model reaches its
final "Answer:" line, so those records are automatically graded
incorrect -- not because the reasoning was wrong, but because it
never finished.

This script does NOT require any API calls. It re-derives, from the
already-collected JSONL, both the "as-reported" accuracy and an
accuracy figure that excludes truncated attempts from the denominator
(i.e. "accuracy among responses that actually finished"), plus the
truncation rate itself per (model, task, language).

Only qwen3-32b is affected: llama3.3-70b's max completion_tokens
across all 1700+ records is 1706, and llama4-scout's is 1117 -- both
well under the 2048 cap, so this script is a no-op for those models.

Output: results/table_accuracy_truncation_adjusted.csv
"""

import json
import argparse
from pathlib import Path
import pandas as pd

CAP = 2048


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    rows = []
    for f in sorted(Path(args.results_dir).glob("qwen3-32b__*.jsonl")):
        records = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        model = records[0]["model"]; task = records[0]["task"]; lang = records[0]["language"]

        n = len(records)
        capped = [r for r in records if r["completion_tokens"] >= CAP]
        n_capped = len(capped)

        n_correct_all = sum(1 for r in records if r["correct"])
        non_capped = [r for r in records if r["completion_tokens"] < CAP]
        n_correct_noncapped = sum(1 for r in non_capped if r["correct"])

        rows.append({
            "model": model, "task": task, "language": lang,
            "n": n,
            "n_truncated": n_capped,
            "truncation_rate": round(n_capped / n, 4),
            "accuracy_as_reported": round(n_correct_all / n, 4),
            "accuracy_excl_truncated": (
                round(n_correct_noncapped / len(non_capped), 4) if non_capped else None
            ),
            # how many of the truncated attempts were (unsurprisingly) wrong
            "truncated_correct": sum(1 for r in capped if r["correct"]),
        })

    df = pd.DataFrame(rows).sort_values(["model", "task", "language"])
    out = Path(args.results_dir) / "table_accuracy_truncation_adjusted.csv"
    df.to_csv(out, index=False)

    print(df.to_string(index=False))
    print(f"\nWrote: {out}")
    print(
        "\nNote: 'accuracy_excl_truncated' answers \"given the model produced a "
        "complete response, was it correct?\" -- it removes the mechanical "
        "penalty of running out of budget. 'accuracy_as_reported' is what's "
        "currently in table_accuracy.csv (truncated == incorrect)."
    )


if __name__ == "__main__":
    main()