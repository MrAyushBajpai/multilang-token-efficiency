"""
dedup_results.py
-----------------
Cleans up results/ after re-runs that appended instead of overwriting.

For each (model, task, language) combination:
  - In each results/*.jsonl file, if it contains MULTIPLE distinct "runs"
    (detected via repeated idx=0 markers, i.e. the dataset restarting),
    only the LAST run's records are kept.
  - In summary.csv, if multiple rows exist for the same
    (model, task, language), only the LAST row (by file order) is kept.

Also drops any (model, task) combinations listed in EXCLUDED_CELLS,
matching analyze.py (e.g. qwen3-32b / code).

Usage:
    python dedup_results.py --results_dir results

This OVERWRITES the jsonl files and summary.csv in place after
backing them up to results/_backup_pre_dedup/.
"""

import json
import argparse
import shutil
from pathlib import Path

import pandas as pd

EXCLUDED_CELLS = {
    ("qwen3-32b", "code"),
}


def dedup_jsonl(path: Path) -> tuple[int, int]:
    """
    Reads a jsonl file. If it contains multiple runs (idx restarts back to 0
    or 1 partway through), keep only the records from the LAST run.
    Returns (n_before, n_after).
    """
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    n_before = len(records)
    if n_before == 0:
        return 0, 0

    # Find indices where idx resets to its minimum value (a new run starting)
    idxs = [r.get("idx") for r in records]
    min_idx = min(i for i in idxs if i is not None)

    run_start_positions = [
        pos for pos, v in enumerate(idxs) if v == min_idx
    ]

    if len(run_start_positions) <= 1:
        # Only one run, nothing to dedup
        return n_before, n_before

    # Keep everything from the last run start onward
    last_start = run_start_positions[-1]
    deduped = records[last_start:]

    with open(path, "w", encoding="utf-8") as fh:
        for r in deduped:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    return n_before, len(deduped)


def dedup_summary_csv(path: Path) -> pd.DataFrame:
    """
    Reads summary.csv. For duplicate (model, task, language) rows, keeps
    the LAST occurrence (assumed to be the most recent re-run, appended
    later in the file). Also drops EXCLUDED_CELLS.
    """
    df = pd.read_csv(path)

    key_cols = ["model", "task", "language"]
    n_before = len(df)

    # Keep last occurrence of each (model, task, language)
    df = df.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)

    # Drop excluded (model, task) cells
    mask = pd.Series(False, index=df.index)
    for model, task in EXCLUDED_CELLS:
        mask |= (df["model"] == model) & (df["task"] == task)
    df = df.loc[~mask].reset_index(drop=True)

    n_after = len(df)
    print(f"  summary.csv: {n_before} rows -> {n_after} rows "
          f"(removed {n_before - n_after} duplicate/excluded rows)")

    df.to_csv(path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="Deduplicate results after re-runs")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--no_backup", action="store_true",
                         help="Skip creating a backup before overwriting")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not args.no_backup:
        backup_dir = results_dir / "_backup_pre_dedup"
        backup_dir.mkdir(exist_ok=True)
        for f in results_dir.glob("*.jsonl"):
            shutil.copy2(f, backup_dir / f.name)
        summary_path = results_dir / "summary.csv"
        if summary_path.exists():
            shutil.copy2(summary_path, backup_dir / "summary.csv")
        print(f"Backed up jsonl files and summary.csv to {backup_dir}\n")

    print("Deduplicating JSONL files...")
    for f in sorted(results_dir.glob("*.jsonl")):
        # Skip excluded cells entirely (e.g. qwen3-32b__code__*.jsonl)
        skip = False
        for model, task in EXCLUDED_CELLS:
            if f"{model}__{task}__" in f.name or f.name.startswith(f"{model}__{task}"):
                skip = True
        if skip:
            print(f"  {f.name}: SKIPPED (excluded cell)")
            continue

        n_before, n_after = dedup_jsonl(f)
        if n_before != n_after:
            print(f"  {f.name}: {n_before} -> {n_after} records (removed earlier run)")
        else:
            print(f"  {f.name}: {n_before} records (no duplicate run detected)")

    print("\nDeduplicating summary.csv...")
    summary_path = results_dir / "summary.csv"
    if summary_path.exists():
        dedup_summary_csv(summary_path)
    else:
        print("  No summary.csv found, skipping.")

    print("\nDone. Now re-run: python analyze.py --results_dir results --plots_dir results/plots")


if __name__ == "__main__":
    main()