"""
recheck_results.py
-------------------
Re-scores all raw results using language-aware answer-prefix matching.

PROBLEM
-------
The original checkers (check_math, check_commonsense) hardcoded the
English prefix "answer:" when searching for the model's final answer
line. But every system prompt instructs the model to respond entirely
in the target language, so compliant non-English responses correctly
write the localized equivalent (e.g. "Respuesta:", "الإجابة:", "答案:",
"उत्तर:", "Cevap:") instead of "Answer:". Those responses were scored
as incorrect even when the underlying answer was correct.

This script re-applies corrected checkers to the EXISTING `response`
field in each results/*.jsonl file (no new API calls), rewrites the
`correct` field in place, and regenerates summary.csv with updated
accuracy and ceff_usd values. Token counts, latency, and cost-per-
attempt are untouched since those don't depend on correctness.

USAGE
-----
    python recheck_results.py --results_dir results

This OVERWRITES results/*.jsonl and results/summary.csv after backing
them up to results/_backup_pre_recheck/.

After running this, re-run:
    python analyze.py --results_dir results --plots_dir results/plots
"""

import json
import re
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


# ── Localized "Answer:" prefixes ───────────────────────────────────────────
# Each model was instructed to think and respond entirely in the target
# language, so the final answer line may use any of these. "answer:" is
# always included as a fallback since some models keep it in English
# regardless of instructions.

ANSWER_PREFIXES = {
    "en": ["answer:"],
    "zh": ["答案:", "答案：", "answer:"],
    "hi": ["उत्तर:", "उत्तर :", "answer:"],
    "ar": ["الإجابة:", "الجواب:", "إجابة:", "answer:"],
    "es": ["respuesta:", "answer:"],
    "tr": ["cevap:", "yanıt:", "answer:"],
}

DEFAULT_PREFIXES = ["answer:"]


# ── Corrected checkers ──────────────────────────────────────────────────────

def check_math(response_text: str, expected_answer, language: str = "en") -> bool:
    """Extract '<Prefix>: <number>' (in any language) and compare."""
    prefixes = ANSWER_PREFIXES.get(language, DEFAULT_PREFIXES)
    for line in reversed(response_text.strip().splitlines()):
        low = line.strip().lower()
        for p in prefixes:
            if low.startswith(p.lower()):
                # Take everything after the first colon on this line
                after_colon = line.split(":", 1)[1] if ":" in line else line[len(p):]
                candidate = after_colon.strip().replace(",", "")
                # Strip currency symbols / stray punctuation, keep digits/.- 
                m = re.search(r"-?\d+(?:\.\d+)?", candidate)
                if m:
                    try:
                        return abs(float(m.group(0)) - float(expected_answer)) < 1e-3
                    except ValueError:
                        pass
    return False


def check_commonsense(response_text: str, expected_answer: str, language: str = "en") -> bool:
    """
    Extract '<Prefix>: <answer>' (in any language) and compare.

    expected_answer may be a letter (A-D) or a digit (e.g. "1"-"4"),
    depending on the dataset item, so both forms are checked.
    """
    prefixes = ANSWER_PREFIXES.get(language, DEFAULT_PREFIXES)
    expected = str(expected_answer).strip().upper()

    for line in reversed(response_text.strip().splitlines()):
        low = line.strip().lower()
        for p in prefixes:
            if low.startswith(p.lower()):
                after_colon = line.split(":", 1)[1] if ":" in line else line[len(p):]
                after_colon = after_colon.strip()

                if expected.isdigit():
                    m = re.search(r"-?\d+", after_colon)
                    if m:
                        return m.group(0) == expected
                else:
                    m = re.search(r"[A-Da-d]", after_colon)
                    if m:
                        return m.group(0).upper() == expected
    return False


def check_code(response_text: str, expected_answer: str) -> bool:
    """Unchanged from original analyze pipeline: extract first python block
    and run it, checking stdout / successful execution."""
    code_blocks = re.findall(r"```python(.*?)```", response_text, re.DOTALL)
    if not code_blocks:
        return False
    try:
        result = subprocess.run(
            [sys.executable, "-c", code_blocks[0]],
            timeout=5, capture_output=True, text=True
        )
        return expected_answer in result.stdout or result.returncode == 0
    except Exception:
        return False


CHECKERS = {
    "math": check_math,
    "commonsense": check_commonsense,
    "code": check_code,
}


# ── Cells excluded from analysis entirely (kept in sync with analyze.py) ────

EXCLUDED_CELLS = {
    ("qwen3-32b", "code"),
}


# ── Re-scoring ────────────────────────────────────────────────────────────

def rescore_jsonl(path: Path) -> dict:
    """
    Re-applies the appropriate corrected checker to every record in `path`,
    rewriting the `correct` field. Returns a small stats dict.
    """
    with open(path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    if not records:
        return {"n": 0, "n_changed": 0, "old_acc": None, "new_acc": None}

    task = records[0]["task"]
    language = records[0]["language"]
    checker = CHECKERS.get(task)

    n_changed = 0
    n_old_correct = 0
    n_new_correct = 0

    for r in records:
        old_correct = bool(r.get("correct", False))
        n_old_correct += int(old_correct)

        if checker is None:
            new_correct = old_correct
        elif task == "code":
            # Code checker doesn't depend on language; only re-run if it
            # was never correctly evaluated (kept as-is here for safety,
            # since this script focuses on the prefix-language bug).
            new_correct = old_correct
        else:
            new_correct = checker(r["response"], r["expected_answer"], language)

        if new_correct != old_correct:
            n_changed += 1
        n_new_correct += int(new_correct)
        r["correct"] = new_correct

    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(records)
    return {
        "n": n,
        "n_changed": n_changed,
        "old_acc": n_old_correct / n,
        "new_acc": n_new_correct / n,
    }


def regenerate_summary(results_dir: Path):
    """
    Recomputes accuracy and ceff_usd in summary.csv from the (now corrected)
    jsonl files, keeping avg_cost_per_attempt_usd and all token/latency
    columns from the existing summary.csv untouched.
    """
    summary_path = results_dir / "summary.csv"
    if not summary_path.exists():
        print("  No summary.csv found, skipping summary regeneration.")
        return

    summary = pd.read_csv(summary_path)

    # Drop duplicate (model, task, language) rows, keep last (most recent run)
    summary = summary.drop_duplicates(
        subset=["model", "task", "language"], keep="last"
    ).reset_index(drop=True)

    # Drop excluded cells
    mask = pd.Series(False, index=summary.index)
    for model, task in EXCLUDED_CELLS:
        mask |= (summary["model"] == model) & (summary["task"] == task)
    summary = summary.loc[~mask].reset_index(drop=True)

    for f in sorted(results_dir.glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        if not records:
            continue

        model = records[0]["model"]
        task = records[0]["task"]
        language = records[0]["language"]

        if (model, task) in EXCLUDED_CELLS:
            continue

        n = len(records)
        n_correct = sum(1 for r in records if r["correct"])
        accuracy = n_correct / n

        row_mask = (
            (summary["model"] == model)
            & (summary["task"] == task)
            & (summary["language"] == language)
        )

        if row_mask.sum() == 0:
            print(f"  WARNING: no summary.csv row for {model}/{task}/{language}, skipping")
            continue

        idx = summary.index[row_mask][0]
        summary.loc[idx, "n"] = n
        summary.loc[idx, "n_correct"] = n_correct
        summary.loc[idx, "accuracy"] = round(accuracy, 4)

        avg_cost = summary.loc[idx, "avg_cost_per_attempt_usd"]
        summary.loc[idx, "ceff_usd"] = (
            round(avg_cost / accuracy, 8) if accuracy > 0 else float("inf")
        )

    summary.to_csv(summary_path, index=False)
    print(f"  Rewrote {summary_path} ({len(summary)} rows)")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Re-score results using language-aware answer checkers"
    )
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--no_backup", action="store_true",
                         help="Skip creating a backup before overwriting")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not args.no_backup:
        backup_dir = results_dir / "_backup_pre_recheck"
        backup_dir.mkdir(exist_ok=True)
        for f in results_dir.glob("*.jsonl"):
            shutil.copy2(f, backup_dir / f.name)
        summary_path = results_dir / "summary.csv"
        if summary_path.exists():
            shutil.copy2(summary_path, backup_dir / "summary.csv")
        print(f"Backed up jsonl files and summary.csv to {backup_dir}\n")

    print("Re-scoring JSONL files with language-aware checkers...")
    print(f"{'file':<42}{'n':>5}{'changed':>9}{'old_acc':>10}{'new_acc':>10}")

    total_changed = 0
    for f in sorted(results_dir.glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            first = fh.readline()
        if not first.strip():
            continue
        first_rec = json.loads(first)
        if (first_rec["model"], first_rec["task"]) in EXCLUDED_CELLS:
            print(f"{f.name:<42}{'':>5}{'SKIP':>9}  (excluded cell)")
            continue

        stats = rescore_jsonl(f)
        total_changed += stats["n_changed"]
        if stats["n"] == 0:
            continue
        flag = " *" if stats["n_changed"] > 0 else ""
        print(f"{f.name:<42}{stats['n']:>5}{stats['n_changed']:>9}"
              f"{stats['old_acc']:>10.3f}{stats['new_acc']:>10.3f}{flag}")

    print(f"\nTotal records re-scored from incorrect -> correct or vice versa: {total_changed}")

    print("\nRegenerating summary.csv (accuracy, ceff_usd)...")
    regenerate_summary(results_dir)

    print("\nDone. Now re-run: python analyze.py --results_dir results --plots_dir results/plots")


if __name__ == "__main__":
    main()