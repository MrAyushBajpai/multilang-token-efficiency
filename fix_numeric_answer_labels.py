#!/usr/bin/env python3
"""
fix_numeric_answer_labels.py
-----------------------------
Root cause: ai2_arc labels ARC-Easy questions with either "A".."D" or
"1".."4" depending on the question, but the prompt always asks the model
to answer with a letter. Old records stored the raw numeric label as
expected_answer (e.g. "2"), so a model answering "Answer: B" (correct)
was graded wrong -- letter vs number never matches.

data/datasets.py now normalizes labels to A/B/C/D at generation time
(ordinal position), but that only fixes *future* runs. This script
regrades the numeric-labeled records already in results/*.jsonl by
mapping N -> chr('A' + N - 1) and rescoring against the letter the
model actually answered (already extracted in check_detail by
fix_commonsense_checker.py, e.g. "mismatch: got 'B' want '2'").

Usage:
    python fix_numeric_answer_labels.py [--results_dir results] [--dry_run]
"""
import json
import re
import argparse
from pathlib import Path

GOT_PAT = re.compile(r"got '([A-D])'")


def fix_record(r):
    if r.get("task") != "commonsense":
        return False
    expected = r.get("expected_answer", "").strip()
    if not expected.isdigit():
        return False

    letter_expected = chr(ord("A") + int(expected) - 1)
    detail = r.get("check_detail", "") or ""
    m = GOT_PAT.search(detail)
    if m:
        got = m.group(1)
    elif r.get("correct") is True:
        # already marked correct some other way; nothing to do
        return False
    else:
        return False  # no letter was ever extracted from the response

    is_correct = got == letter_expected
    if r.get("correct") == is_correct and r.get("expected_answer") == letter_expected:
        return False

    r["expected_answer"] = letter_expected
    r["correct"] = is_correct
    r["check_detail"] = "matched" if is_correct else f"mismatch: got {got!r} want {letter_expected!r}"
    return True


def fix_file(path, dry_run=False):
    records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    n_fixed = sum(fix_record(r) for r in records)
    n_correct = sum(1 for r in records if r.get("correct") is True)

    if n_fixed and not dry_run:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(path)

    return len(records), n_fixed, n_correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    files = sorted(Path(args.results_dir).glob("*commonsense*.jsonl"))
    total_fixed = 0
    for path in files:
        n, fixed, n_correct = fix_file(path, args.dry_run)
        if fixed:
            tag = "[DRY]" if args.dry_run else "FIXED"
            print(f"{tag}  {path.name}: +{fixed} regraded  acc={n_correct/n:.3f} ({n_correct}/{n})")
        total_fixed += fixed

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Total records regraded: {total_fixed}")
    if not args.dry_run and total_fixed:
        print("NOTE: run regen_summary.py next to rebuild summary.csv from the corrected jsonl.")


if __name__ == "__main__":
    main()
