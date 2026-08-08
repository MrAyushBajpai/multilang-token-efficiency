#!/usr/bin/env python3
"""
fix_commonsense_checker.py
--------------------------
Fixes two bugs in check_commonsense() that caused correct answers to be
marked wrong:
  1. Bold markdown wrapping: **Answer: B** not matched by startswith("answer:")
  2. Localized answer words: Antwort:, Réponse:, Vastaus:, Jibu:, 答案：, etc.

Also fixes check_math() for the same bold-markdown issue.

Usage:
    python fix_commonsense_checker.py [--results_dir results] [--dry_run]
"""

import json, re, argparse, shutil
from pathlib import Path

# ── Patterns ──────────────────────────────────────────────────────────────────

COMMONSENSE_PATTERNS = [
    r'\*{0,2}answer\s*:\*{0,2}\s*\*{0,2}([A-D])\b',
    r'\*{0,2}r[eé]ponse\s*(?:correcte\s*)?:\*{0,2}\s*\*{0,2}([A-D])\b',
    r'\*{0,2}antwort\s*:\*{0,2}\s*\*{0,2}([A-D])\b',
    r'\*{0,2}vastaus\s*:\*{0,2}\s*\*{0,2}([A-D])\b',
    r'\*{0,2}jibu\s*:\*{0,2}\s*\*{0,2}([A-D])\b',
    r'(?:الجواب|الإجابة)\s*:\s*([A-D])\b',
    r'答案[：:]\s*([A-D])',
    r'(?:정답|답)\s*:\s*([A-D])',
    r'\*{0,2}cevap\s*:\*{0,2}\s*\*{0,2}([A-D])\b',
    r'\*{0,2}respuesta\s*:\*{0,2}\s*\*{0,2}([A-D])\b',
    r'\*{0,2}jawab(?:an)?\s*:\*{0,2}\s*\*{0,2}([A-D])\b',
    r'जवाब\s*[：:]\s*([A-D])',
    r'উত্তর\s*[：:]\s*([A-D])',
    r'\*{0,2}odpowied[źz]\s*:\*{0,2}\s*\*{0,2}([A-D])\b',
]

MATH_ANSWER_PAT = re.compile(r'\*{0,2}answer\s*:\*{0,2}\s*([\d\.\-]+)', re.IGNORECASE)


def extract_commonsense_letter(response):
    """Bottom-up scan for any answer marker."""
    for line in reversed(response.strip().splitlines()):
        for pat in COMMONSENSE_PATTERNS:
            m = re.search(pat, line, re.IGNORECASE)
            if m:
                return m.group(1).upper()
    return None


def extract_math_number(response):
    """Bottom-up scan for Answer: <number>, including bold variants."""
    for line in reversed(response.strip().splitlines()):
        m = MATH_ANSWER_PAT.search(line)
        if m:
            return m.group(1).strip().replace(",", "").replace(" ", "")
    return None


def fix_record(r):
    """Re-evaluate a record. Returns True if anything changed."""
    task = r.get("task", "")
    if r.get("correct") is True:
        return False  # already correct, skip

    changed = False

    if task == "commonsense" and r.get("check_detail") == "no_answer_line":
        expected = r.get("expected_answer", "").strip().upper()
        found = extract_commonsense_letter(r.get("response", ""))
        if found is not None:
            is_correct = (found == expected)
            r["correct"] = is_correct
            r["check_detail"] = "matched" if is_correct else f"mismatch: got {found!r} want {expected!r}"
            changed = True

    elif task == "math" and r.get("check_detail") == "no_answer_line":
        expected = r.get("expected_answer", "").strip().replace(",", "").replace(" ", "")
        found = extract_math_number(r.get("response", ""))
        if found is not None:
            try:
                is_correct = abs(float(found) - float(expected)) < 1e-3
                r["correct"] = is_correct
                r["check_detail"] = "matched" if is_correct else f"mismatch: got {found!r} want {expected!r}"
                changed = True
            except ValueError:
                pass

    return changed


def fix_file(path, dry_run=False):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    n_fixed = sum(fix_record(r) for r in records)
    n_now_correct = sum(1 for r in records if r.get("correct") is True)

    if n_fixed and not dry_run:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(path)

    return len(records), n_fixed, n_now_correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    files = sorted(results_dir.glob("*.jsonl"))
    if not files:
        print(f"No .jsonl files in {results_dir}")
        return

    total_fixed = 0
    for path in files:
        n, fixed, n_correct = fix_file(path, args.dry_run)
        if fixed:
            acc = n_correct / n if n else 0
            tag = "[DRY]" if args.dry_run else "FIXED"
            print(f"{tag}  {path.name}: +{fixed} corrected  acc={acc:.3f} ({n_correct}/{n})")
        else:
            print(f"  ok   {path.name}")
        total_fixed += fixed

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Total records corrected: {total_fixed}")
    if not args.dry_run and total_fixed:
        print("NOTE: Re-run finalize_run() or regenerate summary.csv to update aggregates.")


if __name__ == "__main__":
    main()