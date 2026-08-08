#!/usr/bin/env python3
"""
sort_jsonl.py
-------------
Re-sorts every JSONL run file in a results directory by the "idx" field.

The concurrent experiment runner (run_experiment_concurrent.py) fans out
API calls across multiple keys and writes records in completion order via
as_completed(), so records in the JSONL can arrive out of sequence.
This script restores natural question order (idx 0, 1, 2, …) in-place.

Usage
-----
    # Sort all *.jsonl files in the default results/ directory
    python sort_jsonl.py

    # Specify a different results directory
    python sort_jsonl.py --results_dir path/to/results

    # Dry run — report what's out of order without changing anything
    python sort_jsonl.py --dry_run

    # Sort a single file
    python sort_jsonl.py --file results/llama3.3-70b__math__en.jsonl

Options
-------
    --results_dir DIR   Directory containing *.jsonl files (default: results)
    --file FILE         Sort a single JSONL file instead of a whole directory
    --dry_run           Report disorder without writing any changes
    --backup            Keep a .bak copy of every file that gets rewritten
    --key FIELD         Record field to sort by (default: idx)
    --missing STR       How to handle records missing the sort key:
                          warn  — print a warning and keep the record (default)
                          drop  — silently discard the record
                          error — abort the whole file
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


# ── Core sort logic ────────────────────────────────────────────────────────────

def sort_jsonl_file(
    path: Path,
    sort_key: str = "idx",
    dry_run: bool = False,
    backup: bool = False,
    missing: str = "warn",
) -> dict:
    """
    Sort a single JSONL file by `sort_key`.

    Returns a summary dict:
        {
          "path":         str,
          "n_total":      int,   # records parsed
          "n_missing_key":int,   # records without the sort key
          "n_dropped":    int,   # records discarded (missing="drop")
          "n_duplicates": int,   # duplicate key values found
          "was_sorted":   bool,  # True if already in order (no write needed)
          "written":      bool,  # True if the file was actually rewritten
          "error":        str | None,
        }
    """
    summary = {
        "path": str(path),
        "n_total": 0,
        "n_missing_key": 0,
        "n_dropped": 0,
        "n_duplicates": 0,
        "was_sorted": False,
        "written": False,
        "error": None,
    }

    # ── Read ──────────────────────────────────────────────────────────────
    records = []
    try:
        with open(path, encoding="utf-8") as f:
            for lineno, raw in enumerate(f, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    records.append(json.loads(raw))
                except json.JSONDecodeError as e:
                    summary["error"] = f"line {lineno}: JSON parse error: {e}"
                    return summary
    except OSError as e:
        summary["error"] = f"cannot read file: {e}"
        return summary

    summary["n_total"] = len(records)

    if not records:
        summary["was_sorted"] = True
        return summary

    # ── Validate sort key presence ────────────────────────────────────────
    keep = []
    for rec in records:
        if sort_key not in rec:
            summary["n_missing_key"] += 1
            if missing == "error":
                summary["error"] = (
                    f"record missing key {sort_key!r}: {list(rec.keys())[:8]}"
                )
                return summary
            elif missing == "drop":
                summary["n_dropped"] += 1
                continue
            else:  # warn — keep but it will sort to position 0 via fallback
                keep.append(rec)
        else:
            keep.append(rec)

    records = keep

    # ── Detect duplicates (informational only) ────────────────────────────
    seen = {}
    for rec in records:
        k = rec.get(sort_key)
        if k in seen:
            summary["n_duplicates"] += 1
        else:
            seen[k] = True

    # ── Sort ──────────────────────────────────────────────────────────────
    # Records without the key sort last (Python's None < int raises TypeError
    # in Python 3, so we use a two-key tuple: (has_key, value)).
    try:
        sorted_records = sorted(
            records,
            key=lambda r: (sort_key not in r, r.get(sort_key)),
        )
    except TypeError as e:
        # Mixed types in the key field (e.g. int and str)
        summary["error"] = f"sort failed — mixed types in {sort_key!r}: {e}"
        return summary

    # ── Check if already sorted ───────────────────────────────────────────
    original_keys = [r.get(sort_key) for r in records]
    sorted_keys   = [r.get(sort_key) for r in sorted_records]
    summary["was_sorted"] = (original_keys == sorted_keys)

    if summary["was_sorted"] and summary["n_missing_key"] == 0:
        return summary  # nothing to do

    if dry_run:
        return summary

    # ── Write ─────────────────────────────────────────────────────────────
    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        try:
            shutil.copy2(path, bak)
        except OSError as e:
            summary["error"] = f"backup failed: {e}"
            return summary

    # Atomic-ish: write to a temp file first, then rename over the original.
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for rec in sorted_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.replace(path)
        summary["written"] = True
    except OSError as e:
        summary["error"] = f"write failed: {e}"
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass

    return summary


# ── Directory-level scan ───────────────────────────────────────────────────────

def sort_results_dir(
    results_dir: Path,
    sort_key: str = "idx",
    dry_run: bool = False,
    backup: bool = False,
    missing: str = "warn",
) -> list[dict]:
    jsonl_files = sorted(results_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No *.jsonl files found in {results_dir}")
        return []

    summaries = []
    for path in jsonl_files:
        s = sort_jsonl_file(path, sort_key, dry_run, backup, missing)
        summaries.append(s)
        _print_summary(s, dry_run)

    return summaries


def _print_summary(s: dict, dry_run: bool) -> None:
    name = Path(s["path"]).name

    if s["error"]:
        print(f"  ERROR  {name}: {s['error']}")
        return

    parts = []

    if s["was_sorted"] and s["n_missing_key"] == 0:
        parts.append("already sorted")
    elif dry_run:
        parts.append("OUT OF ORDER (dry-run, not written)")
    else:
        parts.append("sorted and rewritten" if s["written"] else "sorted (no changes needed)")

    parts.append(f"{s['n_total']} records")

    if s["n_missing_key"]:
        parts.append(f"{s['n_missing_key']} missing key")
    if s["n_dropped"]:
        parts.append(f"{s['n_dropped']} dropped")
    if s["n_duplicates"]:
        parts.append(f"{s['n_duplicates']} duplicate idx values")

    status = "DRY  " if dry_run and not s["was_sorted"] else ("OK   " if s["was_sorted"] else "FIXED")
    print(f"  {status}  {name}: {', '.join(parts)}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sort JSONL run files by idx (or any field) for the multilingual experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[0].strip(),
    )
    parser.add_argument(
        "--results_dir", default="results",
        help="Directory containing *.jsonl files (default: results)",
    )
    parser.add_argument(
        "--file", default=None,
        help="Sort a single JSONL file instead of a whole directory",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Report disorder without writing changes",
    )
    parser.add_argument(
        "--backup", action="store_true",
        help="Keep a .bak copy of every file that gets rewritten",
    )
    parser.add_argument(
        "--key", default="idx",
        help="Record field to sort by (default: idx)",
    )
    parser.add_argument(
        "--missing", choices=["warn", "drop", "error"], default="warn",
        help="How to handle records missing the sort key (default: warn)",
    )
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            sys.exit(1)
        s = sort_jsonl_file(path, args.key, args.dry_run, args.backup, args.missing)
        _print_summary(s, args.dry_run)
        sys.exit(1 if s["error"] else 0)

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"Directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Scanning {results_dir}/ for *.jsonl files …\n")
    summaries = sort_results_dir(results_dir, args.key, args.dry_run, args.backup, args.missing)

    if not summaries:
        sys.exit(0)

    n_errors  = sum(1 for s in summaries if s["error"])
    n_fixed   = sum(1 for s in summaries if s["written"])
    n_already = sum(1 for s in summaries if s["was_sorted"] and not s["error"])

    print(f"\nSummary: {len(summaries)} files — "
          f"{n_already} already sorted, {n_fixed} rewritten, {n_errors} errors")

    sys.exit(1 if n_errors else 0)


if __name__ == "__main__":
    main()
