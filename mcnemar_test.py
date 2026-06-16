"""
mcnemar_test.py
================
Runs McNemar's test on paired per-question correct/incorrect vectors
for every (model, task, lang_a vs lang_b) combination.

Because the SAME N questions are downloaded once and cached (see
data/datasets.py), question index i in language A corresponds to
the identical question i in language B. This makes each question a
matched pair — the condition required for McNemar's test to be valid.

McNemar's test (Edwards' continuity correction):
    chi2 = (|b - c| - 1)² / (b + c)
where:
    b = questions correct in lang_a but WRONG in lang_b
    c = questions correct in lang_b but WRONG in lang_a

Null: P(correct | lang_a) == P(correct | lang_b)
Significant (p < alpha): the language itself — not question difficulty —
is affecting accuracy.

Outputs:
  mcnemar_results.csv          — all pairs, all models, all tasks
  mcnemar_vs_english.csv       — only pairs involving English
  Console summary table
"""

import json
import argparse
from pathlib import Path
from itertools import combinations

import pandas as pd
import numpy as np
from scipy.stats import chi2

LANGUAGES = ["en", "zh", "ar", "hi", "fi", "ko", "sw", "es", "tr", "de", "fr"]
LANG_LABELS = {
    "en": "English", "zh": "Chinese", "ar": "Arabic",
    "hi": "Hindi",   "fi": "Finnish", "ko": "Korean",
    "sw": "Swahili", "es": "Spanish", "tr": "Turkish",
    "de": "German",  "fr": "French",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_cell(results_dir: str, model_file: str, task: str, lang: str) -> list:
    """Load records from <model>__<task>__<lang>.jsonl, sorted by idx."""
    for name in [model_file,
                 model_file.replace(".", "_").replace("-", "_")]:
        f = Path(results_dir) / f"{name}__{task}__{lang}.jsonl"
        if f.exists():
            records = []
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            # Sort by idx to guarantee alignment
            records.sort(key=lambda r: r.get("idx", 0))
            return records
    return []


def mcnemar_test(correct_a: list[bool], correct_b: list[bool]) -> tuple:
    """
    Returns (b, c, chi2_stat, p_value, odds_ratio).
    Uses Edwards' continuity correction.
    """
    assert len(correct_a) == len(correct_b), "Lists must be the same length"
    b = sum(1 for a, bv in zip(correct_a, correct_b) if     a and not bv)
    c = sum(1 for a, bv in zip(correct_a, correct_b) if not a and     bv)
    n_disc = b + c

    if n_disc == 0:
        return b, c, 0.0, 1.0, float("nan")

    stat = (abs(b - c) - 1) ** 2 / n_disc
    p    = 1 - chi2.cdf(stat, df=1)
    odds = b / c if c > 0 else float("inf")

    return b, c, round(stat, 4), round(p, 6), round(odds, 4)


def cohens_h(acc_a: float, acc_b: float) -> float:
    """Effect size for two proportions (arcsine transformation)."""
    h = 2 * (np.arcsin(np.sqrt(acc_a)) - np.arcsin(np.sqrt(acc_b)))
    return round(abs(h), 4)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--out",         default="mcnemar_results.csv")
    parser.add_argument("--alpha",       type=float, default=0.05)
    args = parser.parse_args()

    results_path = Path(args.results_dir)

    # Auto-detect model/task combos from JSONL filenames
    files  = list(results_path.glob("*__*__*.jsonl"))
    combos = sorted({
        (f.stem.split("__")[0], f.stem.split("__")[1])
        for f in files
        if len(f.stem.split("__")) == 3
    })

    if not combos:
        print("No result files found. Run run_experiment.py first.")
        return

    rows = []

    for model_file, task in combos:
        cells: dict[str, list] = {}
        for lang in LANGUAGES:
            recs = load_cell(args.results_dir, model_file, task, lang)
            if recs:
                cells[lang] = recs

        if len(cells) < 2:
            continue

        for lang_a, lang_b in combinations(sorted(cells), 2):
            recs_a = cells[lang_a]
            recs_b = cells[lang_b]

            # Align by idx — both lists are already sorted by idx
            n   = min(len(recs_a), len(recs_b))
            ca  = [bool(r["correct"]) for r in recs_a[:n]]
            cb  = [bool(r["correct"]) for r in recs_b[:n]]

            acc_a = sum(ca) / n
            acc_b = sum(cb) / n

            b, c, stat, p, odds = mcnemar_test(ca, cb)
            h                   = cohens_h(acc_a, acc_b)
            sig                 = p < args.alpha

            if acc_a > acc_b:
                direction = f"{LANG_LABELS[lang_a]} > {LANG_LABELS[lang_b]}"
            elif acc_b > acc_a:
                direction = f"{LANG_LABELS[lang_b]} > {LANG_LABELS[lang_a]}"
            else:
                direction = "tie"

            rows.append({
                "model":              model_file,
                "task":               task,
                "lang_a":             lang_a,
                "lang_b":             lang_b,
                "n_aligned":          n,
                "acc_a":              round(acc_a, 4),
                "acc_b":              round(acc_b, 4),
                "acc_diff":           round(acc_a - acc_b, 4),
                "b_a_right_b_wrong":  b,
                "c_a_wrong_b_right":  c,
                "n_discordant":       b + c,
                "chi2":               stat,
                "p_value":            p,
                "significant":        sig,
                "direction":          direction,
                "odds_ratio":         odds,
                "cohens_h":           h,
            })

    if not rows:
        print("No language pairs found with data in both cells.")
        return

    df = pd.DataFrame(rows).sort_values(["task", "model", "p_value"])

    # ── full table ─────────────────────────────────────────────────────────
    hdr = (f"{'Model':20s} {'Task':14s} {'Lang A':10s} {'Lang B':10s} "
           f"{'n':>5s} {'AccA':>6s} {'AccB':>6s} {'Diff':>6s} "
           f"{'b':>4s} {'c':>4s} {'chi2':>7s} {'p':>9s} {'Sig?':>5s} {'h':>6s}")
    print(f"\n{hdr}")
    print("-" * 130)
    for _, r in df.iterrows():
        print(
            f"{r.model:20s} {r.task:14s} "
            f"{LANG_LABELS[r.lang_a]:10s} {LANG_LABELS[r.lang_b]:10s} "
            f"{int(r.n_aligned):5d} "
            f"{r.acc_a:6.3f} {r.acc_b:6.3f} {r.acc_diff:+6.3f} "
            f"{int(r.b_a_right_b_wrong):4d} {int(r.c_a_wrong_b_right):4d} "
            f"{r.chi2:7.3f} {r.p_value:9.5f} "
            f"{'YES*' if r.significant else 'no':>5s} {r.cohens_h:6.4f}"
        )

    # ── significant pairs ──────────────────────────────────────────────────
    sig = df[df.significant].sort_values("p_value")
    print(f"\n=== SIGNIFICANT PAIRS (p < {args.alpha}) ===")
    if sig.empty:
        print("  None.")
    else:
        for _, r in sig.iterrows():
            print(
                f"  {r.model}/{r.task}: {r.direction}  "
                f"(acc {r.acc_a:.3f} vs {r.acc_b:.3f}, "
                f"p={r.p_value:.5f}, h={r.cohens_h:.3f})"
            )

    # ── counts ─────────────────────────────────────────────────────────────
    print(f"\n=== COUNTS ===")
    print(f"  Total pairs tested   : {len(df)}")
    print(f"  Significant (p<{args.alpha}): {df.significant.sum()}")
    print(f"  Not significant      : {(~df.significant).sum()}")

    # ── vs English ─────────────────────────────────────────────────────────
    vs_en = df[(df.lang_a == "en") | (df.lang_b == "en")].copy()
    vs_en.to_csv("mcnemar_vs_english.csv", index=False)
    print(f"\n=== VS ENGLISH (p < {args.alpha}) ===")
    sig_en = vs_en[vs_en.significant].sort_values("p_value")
    if sig_en.empty:
        print("  None significant vs English.")
    else:
        for _, r in sig_en.iterrows():
            other     = r.lang_b if r.lang_a == "en" else r.lang_a
            other_acc = r.acc_b  if r.lang_a == "en" else r.acc_a
            en_acc    = r.acc_a  if r.lang_a == "en" else r.acc_b
            print(
                f"  {r.model}/{r.task}/{LANG_LABELS[other]:10s}: "
                f"en={en_acc:.3f} vs {LANG_LABELS[other]}={other_acc:.3f}  "
                f"p={r.p_value:.5f}, h={r.cohens_h:.3f}"
            )

    df.to_csv(args.out, index=False)
    print(f"\nFull results    → {args.out}")
    print(f"English-only    → mcnemar_vs_english.csv")


if __name__ == "__main__":
    main()