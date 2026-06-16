"""
mcnemar_test.py
================
Runs McNemar's test on paired per-question correct/incorrect vectors
for every (model, task, lang_a vs lang_b) combination.

Because the SAME 100 questions appear in every language condition
(verified by check_question_overlap.py), we can treat each question
as a matched pair and test whether the model's per-question success
rate differs between two language conditions.

McNemar's test statistic (with continuity correction):
    chi2 = (|b - c| - 1)^2 / (b + c)
where:
    b = questions correct in lang_a but WRONG in lang_b
    c = questions correct in lang_b but WRONG in lang_a

Null hypothesis: P(correct in lang_a) == P(correct in lang_b)

A significant result (p < 0.05) means the accuracy difference between
the two language conditions is not attributable to question difficulty
alone -- the language itself is making questions easier or harder.

Outputs:
  - Console: full results table sorted by p-value
  - mcnemar_results.csv: full results
  - mcnemar_vs_english.csv: only comparisons against English baseline
"""

import json
import argparse
from pathlib import Path
from itertools import combinations
import pandas as pd
import numpy as np
from scipy.stats import chi2

LANGUAGES   = ["en", "zh", "hi", "ar", "es", "tr"]
LANG_LABELS = {"en":"English","zh":"Chinese","hi":"Hindi",
               "ar":"Arabic","es":"Spanish","tr":"Turkish"}

# ── helpers ──────────────────────────────────────────────────────────────────

def load_cell(results_dir, model_file, task, lang):
    for name in [model_file, model_file.replace('.','_').replace('-','_')]:
        f = Path(results_dir) / f"{name}__{task}__{lang}.jsonl"
        if f.exists():
            return [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    return []


def mcnemar(correct_a, correct_b):
    """
    correct_a, correct_b: lists of bool, same length, same question order.
    Returns (b, c, chi2_stat, p_value, odds_ratio).
    Uses Edwards' continuity correction.
    """
    assert len(correct_a) == len(correct_b)
    b = sum(1 for a, bv in zip(correct_a, correct_b) if a and not bv)   # a right, b wrong
    c = sum(1 for a, bv in zip(correct_a, correct_b) if not a and bv)   # a wrong, b right
    n_discordant = b + c

    if n_discordant == 0:
        return b, c, 0.0, 1.0, float('nan')

    # Chi-squared with continuity correction
    stat = (abs(b - c) - 1) ** 2 / n_discordant
    p    = 1 - chi2.cdf(stat, df=1)

    # Odds ratio (lang_a advantage over lang_b)
    odds = b / c if c > 0 else float('inf')

    return b, c, round(stat, 4), round(p, 6), round(odds, 3)


def effect_size(acc_a, acc_b, n=100):
    """Cohen's h for two proportions."""
    h = 2 * (np.arcsin(np.sqrt(acc_a)) - np.arcsin(np.sqrt(acc_b)))
    return round(abs(h), 4)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--out",         default="mcnemar_results.csv")
    parser.add_argument("--alpha",       type=float, default=0.05)
    args = parser.parse_args()

    results_path = Path(args.results_dir)

    # Auto-detect model/task combos
    files  = list(results_path.glob("*__*__*.jsonl"))
    combos = sorted({(f.stem.split("__")[0], f.stem.split("__")[1])
                     for f in files if len(f.stem.split("__")) == 3})

    rows = []

    for model_file, task in combos:
        cells = {}
        for lang in LANGUAGES:
            recs = load_cell(args.results_dir, model_file, task, lang)
            if recs:
                cells[lang] = recs

        if len(cells) < 2:
            continue

        langs_present = sorted(cells.keys())

        for lang_a, lang_b in combinations(langs_present, 2):
            recs_a = cells[lang_a]
            recs_b = cells[lang_b]

            # Align by positional order (verified identical question order)
            n = min(len(recs_a), len(recs_b))
            ca = [bool(r["correct"]) for r in recs_a[:n]]
            cb = [bool(r["correct"]) for r in recs_b[:n]]

            acc_a = sum(ca) / n
            acc_b = sum(cb) / n

            b, c, stat, p, odds = mcnemar(ca, cb)
            h = effect_size(acc_a, acc_b, n)

            sig = p < args.alpha
            direction = (f"{LANG_LABELS[lang_a]} > {LANG_LABELS[lang_b]}"
                         if acc_a > acc_b else
                         f"{LANG_LABELS[lang_b]} > {LANG_LABELS[lang_a]}"
                         if acc_b > acc_a else "tie")

            rows.append({
                "model":       model_file,
                "task":        task,
                "lang_a":      lang_a,
                "lang_b":      lang_b,
                "acc_a":       round(acc_a, 4),
                "acc_b":       round(acc_b, 4),
                "acc_diff":    round(acc_a - acc_b, 4),
                "b_a_right_b_wrong": b,
                "c_a_wrong_b_right": c,
                "n_discordant":      b + c,
                "chi2":        stat,
                "p_value":     p,
                "significant": sig,
                "direction":   direction,
                "cohens_h":    h,
            })

    df = pd.DataFrame(rows).sort_values(["task","model","p_value"])

    # ── print full results ────────────────────────────────────────────────────
    print(f"\n{'Model':20s} {'Task':14s} {'Lang A':10s} {'Lang B':10s} "
          f"{'Acc A':>6s} {'Acc B':>6s} {'Diff':>6s} "
          f"{'b':>4s} {'c':>4s} {'chi2':>7s} {'p':>9s} {'Sig?':>5s} {'h':>6s}")
    print("-" * 125)

    for _, r in df.iterrows():
        print(f"{r.model:20s} {r.task:14s} "
              f"{LANG_LABELS[r.lang_a]:10s} {LANG_LABELS[r.lang_b]:10s} "
              f"{r.acc_a:6.3f} {r.acc_b:6.3f} {r.acc_diff:+6.3f} "
              f"{int(r.b_a_right_b_wrong):4d} {int(r.c_a_wrong_b_right):4d} "
              f"{r.chi2:7.3f} {r.p_value:9.5f} "
              f"{'YES*' if r.significant else 'no':>5s} {r.cohens_h:6.4f}")

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n=== SIGNIFICANT PAIRS (p < {args.alpha}) ===")
    sig = df[df.significant].sort_values("p_value")
    if sig.empty:
        print("  None.")
    else:
        for _, r in sig.iterrows():
            print(f"  {r.model}/{r.task}: {r.direction}  "
                  f"(acc {r.acc_a:.3f} vs {r.acc_b:.3f}, "
                  f"p={r.p_value:.5f}, h={r.cohens_h:.3f})")

    print(f"\n=== COUNTS ===")
    print(f"  Total pairs tested : {len(df)}")
    print(f"  Significant (p<{args.alpha}): {df.significant.sum()}")
    print(f"  Not significant    : {(~df.significant).sum()}")

    # ── vs-English only ───────────────────────────────────────────────────────
    vs_en = df[(df.lang_a == "en") | (df.lang_b == "en")].copy()
    vs_en.to_csv("mcnemar_vs_english.csv", index=False)

    print(f"\n=== VS ENGLISH ONLY (p < {args.alpha}) ===")
    sig_en = vs_en[vs_en.significant].sort_values("p_value")
    if sig_en.empty:
        print("  None significant vs English.")
    else:
        for _, r in sig_en.iterrows():
            other = r.lang_b if r.lang_a == "en" else r.lang_a
            other_acc = r.acc_b if r.lang_a == "en" else r.acc_a
            en_acc = r.acc_a if r.lang_a == "en" else r.acc_b
            print(f"  {r.model}/{r.task}/{LANG_LABELS[other]:10s}: "
                  f"en={en_acc:.3f} vs {LANG_LABELS[other]}={other_acc:.3f}  "
                  f"p={r.p_value:.5f}, h={r.cohens_h:.3f}")

    df.to_csv(args.out, index=False)
    print(f"\nFull results saved to: {args.out}")
    print(f"English-only results: mcnemar_vs_english.csv")


if __name__ == "__main__":
    main()