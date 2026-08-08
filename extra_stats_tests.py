import json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, spearmanr
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import cochrans_q
from statsmodels.formula.api import ols

LANGUAGES = ["en", "zh", "ar", "hi", "fi", "ko", "sw", "es", "tr", "de", "fr"]

def load_records(model, task, lang, results_dir="results"):
    p = Path(results_dir) / f"{model}__{task}__{lang}.jsonl"
    if not p.exists():
        p = Path(results_dir) / f"{model.replace('.', '_').replace('-', '_')}__{task}__{lang}.jsonl"
    if not p.exists():
        return []
    records = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    records.sort(key=lambda x: x.get("idx", 0))
    return records

def run_cochrans_q():
    print("\n" + "="*50)
    print("1. COCHRAN'S Q TEST (Multi-Language Accuracy)")
    print("="*50)
    print("Purpose: Tests if accuracy varies significantly across ALL 11 languages simultaneously.")
    
    results_dir = Path("results")
    files = list(results_dir.glob("*__*__*.jsonl"))
    combos = sorted({(f.stem.split("__")[0], f.stem.split("__")[1]) for f in files if len(f.stem.split("__")) == 3})
    
    for model, task in combos:
        matrix = []
        langs_found = []
        for lang in LANGUAGES:
            records = load_records(model, task, lang)
            if records:
                langs_found.append(lang)
                matrix.append([int(r.get("correct", False)) for r in records])
        
        if len(langs_found) > 2:
            try:
                matrix_t = np.array(matrix).T
                row_sums = matrix_t.sum(axis=1)
                valid_mask = (row_sums > 0) & (row_sums < len(langs_found))
                matrix_filtered = matrix_t[valid_mask]
                
                if len(matrix_filtered) > 0:
                    q_stat = cochrans_q(matrix_t)
                    sig = "***" if q_stat.pvalue < 0.001 else ("*" if q_stat.pvalue < 0.05 else "")
                    print(f"[{model} | {task}] Q-Stat: {q_stat.statistic:.2f}, p-value: {q_stat.pvalue:.5e} {sig}")
                else:
                    print(f"[{model} | {task}] No variance across languages.")
            except Exception as e:
                print(f"[{model} | {task}] Error: {e}")

def run_wilcoxon():
    print("\n" + "="*50)
    print("2. WILCOXON SIGNED-RANK TEST (Token Lengths)")
    print("="*50)
    print("Purpose: Tests if foreign languages use significantly more/less completion tokens than English for the same exact questions.")
    
    results_dir = Path("results")
    files = list(results_dir.glob("*__*__*.jsonl"))
    combos = sorted({(f.stem.split("__")[0], f.stem.split("__")[1]) for f in files if len(f.stem.split("__")) == 3})
    
    for model, task in combos:
        en_records = load_records(model, task, "en")
        if not en_records: continue
        
        en_tokens = np.array([r.get("completion_tokens", 0) for r in en_records])
        
        for lang in LANGUAGES:
            if lang == "en": continue
            lang_records = load_records(model, task, lang)
            if not lang_records or len(lang_records) != len(en_records): continue
            
            lang_tokens = np.array([r.get("completion_tokens", 0) for r in lang_records])
            
            diffs = lang_tokens - en_tokens
            if np.all(diffs == 0): continue
            
            try:
                res = wilcoxon(en_tokens, lang_tokens)
                sig = "***" if res.pvalue < 0.001 else ("*" if res.pvalue < 0.05 else "")
                mean_diff = lang_tokens.mean() - en_tokens.mean()
                if res.pvalue < 0.05:
                    print(f"[{model} | {task} | {lang} vs EN] Diff: {mean_diff:+.1f} tokens. p-value: {res.pvalue:.5e} {sig}")
            except Exception:
                pass


def run_spearman():
    print("\n" + "="*50)
    print("3. SPEARMAN'S RANK CORRELATION (Fertility vs Accuracy)")
    print("="*50)
    print("Purpose: Tests the strength of the relationship between token fertility and accuracy.")
    
    summary_path = Path("results/summary.csv")
    if not summary_path.exists():
        return
        
    df = pd.read_csv(summary_path)
    if "mean_fertility" not in df.columns or "accuracy" not in df.columns:
        return
        
    df = df.dropna(subset=["mean_fertility", "accuracy"])
    
    rho, p = spearmanr(df["mean_fertility"], df["accuracy"])
    sig = "***" if p < 0.001 else ("*" if p < 0.05 else "")
    print(f"[GLOBAL] Spearman's rho: {rho:.3f}, p-value: {p:.5e} {sig}")
    
    for task in df["task"].unique():
        sub_df = df[df["task"] == task]
        if len(sub_df) > 2:
            rho, p = spearmanr(sub_df["mean_fertility"], sub_df["accuracy"])
            sig = "***" if p < 0.001 else ("*" if p < 0.05 else "")
            print(f"[{task.upper()}] Spearman's rho: {rho:.3f}, p-value: {p:.5e} {sig}")

def run_anova():
    print("\n" + "="*50)
    print("4. TWO-WAY ANOVA (Model x Language Interaction on Token Bloat)")
    print("="*50)
    print("Purpose: Tests if the 'token tax' (generating more tokens for foreign languages) disproportionately affects smaller models.")
    print("Method: Runs Two-Way ANOVA on log(total_tokens) across all individual questions.\n")
    
    results_dir = Path("results")
    files = list(results_dir.glob("*__*__*.jsonl"))
    
    # Load all raw records into a massive DataFrame
    all_rows = []
    for f in files:
        parts = f.stem.split("__")
        if len(parts) != 3: continue
        model, task, lang = parts
        
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip(): continue
                r = json.loads(line)
                tot_toks = r.get("total_tokens", 0)
                if tot_toks > 0:
                    all_rows.append({
                        "model": model,
                        "task": task,
                        "language": lang,
                        "log_tokens": np.log1p(tot_toks) # Log transform for ANOVA homoscedasticity assumption
                    })
    
    if not all_rows:
        print("No raw data found.")
        return
        
    df = pd.DataFrame(all_rows)
    
    for task in df["task"].unique():
        sub_df = df[df["task"] == task].copy()
        if len(sub_df["model"].unique()) < 2 or len(sub_df["language"].unique()) < 2:
            continue
            
        print(f"--- ANOVA for {task.upper()} ({len(sub_df)} records) ---")
        try:
            # OLS regression for ANOVA
            ols_model = ols('log_tokens ~ C(model) + C(language) + C(model):C(language)', data=sub_df).fit()
            anova_table = sm.stats.anova_lm(ols_model, typ=2)
            
            inter_p = anova_table.loc['C(model):C(language)', 'PR(>F)']
            sig = "***" if inter_p < 0.001 else ("*" if inter_p < 0.05 else "")
            
            print(f"Interaction (Model x Language) p-value: {inter_p:.5e} {sig}")
            if inter_p < 0.05:
                print("  => PROVEN: The token inflation from foreign languages is significantly worse for some models (likely smaller ones) than others!\n")
            else:
                print("  => NOT SIGNIFICANT: Models scale their token usage proportionally across languages.\n")
        except Exception as e:
            print(f"Error running ANOVA: {e}")

if __name__ == "__main__":
    run_cochrans_q()
    run_wilcoxon()
    run_spearman()
    run_anova()
    print("\n" + "="*50)
    print("All tests completed.")
