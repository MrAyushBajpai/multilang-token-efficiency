"""
analyze.py
----------
Loads results/ and produces:
  1. Token efficiency ratio tables (paper Table 1 style)
  2. Accuracy delta table
  3. Ceff ratio table
  4. Truncation rate table
  5. Plots:
     - Bar chart: mean completion tokens by language/model/task
     - Scatter: accuracy vs token count (one dot per model × lang × task)
     - Heatmap: efficiency ratios across language × task
     - Bar chart: Ceff ratio vs English
     - Bar chart: truncation rate by language/model

Language groupings (for paper narrative):
  HIGH FERTILITY (expected more tokens):  zh, ar, hi, fi, ko, sw
  LOW  FERTILITY (expected similar):      es, tr, de, fr
  BASELINE:                               en
"""

import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path

from scripts.metrics import aggregate_results_dir, compute_efficiency_ratios

LANG_LABELS = {
    "en": "English",  "zh": "Chinese",  "ar": "Arabic",
    "hi": "Hindi",    "fi": "Finnish",  "ko": "Korean",
    "sw": "Swahili",  "es": "Spanish",  "tr": "Turkish",
    "de": "German",   "fr": "French",
}
# Ordered for consistent display (baseline first, then high-fertility, then low)
LANG_ORDER = ["en", "zh", "ar", "hi", "fi", "ko", "sw", "es", "tr", "de", "fr"]

TASK_LABELS = {"math": "Math", "commonsense": "Commonsense", "code": "Code"}

# (model, task) pairs to drop — add any known-bad cells here
EXCLUDED_CELLS: set[tuple[str, str]] = set()


# ── Load & process ────────────────────────────────────────────────────────────

def load_summary_df(results_dir: str) -> pd.DataFrame:
    results_path = Path(results_dir)
    jsonl_files  = list(results_path.glob("*.jsonl"))

    if jsonl_files:
        records = []
        for f in jsonl_files:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))

        raw = pd.DataFrame(records)

        agg_spec = {
            "n":                        ("correct", "count"),
            "n_correct":                ("correct", lambda x: (x == True).sum()),
            "accuracy":                 ("correct", lambda x: (x == True).mean()),
            "mean_completion_tokens":   ("completion_tokens", "mean"),
            "median_completion_tokens": ("completion_tokens", "median"),
            "std_completion_tokens":    ("completion_tokens", "std"),
            "mean_total_tokens":        ("total_tokens", "mean"),
            "mean_prompt_tokens":       ("prompt_tokens", "mean"),
            "mean_latency_s":           ("latency_s", "mean"),
            "median_latency_s":         ("latency_s", "median"),
            "mean_response_chars":      ("response_length", "mean"),
        }

        if "finish_reason" in raw.columns:
            agg_spec["n_truncated"] = (
                "finish_reason", lambda x: (x == "length").sum()
            )

        df = (
            raw.groupby(["model", "task", "language"])
            .agg(**agg_spec)
            .reset_index()
        )
        if "n_truncated" not in df.columns:
            df["n_truncated"] = 0
        df["truncation_rate"] = df["n_truncated"] / df["n"].replace(0, 1)

        summary_path = results_path / "summary.csv"
        if summary_path.exists():
            sdf = pd.read_csv(summary_path)
            cost_cols = [c for c in ["avg_cost_per_attempt_usd", "ceff_usd"]
                         if c in sdf.columns]
            if cost_cols:
                df = df.merge(
                    sdf[["model", "task", "language"] + cost_cols],
                    on=["model", "task", "language"], how="left"
                )
        if "ceff_usd" not in df.columns:
            df["avg_cost_per_attempt_usd"] = float("nan")
            df["ceff_usd"]                 = float("nan")

        print(f"  Loaded {len(records)} raw records → {len(df)} run rows.")

    else:
        summary_path = results_path / "summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"No JSONL files or summary.csv found in {results_dir}"
            )
        df = pd.read_csv(summary_path)
        print(f"  Loaded {len(df)} rows from summary.csv.")

    df["lang_label"] = df["language"].map(LANG_LABELS).fillna(df["language"])
    df["task_label"] = df["task"].map(TASK_LABELS).fillna(df["task"])
    df = exclude_cells(df)
    return df


def exclude_cells(df: pd.DataFrame) -> pd.DataFrame:
    if not EXCLUDED_CELLS:
        return df
    mask = pd.Series(False, index=df.index)
    for model, task in EXCLUDED_CELLS:
        mask |= (df["model"] == model) & (df["task"] == task)
    if mask.any():
        dropped = df.loc[mask, ["model", "task"]].drop_duplicates()
        print(f"  Excluding {mask.sum()} row(s): {list(dropped.itertuples(index=False))}")
    return df.loc[~mask].reset_index(drop=True)


def compute_ratio_table(
    df: pd.DataFrame, metric: str = "mean_completion_tokens"
) -> pd.DataFrame:
    en_df = df[df["language"] == "en"][["model", "task", metric]].rename(
        columns={metric: "en_baseline"}
    )
    merged = df.merge(en_df, on=["model", "task"], how="left")
    merged["ratio"] = merged[metric] / merged["en_baseline"]
    pivot = merged.pivot_table(
        index=["model", "task"], columns="language",
        values="ratio", aggfunc="mean"
    )
    cols = [c for c in LANG_ORDER if c in pivot.columns]
    return pivot[cols].round(3)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_token_bars(df: pd.DataFrame, output_dir: str) -> None:
    for model in df["model"].unique():
        mdf   = df[df["model"] == model]
        tasks = [t for t in TASKS if not mdf[mdf["task"] == t].empty]
        if not tasks:
            continue

        TASKS = ["math", "commonsense", "code"]
        tasks = [t for t in TASKS if not mdf[mdf["task"] == t].empty]

        fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 5), sharey=False)
        if len(tasks) == 1:
            axes = [axes]
        fig.suptitle(f"Mean Completion Tokens — {model}", fontsize=13)

        lang_order = [l for l in LANG_ORDER if l in mdf["language"].values]
        palette    = sns.color_palette("Set2", len(lang_order))

        for ax, task in zip(axes, tasks):
            tdf = (mdf[mdf["task"] == task]
                   .set_index("language")
                   .reindex(lang_order)
                   .reset_index())
            bars = ax.bar(
                tdf["lang_label"].fillna(tdf["language"]),
                tdf["mean_completion_tokens"],
                color=palette,
            )
            en_idx = lang_order.index("en") if "en" in lang_order else None
            if en_idx is not None:
                bars[en_idx].set_edgecolor("black")
                bars[en_idx].set_linewidth(2)
            ax.set_title(TASK_LABELS.get(task, task))
            ax.set_xlabel("Language")
            if ax is axes[0]:
                ax.set_ylabel("Mean Completion Tokens")
            ax.tick_params(axis="x", rotation=35)
            ax.yaxis.set_major_formatter(
                mtick.FuncFormatter(lambda x, _: f"{int(x):,}")
            )

        plt.tight_layout()
        out = Path(output_dir) / f"token_bars_{model}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")


def plot_accuracy_vs_tokens(df: pd.DataFrame, output_dir: str) -> None:
    all_tasks = [t for t in ["math", "commonsense", "code"]
                 if not df[df["task"] == t].empty]
    fig, axes = plt.subplots(
        1, len(all_tasks), figsize=(5 * len(all_tasks), 5), sharey=False
    )
    if len(all_tasks) == 1:
        axes = [axes]

    lang_palette = {
        l: c for l, c in zip(
            LANG_ORDER, sns.color_palette("tab10", len(LANG_ORDER))
        )
    }
    model_markers = {
        "llama3.3-70b": "o", "llama4-scout": "s",
        "llama3.1-8b": "^",  "gemma2-9b": "D", "mistral-saba": "P",
    }

    for ax, task in zip(axes, all_tasks):
        tdf = df[df["task"] == task]
        for _, row in tdf.iterrows():
            ax.scatter(
                row["mean_completion_tokens"], row["accuracy"],
                color=lang_palette.get(row["language"], "grey"),
                marker=model_markers.get(row["model"], "o"),
                s=120, alpha=0.85,
                label=f"{row['lang_label']} / {row['model']}",
            )
        ax.set_title(TASK_LABELS.get(task, task))
        ax.set_xlabel("Mean Completion Tokens")
        ax.set_ylabel("Accuracy")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

    # Deduplicated legend (language only)
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
    # Build unique lang-colour patches
    import matplotlib.patches as mpatches
    lang_handles = [
        mpatches.Patch(color=c, label=LANG_LABELS.get(l, l))
        for l, c in lang_palette.items()
        if l in df["language"].values
    ]
    fig.legend(handles=lang_handles, loc="lower center",
               ncol=6, bbox_to_anchor=(0.5, -0.08), title="Language")
    plt.suptitle("Accuracy vs Mean Completion Tokens", fontsize=13)
    plt.tight_layout()
    out = Path(output_dir) / "accuracy_vs_tokens.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_efficiency_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    ratio_df = compute_ratio_table(df, "mean_completion_tokens")
    fig, ax  = plt.subplots(figsize=(14, max(4, len(ratio_df) * 0.55)))
    sns.heatmap(
        ratio_df, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn_r",
        center=1.0, vmin=0.5, vmax=2.0, linewidths=0.5,
        mask=ratio_df.isnull(),
        cbar_kws={"label": "Token Ratio vs English"}
    )
    ax.set_title(
        "Completion Token Ratio vs English  (<1.0 = fewer tokens)", fontsize=12
    )
    ax.set_xlabel("Language")
    ax.set_ylabel("Model / Task")
    plt.tight_layout()
    out = Path(output_dir) / "efficiency_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_truncation_rates(df: pd.DataFrame, output_dir: str) -> None:
    if "truncation_rate" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    pivot = df.pivot_table(
        index=["model", "task"], columns="language",
        values="truncation_rate"
    )
    cols = [c for c in LANG_ORDER if c in pivot.columns]
    pivot[cols].plot(kind="bar", ax=ax, width=0.75)
    ax.set_title("Truncation Rate by Language (finish_reason == 'length')")
    ax.set_ylabel("Truncation Rate")
    ax.set_xlabel("Model / Task")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    ax.legend(title="Language", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    out = Path(output_dir) / "truncation_rates.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_ceff_comparison(df: pd.DataFrame, output_dir: str) -> None:
    if df["ceff_usd"].isna().all():
        print("  Skipping Ceff plot (no cost data).")
        return
    en_df = df[df["language"] == "en"][["model", "task", "ceff_usd"]].rename(
        columns={"ceff_usd": "en_ceff"}
    )
    merged = df.merge(en_df, on=["model", "task"])
    merged["ceff_ratio"] = merged["ceff_usd"] / merged["en_ceff"]

    fig, ax = plt.subplots(figsize=(14, 5))
    pivot = merged.pivot_table(
        index=["model", "task"], columns="language", values="ceff_ratio"
    )
    cols = [c for c in LANG_ORDER if c in pivot.columns]
    pivot[cols].plot(kind="bar", ax=ax, width=0.75)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="English baseline")
    ax.set_title(
        "Cost-Efficiency Ratio (Ceff) vs English\n(< 1.0 = cheaper per correct answer)"
    )
    ax.set_ylabel("Ceff Ratio")
    ax.set_xlabel("Model / Task")
    ax.legend(title="Language", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    out = Path(output_dir) / "ceff_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Tables ─────────────────────────────────────────────────────────────────────

def print_tables(df: pd.DataFrame, output_dir: str) -> None:
    out_dir = Path(output_dir)

    # Table 1: completion token ratios
    t1 = compute_ratio_table(df, "mean_completion_tokens")
    t1.to_csv(out_dir / "table_token_ratios.csv")
    print("\n=== Table 1: Completion Token Ratio vs English ===")
    print(t1.to_string())

    # Table 2: accuracy
    t2 = df.pivot_table(
        index=["model", "task"], columns="language", values="accuracy"
    )
    cols = [c for c in LANG_ORDER if c in t2.columns]
    t2   = t2[cols].round(3)
    t2.to_csv(out_dir / "table_accuracy.csv")
    print("\n=== Table 2: Accuracy ===")
    print(t2.to_string())

    # Table 3: truncation rates
    if "truncation_rate" in df.columns:
        t3 = df.pivot_table(
            index=["model", "task"], columns="language", values="truncation_rate"
        )
        cols = [c for c in LANG_ORDER if c in t3.columns]
        t3   = t3[cols].round(4)
        t3.to_csv(out_dir / "table_truncation_rates.csv")
        print("\n=== Table 3: Truncation Rates ===")
        print(t3.to_string())

    # Table 4: Ceff ratios
    if not df["ceff_usd"].isna().all():
        en_df  = df[df["language"] == "en"][["model", "task", "ceff_usd"]].rename(
            columns={"ceff_usd": "en_ceff"}
        )
        merged = df.merge(en_df, on=["model", "task"])
        merged["ceff_ratio"] = (merged["ceff_usd"] / merged["en_ceff"]).round(3)
        t4 = merged.pivot_table(
            index=["model", "task"], columns="language", values="ceff_ratio"
        )
        cols = [c for c in LANG_ORDER if c in t4.columns]
        t4   = t4[cols]
        t4.to_csv(out_dir / "table_ceff_ratios.csv")
        print("\n=== Table 4: Ceff Ratio vs English ===")
        print(t4.to_string())


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--plots_dir",   default="results/plots")
    args = parser.parse_args()

    Path(args.plots_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {args.results_dir}")
    df = load_summary_df(args.results_dir)
    print(f"  {len(df)} run records loaded.\n")

    print_tables(df, args.results_dir)

    print("\nGenerating plots …")
    plot_token_bars(df, args.plots_dir)
    plot_accuracy_vs_tokens(df, args.plots_dir)
    plot_efficiency_heatmap(df, args.plots_dir)
    plot_truncation_rates(df, args.plots_dir)
    plot_ceff_comparison(df, args.plots_dir)

    print(f"\nDone.  Tables → {args.results_dir}/   Plots → {args.plots_dir}/")


if __name__ == "__main__":
    main()