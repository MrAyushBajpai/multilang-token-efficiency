"""
analyze.py
----------
Loads results/ and produces:
  1. Token efficiency ratio tables (paper Table 1 style)
  2. Accuracy delta table
  3. Ceff ratio table
  4. Plots:
     - Bar chart: mean completion tokens by language/model/task
     - Scatter: accuracy vs token count (bubble = language)
     - Heatmap: efficiency ratios across language × task
     - Bar chart: Ceff ratio vs English

NOTE: qwen3-32b / code is excluded from all tables and plots.
Qwen3-32B's chain-of-thought overhead on code tasks caused severe
truncation at the original 2048-token cap, making those results
invalid for cross-language comparison. The cell is dropped here
rather than re-included with a different cap, to keep the
qwen3-32b comparison internally consistent across tasks.
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
    "en": "English", "zh": "Chinese", "hi": "Hindi",
    "ar": "Arabic",  "es": "Spanish", "tr": "Turkish",
}
TASK_LABELS = {"math": "Math", "commonsense": "Commonsense", "code": "Code"}

# Cells to exclude from analysis entirely.
# Each entry is (model, task); all languages for that combination are dropped.
EXCLUDED_CELLS = {
    ("qwen3-32b", "code"),
}


# ── Load & process ────────────────────────────────────────────────────────────

def load_summary_df(results_dir: str) -> pd.DataFrame:
    """
    Tries JSONL files first (richer data), falls back to summary.csv.
    Aggregates per-record JSONL into per-run summary rows to match
    the shape the rest of analyze.py expects.
    """
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

        # Aggregate per (model, task, language) — same shape as summary.csv
        df = (
            raw.groupby(["model", "task", "language"])
            .agg(
                n                      = ("correct", "count"),
                n_correct              = ("correct", lambda x: (x == True).sum()),
                accuracy               = ("correct", lambda x: (x == True).mean()),
                mean_completion_tokens = ("completion_tokens", "mean"),
                median_completion_tokens = ("completion_tokens", "median"),
                std_completion_tokens  = ("completion_tokens", "std"),
                mean_total_tokens      = ("total_tokens", "mean"),
                mean_latency_s         = ("latency_s", "mean"),
            )
            .reset_index()
        )

        # Reconstruct ceff_usd: needs avg_cost_per_attempt — pull from summary.csv
        # if available, otherwise leave as NaN (ceff plot will be skipped gracefully)
        summary_path = results_path / "summary.csv"
        if summary_path.exists():
            sdf = pd.read_csv(summary_path)[
                ["model", "task", "language", "avg_cost_per_attempt_usd", "ceff_usd"]
            ]
            df = df.merge(sdf, on=["model", "task", "language"], how="left")
        else:
            df["avg_cost_per_attempt_usd"] = float("nan")
            df["ceff_usd"]                 = float("nan")

        print(f"  Loaded {len(records)} raw records from {len(jsonl_files)} JSONL files → {len(df)} run rows.")

    else:
        # fallback
        summary_path = results_path / "summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"No JSONL files or summary.csv found in {results_dir}")
        df = pd.read_csv(summary_path)
        print(f"  Loaded {len(df)} rows from summary.csv (fallback).")

    df["lang_label"] = df["language"].map(LANG_LABELS)
    df["task_label"] = df["task"].map(TASK_LABELS)

    df = exclude_cells(df)

    return df


def exclude_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any (model, task) combinations listed in EXCLUDED_CELLS."""
    if not EXCLUDED_CELLS:
        return df

    mask = pd.Series(False, index=df.index)
    for model, task in EXCLUDED_CELLS:
        mask |= (df["model"] == model) & (df["task"] == task)

    if mask.any():
        dropped = df.loc[mask, ["model", "task", "language"]]
        print(f"  Excluding {mask.sum()} row(s) per EXCLUDED_CELLS: "
              f"{sorted(set(zip(dropped['model'], dropped['task'])))}")

    return df.loc[~mask].reset_index(drop=True)


def compute_ratio_table(df: pd.DataFrame, metric: str = "mean_completion_tokens") -> pd.DataFrame:
    """
    Returns a pivot table of metric ratios relative to English,
    indexed by (model, task), columns = languages.
    """
    en_df = df[df["language"] == "en"][["model", "task", metric]].rename(
        columns={metric: "en_baseline"}
    )
    merged = df.merge(en_df, on=["model", "task"], how="left")
    merged["ratio"] = merged[metric] / merged["en_baseline"]
    pivot = merged.pivot_table(index=["model", "task"], columns="language",
                               values="ratio", aggfunc="mean")
    return pivot.round(3)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_token_bars(df: pd.DataFrame, output_dir: str):
    """Grouped bar chart: completion tokens by language, faceted by task."""
    models = df["model"].unique()
    for model in models:
        mdf = df[df["model"] == model]
        tasks = [t for t in ["math", "commonsense", "code"] if not mdf[mdf["task"] == t].empty]
        if not tasks:
            continue

        fig, axes = plt.subplots(1, len(tasks), figsize=(14, 4), sharey=False)
        if len(tasks) == 1:
            axes = [axes]
        fig.suptitle(f"Mean Completion Tokens — {model}", fontsize=13)

        for ax, task in zip(axes, tasks):
            tdf = mdf[mdf["task"] == task].sort_values("language")
            bars = ax.bar(tdf["lang_label"], tdf["mean_completion_tokens"],
                          color=sns.color_palette("Set2", len(tdf)))
            # Highlight English bar in a darker shade
            en_idx = list(tdf["language"]).index("en") if "en" in tdf["language"].values else None
            if en_idx is not None:
                bars[en_idx].set_edgecolor("black")
                bars[en_idx].set_linewidth(1.5)
            ax.set_title(TASK_LABELS.get(task, task))
            ax.set_xlabel("Language")
            if ax == axes[0]:
                ax.set_ylabel("Mean Completion Tokens")
            ax.tick_params(axis="x", rotation=30)
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))

        plt.tight_layout()
        out = Path(output_dir) / f"token_bars_{model}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")


def plot_accuracy_vs_tokens(df: pd.DataFrame, output_dir: str):
    """Scatter: accuracy vs mean_completion_tokens. One dot per (model, task, lang)."""
    tasks = [t for t in ["math", "commonsense", "code"] if not df[df["task"] == t].empty]
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 5), sharey=False)
    if len(tasks) == 1:
        axes = [axes]

    palette = sns.color_palette("tab10", len(LANG_LABELS))
    lang_colors = {lang: palette[i] for i, lang in enumerate(LANG_LABELS)}

    for ax, task in zip(axes, tasks):
        tdf = df[df["task"] == task]
        for _, row in tdf.iterrows():
            ax.scatter(
                row["mean_completion_tokens"], row["accuracy"],
                color=lang_colors.get(row["language"], "grey"),
                s=120, alpha=0.8,
                label=LANG_LABELS.get(row["language"], row["language"]),
                marker={"qwen3-32b": "o", "llama3.3-70b": "s",
                        "llama4-scout": "^"}.get(row["model"], "o"),
            )
        ax.set_title(TASK_LABELS.get(task, task))
        ax.set_xlabel("Mean Completion Tokens")
        ax.set_ylabel("Accuracy")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

    # Deduplicated legend
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               loc="lower center", ncol=len(LANG_LABELS), bbox_to_anchor=(0.5, -0.05))
    plt.suptitle("Accuracy vs Completion Tokens", fontsize=13)
    plt.tight_layout()
    out = Path(output_dir) / "accuracy_vs_tokens.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_efficiency_heatmap(df: pd.DataFrame, output_dir: str):
    """
    Heatmap of completion token ratio vs English,
    rows = (task, model), cols = language.
    """
    ratio_df = compute_ratio_table(df, "mean_completion_tokens")

    # Reorder columns: en first then rest
    cols = [c for c in ["en", "zh", "hi", "ar", "es", "tr"] if c in ratio_df.columns]
    ratio_df = ratio_df[cols]

    fig, ax = plt.subplots(figsize=(10, max(4, len(ratio_df) * 0.55)))
    mask = ratio_df.isnull()
    sns.heatmap(
        ratio_df, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn_r",
        center=1.0, vmin=0.5, vmax=1.5, linewidths=0.5,
        mask=mask, cbar_kws={"label": "Token Ratio vs English"}
    )
    ax.set_title("Completion Token Ratio vs English (< 1.0 = fewer tokens)", fontsize=12)
    ax.set_xlabel("Language")
    ax.set_ylabel("Model / Task")
    plt.tight_layout()
    out = Path(output_dir) / "efficiency_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_ceff_comparison(df: pd.DataFrame, output_dir: str):
    """Bar chart of Ceff (expected cost per successful task) by language."""
    en_df = df[df["language"] == "en"][["model", "task", "ceff_usd"]].rename(
        columns={"ceff_usd": "en_ceff"}
    )
    merged = df.merge(en_df, on=["model", "task"])
    merged["ceff_ratio"] = merged["ceff_usd"] / merged["en_ceff"]

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = merged.pivot_table(index=["model", "task"], columns="language",
                                values="ceff_ratio").round(3)
    cols = [c for c in ["en", "zh", "hi", "ar", "es", "tr"] if c in pivot.columns]
    pivot[cols].plot(kind="bar", ax=ax, width=0.7)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="English baseline")
    ax.set_title("Cost-Efficiency Ratio (Ceff) vs English\n(< 1.0 = cheaper per successful task)")
    ax.set_ylabel("Ceff Ratio")
    ax.set_xlabel("Model / Task")
    ax.legend(title="Language", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    out = Path(output_dir) / "ceff_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Tables ────────────────────────────────────────────────────────────────────

def print_tables(df: pd.DataFrame, output_dir: str):
    out_dir = Path(output_dir)

    # Table 1: completion token ratios
    t1 = compute_ratio_table(df, "mean_completion_tokens")
    t1.to_csv(out_dir / "table_token_ratios.csv")
    print("\n=== Table 1: Completion Token Ratio vs English ===")
    print(t1.to_string())

    # Table 2: accuracy
    t2 = df.pivot_table(index=["model", "task"], columns="language",
                         values="accuracy").round(3)
    t2.to_csv(out_dir / "table_accuracy.csv")
    print("\n=== Table 2: Accuracy ===")
    print(t2.to_string())

    # Table 3: Ceff ratios
    en_df = df[df["language"] == "en"][["model", "task", "ceff_usd"]].rename(
        columns={"ceff_usd": "en_ceff"}
    )
    merged = df.merge(en_df, on=["model", "task"])
    merged["ceff_ratio"] = (merged["ceff_usd"] / merged["en_ceff"]).round(3)
    t3 = merged.pivot_table(index=["model", "task"], columns="language",
                             values="ceff_ratio")
    t3.to_csv(out_dir / "table_ceff_ratios.csv")
    print("\n=== Table 3: Ceff Ratio vs English ===")
    print(t3.to_string())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--plots_dir",   default="results/plots")
    args = parser.parse_args()

    Path(args.plots_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {args.results_dir}")
    df = load_summary_df(args.results_dir)
    print(f"  Loaded {len(df)} run records.\n")

    print_tables(df, args.results_dir)

    print("\nGenerating plots...")
    plot_token_bars(df, args.plots_dir)
    plot_accuracy_vs_tokens(df, args.plots_dir)
    plot_efficiency_heatmap(df, args.plots_dir)
    plot_ceff_comparison(df, args.plots_dir)

    print("\nDone. Tables in results/, plots in results/plots/")


if __name__ == "__main__":
    main()