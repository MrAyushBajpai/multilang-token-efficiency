"""
logger.py
---------
Saves each run's raw records as JSON and appends a summary CSV.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


class ExperimentLogger:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.results_dir / "summary.csv"
        self._init_csv()

    def _init_csv(self):
        if not self.summary_path.exists():
            with open(self.summary_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "run_key", "model", "task", "language",
                    "n", "n_correct", "accuracy",
                    "mean_completion_tokens", "median_completion_tokens",
                    "std_completion_tokens", "mean_total_tokens",
                    "mean_latency_s", "avg_cost_per_attempt_usd", "ceff_usd",
                ])

    def save_run(self, run_key: str, records: List[Dict[str, Any]]):
        # Save raw records
        out_path = self.results_dir / f"{run_key}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Append summary row
        from scripts.metrics import compute_metrics
        m = compute_metrics(records)
        if not m or not records:
            return

        parts = run_key.split("__")
        model, task, lang = parts if len(parts) == 3 else ("?", "?", "?")

        with open(self.summary_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(),
                run_key, model, task, lang,
                m["n"], m["n_correct"], round(m["accuracy"], 4),
                round(m["mean_completion_tokens"], 2),
                round(m["median_completion_tokens"], 2),
                round(m["std_completion_tokens"], 2),
                round(m["mean_total_tokens"], 2),
                round(m["mean_latency_s"], 3),
                round(m["avg_cost_per_attempt_usd"], 6),
                round(m["ceff_usd"], 6),
            ])