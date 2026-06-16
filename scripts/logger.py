"""
logger.py
---------
Thread-safe, Ctrl-C-safe logger for the experiment.

Design goals:
  1. Every completed question is written to disk immediately after the API
     call returns. A SIGINT (Ctrl-C) between questions only loses the
     question currently in-flight.
  2. Each run file is a JSONL: one JSON object per line. Partial files are
     still parseable — resume logic reads them to find which questions
     were already answered.
  3. A CSV summary row is appended once per completed run so downstream
     analysis doesn't have to re-aggregate JSONL files.
  4. Run state (RUNNING / DONE / FAILED) is tracked in a lightweight
     state file so the resume logic can skip completed runs and retry
     failed ones.
"""

import csv
import json
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


STATE_FILE = "run_state.json"   # relative to results_dir


class ExperimentLogger:
    def __init__(self, results_dir: str = "results"):
        self.results_dir  = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.results_dir / "summary.csv"
        self.state_path   = self.results_dir / STATE_FILE
        self._state: Dict[str, str] = self._load_state()
        self._init_csv()

    # ── State persistence ──────────────────────────────────────────────────

    def _load_state(self) -> Dict[str, str]:
        if self.state_path.exists():
            with open(self.state_path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_state(self) -> None:
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2)

    def mark_running(self, run_key: str) -> None:
        self._state[run_key] = "RUNNING"
        self._save_state()

    def mark_done(self, run_key: str) -> None:
        self._state[run_key] = "DONE"
        self._save_state()

    def mark_failed(self, run_key: str, reason: str = "") -> None:
        self._state[run_key] = f"FAILED:{reason}"
        self._save_state()

    def get_status(self, run_key: str) -> str:
        """Returns 'DONE', 'RUNNING', 'FAILED:<reason>', or 'NEW'."""
        return self._state.get(run_key, "NEW")

    def is_done(self, run_key: str) -> bool:
        return self._state.get(run_key, "") == "DONE"

    # ── Resume support ─────────────────────────────────────────────────────

    def completed_indices(self, run_key: str) -> set:
        """
        Return the set of question idx values already saved for this run.
        Used by the runner to skip questions already answered.
        """
        jsonl_path = self.results_dir / f"{run_key}.jsonl"
        if not jsonl_path.exists():
            return set()
        done = set()
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        done.add(r["idx"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        return done

    # ── Per-question write ─────────────────────────────────────────────────

    def append_record(self, run_key: str, record: Dict[str, Any]) -> None:
        """
        Appends one question record to the run's JSONL file immediately.
        Safe against Ctrl-C between questions.
        """
        jsonl_path = self.results_dir / f"{run_key}.jsonl"
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    # ── End-of-run summary ─────────────────────────────────────────────────

    def finalize_run(self, run_key: str) -> None:
        """
        Read the completed JSONL, compute metrics, append to summary CSV,
        and mark the run as DONE.
        """
        jsonl_path = self.results_dir / f"{run_key}.jsonl"
        if not jsonl_path.exists():
            self.mark_failed(run_key, "no_jsonl")
            return

        records = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        if not records:
            self.mark_failed(run_key, "empty_jsonl")
            return

        from scripts.metrics import compute_metrics
        m = compute_metrics(records)
        if not m:
            self.mark_failed(run_key, "metrics_failed")
            return

        parts = run_key.split("__")
        model, task, lang = parts if len(parts) == 3 else ("?", "?", "?")

        with open(self.summary_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                run_key, model, task, lang,
                m["n"], m["n_correct"], round(m["accuracy"], 6),
                round(m["mean_completion_tokens"], 2),
                round(m["median_completion_tokens"], 2),
                round(m["std_completion_tokens"], 2),
                round(m["p10_completion_tokens"], 2),
                round(m["p90_completion_tokens"], 2),
                round(m["mean_total_tokens"], 2),
                round(m["mean_prompt_tokens"], 2),
                round(m["mean_latency_s"], 3),
                round(m["median_latency_s"], 3),
                round(m["p90_latency_s"], 3),
                round(m["mean_response_chars"], 1),
                round(m["mean_fertility"], 4),
                m["n_truncated"],
                round(m["truncation_rate"], 6),
                round(m["avg_cost_per_attempt_usd"], 8),
                round(m["ceff_usd"], 8),
            ])

        self.mark_done(run_key)

    # ── CSV header ─────────────────────────────────────────────────────────

    def _init_csv(self) -> None:
        if not self.summary_path.exists():
            with open(self.summary_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "timestamp", "run_key", "model", "task", "language",
                    "n", "n_correct", "accuracy",
                    "mean_completion_tokens", "median_completion_tokens",
                    "std_completion_tokens", "p10_completion_tokens", "p90_completion_tokens",
                    "mean_total_tokens", "mean_prompt_tokens",
                    "mean_latency_s", "median_latency_s", "p90_latency_s",
                    "mean_response_chars", "mean_fertility",
                    "n_truncated", "truncation_rate",
                    "avg_cost_per_attempt_usd", "ceff_usd",
                ])