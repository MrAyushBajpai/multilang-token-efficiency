"""
recompute_reasoning_length.py
==============================
The stored `reasoning_tokens` field is `len(reasoning_text.split())` --
a whitespace word count. That's a reasonable rough proxy for English,
but breaks badly for languages that don't use spaces between words
(Chinese, Japanese, etc.): e.g. qwen3-32b/commonsense/zh comes out to
a mean of ~7.5 "reasoning tokens" vs ~250-285 for every other language,
purely because Chinese text has almost no whitespace -- not because
the model reasoned less.

This field is NOT used anywhere in analyze.py's current tables/plots
(those use the API's real `completion_tokens`/`total_tokens`), so this
script doesn't change any existing figure. It only matters if/when you
want to report "how much does the model reason, per language".

This script recomputes a script-aware segment count:
  - each CJK / Hiragana / Katakana / Hangul character counts as 1 segment
    (these scripts are tokenized roughly char-by-char by most BPE
    tokenizers, so this is a much closer proxy than whitespace splitting)
  - runs of other non-space characters count as 1 segment each (≈ a "word",
    same as the original .split() behaviour for space-delimited scripts)

This is still an APPROXIMATION, not the true Qwen3 tokenizer count.
If you have internet access to Hugging Face, the gold-standard fix is:

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
    reasoning_tokens_v2 = len(tok.encode(reasoning_text))

...applied to the `reasoning` field of every qwen3-32b record. The
script-aware proxy below is the no-internet-required fallback.

Output: results/table_reasoning_length_v2.csv  (mean per model/task/lang,
        old whitespace-count vs new script-aware count)
Also writes results_regraded_reasoning/<file>.jsonl with a new field
`reasoning_length_v2` per record.
"""

import json
import re
import argparse
from pathlib import Path
import pandas as pd

# One CJK/Kana/Hangul char = 1 segment; runs of everything else
# (non-whitespace) = 1 segment, same granularity as .split().
SEGMENT_RE = re.compile(
    r'[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\u31f0-\u31ff\uac00-\ud7a3]'
    r'|[^\s\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\u31f0-\u31ff\uac00-\ud7a3]+'
)


def script_aware_length(text: str) -> int:
    if not text:
        return 0
    return len(SEGMENT_RE.findall(text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--out_dir", default="results_regraded_reasoning")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in sorted(Path(args.results_dir).glob("qwen3-32b__*.jsonl")):
        records = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        if not records:
            continue
        model = records[0]["model"]; task = records[0]["task"]; lang = records[0]["language"]

        for r in records:
            r["reasoning_length_v2"] = script_aware_length(r.get("reasoning", ""))

        with open(out_dir / f.name, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        old_vals = [r["reasoning_tokens"] for r in records]
        new_vals = [r["reasoning_length_v2"] for r in records]
        rows.append({
            "model": model, "task": task, "language": lang,
            "n": len(records),
            "mean_reasoning_tokens_old_wordsplit": round(sum(old_vals) / len(old_vals), 1),
            "mean_reasoning_length_v2_script_aware": round(sum(new_vals) / len(new_vals), 1),
            "mean_completion_tokens": round(
                sum(r["completion_tokens"] for r in records) / len(records), 1
            ),
        })

    df = pd.DataFrame(rows).sort_values(["model", "task", "language"])
    out = Path(args.results_dir) / "table_reasoning_length_v2.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote: {out}")
    print(f"Wrote per-record JSONL with 'reasoning_length_v2' to: {out_dir}/")


if __name__ == "__main__":
    main()