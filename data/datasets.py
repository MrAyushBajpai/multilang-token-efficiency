"""
datasets.py
-----------
Downloads benchmark questions ONCE and caches them locally so that
every language condition and every run uses the EXACT same questions
in the EXACT same order. This is a hard requirement for McNemar's test
(which needs matched pairs across languages).

Cache location: data/cache/<task>_<n>.jsonl
If the cache file exists it is loaded directly — no network call.

Tasks:
  math        → GSM8k (openai/gsm8k, test split)
  commonsense → ARC-Easy (allenai/ai2_arc ARC-Easy, test split)
  code        → MBPP (Muennighoff/mbpp, "full" config, test split, 974 problems)

Fallback hardcoded banks (20 items each) are used when the HF
datasets package is unavailable or the download fails.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

# Where caches live (relative to this file's directory)
CACHE_DIR = Path(__file__).parent / "cache"

# ── Fallback problem banks (kept at 20 to allow flexible repetition) ──────────

MATH_PROBLEMS = [
    {"id": "m01", "question": "What is 15% of 240?", "answer": "36"},
    {"id": "m02", "question": "A train travels 360 km in 4 hours. What is its speed in km/h?", "answer": "90"},
    {"id": "m03", "question": "Solve for x: 2x + 7 = 19", "answer": "6"},
    {"id": "m04", "question": "What is the area of a circle with radius 7? (use π ≈ 3.14159)", "answer": "153.938"},
    {"id": "m05", "question": "A rectangle has perimeter 54 cm and width 9 cm. What is its length?", "answer": "18"},
    {"id": "m06", "question": "What is 3/8 + 5/12 expressed as a fraction in lowest terms?", "answer": "19/24"},
    {"id": "m07", "question": "If f(x) = 2x^2 - 3x + 1, what is f(4)?", "answer": "21"},
    {"id": "m08", "question": "How many prime numbers are there between 10 and 30?", "answer": "5"},
    {"id": "m09", "question": "A store sells an item for $84 after a 30% discount. What was the original price?", "answer": "120"},
    {"id": "m10", "question": "What is the sum of interior angles of a hexagon?", "answer": "720"},
    {"id": "m11", "question": "Solve: log₂(32) = ?", "answer": "5"},
    {"id": "m12", "question": "Two numbers have sum 47 and difference 13. What is the larger number?", "answer": "30"},
    {"id": "m13", "question": "What is the LCM of 12, 15, and 20?", "answer": "60"},
    {"id": "m14", "question": "Evaluate: 7! / (5! × 2!)", "answer": "21"},
    {"id": "m15", "question": "The hypotenuse of a right triangle is 13 and one leg is 5. What is the other leg?", "answer": "12"},
    {"id": "m16", "question": "What is the median of: 3, 7, 2, 9, 4, 7, 1?", "answer": "4"},
    {"id": "m17", "question": "Convert 0.375 to a fraction in lowest terms.", "answer": "3/8"},
    {"id": "m18", "question": "A geometric sequence has first term 3 and common ratio 2. What is the 6th term?", "answer": "96"},
    {"id": "m19", "question": "If a car depreciates 15% per year, what fraction of its value remains after 2 years?", "answer": "0.7225"},
    {"id": "m20", "question": "Simplify: (3x^2 + 2x - 5) + (x^2 - 4x + 3)", "answer": "4x^2 - 2x - 2"},
]

COMMONSENSE_PROBLEMS = [
    {"id": "c01", "question": "Which of the following is NOT a primary color?\nA) Red\nB) Blue\nC) Green\nD) Yellow", "answer": "C"},
    {"id": "c02", "question": "What happens to water when it is heated to 100°C at sea level?\nA) It freezes\nB) It boils\nC) It becomes denser\nD) It turns into a solid", "answer": "B"},
    {"id": "c03", "question": "Which planet is closest to the Sun?\nA) Venus\nB) Earth\nC) Mercury\nD) Mars", "answer": "C"},
    {"id": "c04", "question": "What is the main purpose of a dictionary?\nA) To store money\nB) To define words\nC) To calculate numbers\nD) To translate languages", "answer": "B"},
    {"id": "c05", "question": "If it is currently winter in Australia, what season is it in Canada?\nA) Summer\nB) Autumn\nC) Winter\nD) Spring", "answer": "C"},
    {"id": "c06", "question": "Which material conducts electricity best?\nA) Wood\nB) Rubber\nC) Copper\nD) Glass", "answer": "C"},
    {"id": "c07", "question": "A doctor prescribes medicine twice a day. How many times in a week?\nA) 7\nB) 14\nC) 21\nD) 2", "answer": "B"},
    {"id": "c08", "question": "What is the term for animals that eat both plants and meat?\nA) Carnivores\nB) Herbivores\nC) Omnivores\nD) Decomposers", "answer": "C"},
    {"id": "c09", "question": "Which of the following is a renewable energy source?\nA) Coal\nB) Natural gas\nC) Solar power\nD) Petroleum", "answer": "C"},
    {"id": "c10", "question": "What does a barometer measure?\nA) Temperature\nB) Humidity\nC) Atmospheric pressure\nD) Wind speed", "answer": "C"},
    {"id": "c11", "question": "Which organ pumps blood through the human body?\nA) Lungs\nB) Liver\nC) Brain\nD) Heart", "answer": "D"},
    {"id": "c12", "question": "If you fold a square piece of paper in half twice, how many layers?\nA) 2\nB) 3\nC) 4\nD) 8", "answer": "C"},
    {"id": "c13", "question": "Which is heavier: 1 kg of feathers or 1 kg of iron?\nA) Iron\nB) Feathers\nC) They weigh the same\nD) Depends on volume", "answer": "C"},
    {"id": "c14", "question": "A car faces north and turns right twice. Which direction does it now face?\nA) North\nB) South\nC) East\nD) West", "answer": "B"},
    {"id": "c15", "question": "What process do plants use to make their own food using sunlight?\nA) Respiration\nB) Fermentation\nC) Photosynthesis\nD) Digestion", "answer": "C"},
    {"id": "c16", "question": "If today is Wednesday and an event is in 10 days, what day will it be?\nA) Friday\nB) Saturday\nC) Sunday\nD) Monday", "answer": "B"},
    {"id": "c17", "question": "Which of the following is NOT a mammal?\nA) Dolphin\nB) Bat\nC) Salmon\nD) Whale", "answer": "C"},
    {"id": "c18", "question": "If a recipe needs 2 cups of flour for 12 cookies, how much for 36 cookies?\nA) 4 cups\nB) 6 cups\nC) 3 cups\nD) 8 cups", "answer": "B"},
    {"id": "c19", "question": "A person has 3 apples and gives away 2. How many do they have?\nA) 5\nB) 1\nC) 0\nD) 3", "answer": "B"},
    {"id": "c20", "question": "What is 1 kg of cotton compared to 1 kg of gold?\nA) Lighter\nB) Heavier\nC) The same weight\nD) Impossible to compare", "answer": "C"},
]

CODE_PROBLEMS = [
    {"id": "p01", "question": "Write a Python function `sum_list(lst)` that returns the sum of all numbers in a list.", "answer": "def sum_list"},
    {"id": "p02", "question": "Write a Python function `is_palindrome(s)` that returns True if a string is a palindrome.", "answer": "def is_palindrome"},
    {"id": "p03", "question": "Write a Python function `fibonacci(n)` that returns the nth Fibonacci number (0-indexed).", "answer": "def fibonacci"},
    {"id": "p04", "question": "Write a Python function `count_vowels(s)` that returns the number of vowels in a string.", "answer": "def count_vowels"},
    {"id": "p05", "question": "Write a Python function `flatten(lst)` that flattens a list of lists into a single list.", "answer": "def flatten"},
    {"id": "p06", "question": "Write a Python function `remove_duplicates(lst)` that removes duplicate values while preserving order.", "answer": "def remove_duplicates"},
    {"id": "p07", "question": "Write a Python function `binary_search(arr, target)` that returns the index of target in a sorted array, or -1 if not found.", "answer": "def binary_search"},
    {"id": "p08", "question": "Write a Python function `word_frequency(text)` that returns a dictionary of word frequencies.", "answer": "def word_frequency"},
    {"id": "p09", "question": "Write a Python function `is_prime(n)` that returns True if n is a prime number.", "answer": "def is_prime"},
    {"id": "p10", "question": "Write a Python function `rotate_list(lst, k)` that rotates a list by k positions to the right.", "answer": "def rotate_list"},
    {"id": "p11", "question": "Write a Python function `merge_sorted(a, b)` that merges two sorted lists into a single sorted list.", "answer": "def merge_sorted"},
    {"id": "p12", "question": "Write a Python function `matrix_transpose(m)` that returns the transpose of a 2D matrix.", "answer": "def matrix_transpose"},
    {"id": "p13", "question": "Write a Python function `group_anagrams(words)` that groups a list of words by anagram.", "answer": "def group_anagrams"},
    {"id": "p14", "question": "Write a Python function `longest_common_prefix(strs)` that finds the longest common prefix in a list of strings.", "answer": "def longest_common_prefix"},
    {"id": "p15", "question": "Write a Python function `two_sum(nums, target)` that returns indices of two numbers that add up to target.", "answer": "def two_sum"},
    {"id": "p16", "question": "Write a Python function `valid_brackets(s)` that returns True if brackets in a string are properly balanced.", "answer": "def valid_brackets"},
    {"id": "p17", "question": "Write a Python function `run_length_encode(s)` that encodes a string using run-length encoding.", "answer": "def run_length_encode"},
    {"id": "p18", "question": "Write a Python function `max_subarray_sum(nums)` that returns the maximum subarray sum (Kadane's algorithm).", "answer": "def max_subarray_sum"},
    {"id": "p19", "question": "Write a Python function `gcd(a, b)` that returns the greatest common divisor of two integers.", "answer": "def gcd"},
    {"id": "p20", "question": "Write a Python function `caesar_cipher(text, shift)` that encodes text with a Caesar cipher.", "answer": "def caesar_cipher"},
]

_FALLBACK_BANKS = {
    "math":        MATH_PROBLEMS,
    "commonsense": COMMONSENSE_PROBLEMS,
    "code":        CODE_PROBLEMS,
}


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(task: str, n: int) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{task}_{n}.jsonl"


def _save_cache(task: str, problems: List[Dict[str, Any]]) -> None:
    path = _cache_path(task, len(problems))
    with open(path, "w", encoding="utf-8") as f:
        for p in problems:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  [dataset] Cached {len(problems)} {task} problems → {path}")


def _load_cache(task: str, n: int) -> List[Dict[str, Any]] | None:
    path = _cache_path(task, n)
    if not path.exists():
        return None
    problems = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    if len(problems) == n:
        print(f"  [dataset] Loaded {n} {task} problems from cache ({path}).")
        return problems
    print(f"  [dataset] Cache size mismatch ({len(problems)} vs {n}), re-downloading.")
    return None


# ── HuggingFace loaders ───────────────────────────────────────────────────────

def _load_from_hf(task: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """Try to download from HuggingFace. Returns [] on any failure."""
    from datasets import load_dataset  # type: ignore

    if task == "math":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        return [
            {"id": f"gsm8k_{i}", "question": r["question"],
             "answer": r["answer"].split("####")[-1].strip()}
            for i, r in enumerate(ds)
        ]

    elif task == "commonsense":
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        problems = []
        for i, r in enumerate(ds):
            choices = r["choices"]["text"]
            labels  = r["choices"]["label"]
            options = "\n".join(f"{l}) {t}" for l, t in zip(labels, choices))
            problems.append({
                "id":       r["id"],
                "question": f"{r['question']}\n{options}",
                "answer":   r["answerKey"],
            })
        return problems

    elif task == "code":
        # MBPP (Mostly Basic Python Problems) — 974 problems, well above the
        # 500-sample default. Using Muennighoff/mbpp which is the community
        # mirror with confirmed availability (google-research-datasets/mbpp
        # may require extra auth on some HF plan tiers).
        #
        # Fields used:
        #   task_id   → unique integer id
        #   text      → natural-language problem description (the prompt)
        #   code      → reference solution (entry-point fn name extracted)
        #   test_list → list of assert strings for pass/fail evaluation
        #
        # test_list is stored so downstream evaluators can execute the asserts
        # against the model's generated code rather than doing string matching.
        ds = load_dataset("Muennighoff/mbpp", "full", split="test")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        problems = []
        for r in ds:
            # Extract the entry-point function name from the reference solution.
            code = r["code"]
            if "def " in code:
                entry_point = code.split("def ")[1].split("(")[0].strip()
            else:
                entry_point = str(r["task_id"])
            problems.append({
                "id":        f"mbpp_{r['task_id']}",
                "question":  r["text"],
                "answer":    entry_point,
                "test_list": r["test_list"],          # list of assert strings
            })
        return problems

    return []


# ── Public interface ──────────────────────────────────────────────────────────

def get_dataset(task: str, n_samples: int = 500, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Returns exactly n_samples problems for the given task.

    Order is DETERMINISTIC and IDENTICAL across all calls with the same
    (task, n_samples, seed). The questions are downloaded once, cached to
    data/cache/<task>_<n_samples>.jsonl, and reused on every subsequent run.

    This guarantees McNemar's test validity: question i in language A
    corresponds to question i in language B for every model.
    """
    assert task in ("math", "commonsense", "code"), f"Unknown task: {task}"

    # 1. Try cache
    cached = _load_cache(task, n_samples)
    if cached is not None:
        return cached

    # 2. Try HuggingFace
    problems = []
    try:
        problems = _load_from_hf(task, n_samples, seed)
        if problems:
            print(f"  [dataset] Downloaded {len(problems)} {task} problems from HuggingFace.")
    except Exception as e:
        print(f"  [dataset] HF download failed ({e}), using fallback bank.")

    # 3. Fallback: repeat the hardcoded bank
    if not problems:
        bank = _FALLBACK_BANKS[task]
        rng  = random.Random(seed)
        while len(problems) < n_samples:
            chunk = bank[:]
            rng.shuffle(chunk)
            problems.extend(chunk)
        problems = problems[:n_samples]
        # Give unique IDs to repeated items
        seen: Dict[str, int] = {}
        for p in problems:
            base = p["id"]
            seen[base] = seen.get(base, 0) + 1
            if seen[base] > 1:
                p = dict(p)
                p["id"] = f"{base}_r{seen[base]}"
        print(f"  [dataset] Using fallback bank: {len(problems)} {task} problems.")

    # Trim to exactly n_samples and save
    problems = problems[:n_samples]
    _save_cache(task, problems)
    return problems