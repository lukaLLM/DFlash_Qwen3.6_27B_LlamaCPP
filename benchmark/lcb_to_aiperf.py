#!/usr/bin/env python3
"""Extract LiveCodeBench problems into an aiperf inputs-json file.

Run with the LCB venv (needs datasets<4, python 3.11):
    LiveCodeBench/.venv/bin/python benchmark/lcb_to_aiperf.py --limit 100

Order matches the official harness exactly: date filter, then STRING sort on
question_id (lcb_runner/runner/scenario_router.py). Same file -> same prompts
in the same order on every server, so dflash vs dflash-ngram is a clean A/B.

aiperf sends inputs-json payloads verbatim (raw-payload path bypasses --model
and --extra-inputs), so sampling params live IN each payload and the "model"
field is a placeholder the runner patches to the live server alias.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INVOKE_CWD = Path.cwd()
sys.path.insert(0, str(REPO / "LiveCodeBench"))
# lcb_runner.prompts opens few-shot files by relative path at import time.
os.chdir(REPO / "LiveCodeBench")

from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
from lcb_runner.lm_styles import LMStyle
from lcb_runner.prompts.code_generation import format_prompt_generation

MODEL_PLACEHOLDER = "MODEL_PLACEHOLDER"


def payload(messages, model, max_tokens):
    # Greedy + stream:true embedded here because aiperf won't inject them.
    return {
        "messages": messages,
        "model": model,
        "stream": True,
        "max_completion_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1.0,
        "top_k": 1,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--release", default="release_v5")
    ap.add_argument("--start-date", default=None, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--end-date", default=None, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--limit", type=int, default=100, help="first N problems in official order (0 = all)")
    ap.add_argument("--warmup-entries", type=int, default=1,
                    help="copies of problem 0 prepended as dedicated warmup slots")
    ap.add_argument("--max-completion-tokens", type=int, default=36864,
                    help="per-request cap; 36864 never truncates within ctx 40960")
    ap.add_argument("--out", default=None,
                    help="output path (default benchmark/data/lcb_<release>_first<N>.inputs.json)")
    args = ap.parse_args()

    problems = load_code_generation_dataset(args.release, args.start_date, args.end_date)
    problems = sorted(problems, key=lambda x: x.question_id)  # official order (string sort)
    if args.limit:
        problems = problems[: args.limit]
    if not problems:
        sys.exit("no problems matched the filter")

    data = []
    for w in range(args.warmup_entries):
        msgs = format_prompt_generation(problems[0], LMStyle.OpenAIChat)
        data.append({"session_id": f"warmup_{w:03d}",
                     "payloads": [payload(msgs, MODEL_PLACEHOLDER, args.max_completion_tokens)]})
    for i, p in enumerate(problems):
        msgs = format_prompt_generation(p, LMStyle.OpenAIChat)
        data.append({"session_id": f"q{i:03d}_{p.question_id}",
                     "payloads": [payload(msgs, MODEL_PLACEHOLDER, args.max_completion_tokens)]})

    out = (INVOKE_CWD / args.out) if args.out else \
        REPO / "benchmark" / "data" / f"lcb_{args.release}_first{len(problems)}.inputs.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = {
        # aiperf reads only "data"; the manifest is for the runner + humans.
        "manifest": {
            "release": args.release,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "num_problems": len(problems),
            "warmup_entries": args.warmup_entries,
            "max_completion_tokens": args.max_completion_tokens,
            "question_ids": [p.question_id for p in problems],
        },
        "data": data,
    }
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}  ({args.warmup_entries} warmup + {len(problems)} problems)")


if __name__ == "__main__":
    main()
