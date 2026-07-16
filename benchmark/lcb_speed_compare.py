#!/usr/bin/env python3
"""Side-by-side compare of two (or more) lcb_speed aiperf runs.

    python3 benchmark/lcb_speed_compare.py \
        artifacts/dflash/lcb_speed artifacts/dflash_ngram/lcb_speed

Reads profile_export_aiperf.csv from each dir. The ratio column is
run2/run1 (>1 = faster for throughput rows, slower for latency rows).
Warns if avg OSL differs >1% between runs - at temp 0 spec decoding is
lossless, so a mismatch means the runs did not decode the same tokens
and the speed comparison is not apples-to-apples.
"""

import csv
import sys
from pathlib import Path

ROWS = [
    "Time to First Token (ms)",
    "Time to Second Token (ms)",
    "Inter Token Latency (ms)",
    "Request Latency (ms)",
    "Output Token Throughput Per User (tokens/sec/user)",
    "Output Sequence Length (tokens)",
]
COLS = ["avg", "p50", "p99"]


def load(art_dir):
    path = Path(art_dir) / "profile_export_aiperf.csv"
    stats = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            metric = row.get("Metric")
            if metric in ROWS:
                stats[metric] = {c: float(row[c]) for c in COLS if row.get(c)}
    missing = [r for r in ROWS if r not in stats]
    if missing:
        sys.exit(f"{path}: missing metric rows {missing}")
    return stats


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    dirs = sys.argv[1:]
    names = [Path(d).parent.name for d in dirs]  # artifacts/<tag>/lcb_speed -> tag
    runs = [load(d) for d in dirs]

    w = max(len(r) for r in ROWS)
    header = f"{'metric':<{w}}  {'col':>4}" + "".join(f"  {n:>14}" for n in names)
    if len(runs) == 2:
        header += f"  {names[1] + '/' + names[0]:>16}"
    print(header)
    print("-" * len(header))
    for metric in ROWS:
        for c in COLS:
            vals = [r[metric][c] for r in runs]
            line = f"{metric if c == 'avg' else '':<{w}}  {c:>4}" + "".join(f"  {v:>14.2f}" for v in vals)
            if len(runs) == 2 and vals[0]:
                line += f"  {vals[1] / vals[0]:>15.3f}x"
            print(line)

    osl = [r["Output Sequence Length (tokens)"]["avg"] for r in runs]
    if max(osl) and (max(osl) - min(osl)) / max(osl) > 0.01:
        print(f"\nWARNING: avg OSL differs >1% across runs ({osl}) - "
              "decodes were not identical, speed comparison is suspect")
    else:
        print("\nOSL check: outputs match across runs (lossless spec decode confirmed)")


if __name__ == "__main__":
    main()
