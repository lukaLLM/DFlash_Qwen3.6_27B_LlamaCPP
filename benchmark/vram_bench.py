#!/usr/bin/env python3
"""VRAM footprint bench - base vs DFlash vs DFlash+ngram.

For each service in docker/docker-compose.yaml (all bind :8001, so they run
one at a time, compose defaults -> ctx 256000, same as the speed sweep):

  stop all llama services -> record idle -> up -> wait /health ->
  record steady after-load -> run 3 prompts (code, long generation,
  repetitive context so the n-gram drafters actually fire) -> record
  inference peak -> stop -> wait for VRAM release.

nvidia-smi is sampled every 0.25 s for the whole cycle. "compute_mb" is the
sum of compute-app memory on the GPU (the llama-server process), which
excludes the desktop's graphics allocations; "total_mb" is the raw
memory.used for sanity.

Outputs (benchmark/bench_results/vram/):
  vram_timeline_<service>.csv   raw samples with phase labels
  vram_footprint.csv            summary table, one row per service
  vram_details.json             per-prompt timings from the server
"""

import argparse
import csv
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMPOSE = REPO / "docker" / "docker-compose.yaml"
OUT_DIR = REPO / "benchmark" / "bench_results" / "vram"
BASE_URL = "http://localhost:8001"
GPU_INDEX = "0"
SAMPLE_PERIOD_S = 0.25
HEALTH_TIMEOUT_S = 900
RELEASE_TIMEOUT_S = 180
SETTLE_S = 10

ALL_SERVICES = [
    "llamacpp_baseline",
    "llamacpp_dflash",
    "llamacpp_dflash_ngram",
    "llama_cpp_qwen36_mtp",
]

# (key, compose service, human label)
BENCH_SERVICES = [
    ("base", "llamacpp_baseline", "base"),
    ("dflash", "llamacpp_dflash", "DFlash"),
    ("dflash_ngram", "llamacpp_dflash_ngram", "DFlash+ngram"),
]

FIB_PROMPT = (
    "Write a complete, well-commented C++ program that prints the first 50 "
    "Fibonacci numbers using an iterative algorithm, then explain its time "
    "complexity."
)
STORY_PROMPT = (
    "Write a long, detailed short story about a lighthouse keeper who "
    "discovers a message in a bottle. Include dialogue and descriptions."
)


def make_repetitive_prompt() -> str:
    """~2.5k tokens of near-identical code blocks - prime n-gram territory."""
    block = "\n".join(
        f"def handler_{i:03d}(request):\n"
        f"    payload = validate(request.json, schema=SCHEMA_{i:03d})\n"
        f"    record = db.session.query(Model_{i:03d}).filter_by(id=payload['id']).first()\n"
        f"    if record is None:\n"
        f"        return jsonify(error='not found'), 404\n"
        f"    record.update(payload)\n"
        f"    db.session.commit()\n"
        f"    return jsonify(record.to_dict()), 200\n"
        for i in range(40)
    )
    return (
        "Here is a Flask API module:\n\n" + block +
        "\n\nContinue the module with handler_040 through handler_049 in "
        "exactly the same style."
    )


PROMPTS = [
    ("code_512", FIB_PROMPT, 512),
    ("story_2048", STORY_PROMPT, 2048),
    ("repeat_ctx_512", make_repetitive_prompt(), 512),
]


def nvsmi(args: list[str]) -> str:
    return subprocess.run(
        ["nvidia-smi", "-i", GPU_INDEX] + args,
        capture_output=True, text=True, timeout=10,
    ).stdout.strip()


def read_vram() -> tuple[int, int]:
    total = nvsmi(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    apps = nvsmi(["--query-compute-apps=used_memory", "--format=csv,noheader,nounits"])
    total_mb = int(total) if total else 0
    compute_mb = sum(int(x) for x in apps.splitlines() if x.strip().isdigit())
    return total_mb, compute_mb


class Sampler(threading.Thread):
    """Samples VRAM every SAMPLE_PERIOD_S, tagging each sample with a phase."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []  # (t, phase, total_mb, compute_mb)
        self.phase = "init"
        self.t0 = time.monotonic()
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def set_phase(self, phase: str):
        with self._lock:
            self.phase = phase

    def run(self):
        while not self._stop.is_set():
            try:
                total, compute = read_vram()
            except Exception:
                total, compute = -1, -1
            with self._lock:
                self.samples.append(
                    (time.monotonic() - self.t0, self.phase, total, compute)
                )
            self._stop.wait(SAMPLE_PERIOD_S)

    def stop(self):
        self._stop.set()

    def phase_stats(self, phase_prefix: str, stat: str = "max") -> int:
        with self._lock:
            vals = [c for _, p, _, c in self.samples
                    if p.startswith(phase_prefix) and c >= 0]
        if not vals:
            return -1
        if stat == "max":
            return max(vals)
        if stat == "median":
            return int(statistics.median(vals))
        raise ValueError(stat)

    def last_median(self, phase: str, n: int = 20) -> int:
        with self._lock:
            vals = [c for _, p, _, c in self.samples if p == phase and c >= 0]
        vals = vals[-n:]
        return int(statistics.median(vals)) if vals else -1


def compose(*args: str):
    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE)] + list(args),
        check=True, capture_output=True, text=True, timeout=300,
    )


def wait_health(timeout_s: float) -> float:
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        try:
            with urllib.request.urlopen(BASE_URL + "/health", timeout=5) as r:
                if r.status == 200:
                    return time.monotonic() - start
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(3)
    raise TimeoutError(f"server not healthy after {timeout_s}s")


def wait_release(timeout_s: float):
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        _, compute = read_vram()
        if compute < 500:
            time.sleep(3)  # one extra beat for the allocator to finish
            return
        time.sleep(2)
    print(f"WARN: compute VRAM still {compute} MiB after {timeout_s}s", flush=True)


def run_prompt(prompt: str, n_predict: int) -> dict:
    body = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_k": 1,
        "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(
        BASE_URL + "/completion", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read())
    t = out.get("timings", {})
    return {
        "tokens_predicted": out.get("tokens_predicted"),
        "prompt_n": t.get("prompt_n"),
        "predicted_per_second": round(t.get("predicted_per_second", 0), 2),
        "draft_n": t.get("draft_n"),
        "draft_n_accepted": t.get("draft_n_accepted"),
    }


def bench_service(key: str, service: str, label: str) -> dict:
    print(f"\n=== {label} ({service}) ===", flush=True)
    sampler = Sampler()
    sampler.start()
    row = {"service": key, "label": label}
    details = {"prompts": {}}
    try:
        sampler.set_phase("idle")
        compose("stop", *ALL_SERVICES)
        wait_release(RELEASE_TIMEOUT_S)
        time.sleep(3)
        row["idle_mb"] = sampler.last_median("idle", n=10)
        print(f"idle compute VRAM: {row['idle_mb']} MiB", flush=True)

        sampler.set_phase("load")
        compose("up", "-d", service)
        load_s = wait_health(HEALTH_TIMEOUT_S)
        details["load_seconds"] = round(load_s, 1)
        print(f"healthy after {load_s:.0f}s", flush=True)
        time.sleep(SETTLE_S)

        sampler.set_phase("after_load")
        time.sleep(5)
        row["load_peak_mb"] = sampler.phase_stats("load", "max")
        row["after_load_mb"] = sampler.last_median("after_load")

        for name, prompt, n_predict in PROMPTS:
            sampler.set_phase(f"infer_{name}")
            print(f"prompt {name} ...", flush=True, end=" ")
            info = run_prompt(prompt, n_predict)
            details["prompts"][name] = info
            print(f"{info['predicted_per_second']} tok/s "
                  f"(draft acc: {info['draft_n_accepted']}/{info['draft_n']})",
                  flush=True)

        sampler.set_phase("after_infer")
        time.sleep(5)
        row["infer_peak_mb"] = sampler.phase_stats("infer_", "max")
        row["after_infer_mb"] = sampler.last_median("after_infer")
    finally:
        sampler.set_phase("teardown")
        try:
            compose("stop", service)
            wait_release(RELEASE_TIMEOUT_S)
        finally:
            sampler.stop()
            sampler.join(timeout=5)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"vram_timeline_{key}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "phase", "total_mb", "compute_mb"])
        for t, phase, total, computed in sampler.samples:
            w.writerow([f"{t:.2f}", phase, total, computed])

    row["overhead_vs_idle_mb"] = row["after_load_mb"] - row["idle_mb"]
    return row, details


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--services", nargs="*", default=[k for k, _, _ in BENCH_SERVICES],
                    help="subset of service keys to run")
    args = ap.parse_args()

    todo = [(k, s, l) for k, s, l in BENCH_SERVICES if k in args.services]
    rows, all_details = [], {}
    for key, service, label in todo:
        row, details = bench_service(key, service, label)
        rows.append(row)
        all_details[key] = details
        print(json.dumps(row, indent=2), flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["service", "label", "idle_mb", "load_peak_mb", "after_load_mb",
            "infer_peak_mb", "after_infer_mb", "overhead_vs_idle_mb"]
    with open(OUT_DIR / "vram_footprint.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(OUT_DIR / "vram_details.json", "w") as f:
        json.dump(all_details, f, indent=2)
    print(f"\nwrote {OUT_DIR}/vram_footprint.csv", flush=True)


if __name__ == "__main__":
    main()
