# DFlash

Qwen3.6-27B with **DFlash speculative decoding** on llama.cpp, benchmarked for
both **speed** (tok/s across context sizes) and **intelligence** (pass@1 on math
benchmarks) against a plain baseline and an MTP setup. Because greedy speculative
decoding is output-lossless, the questions these benchmarks answer are: *how much
faster is DFlash, and does it cost any accuracy?* Full setup, the "what broke and
how I fixed it" log, and the config knobs live in [`DFLASH.md`](DFLASH.md).

## YouTube

LOCAL AI / SPECULATIVE DECODING SERIES:

- Episode 1: DFlash on llama.cpp — _[title TBD]_ — _[YouTube link TBD]_

## Hardware

All numbers here are measured on a single card:

| | |
|---|---|
| **GPU** | NVIDIA RTX PRO 6000 Blackwell Workstation Edition (Blackwell, compute capability 12.0) |
| **VRAM** | 97 GB — target + draft both fit on one card |
| **Host** | Zen4 CPU |
| **llama.cpp** | `ghcr.io/ggml-org/llama.cpp:server-cuda13` (CUDA 13) |
| **GPU layout** | single GPU, `-ngl -1`, no tensor split |
| **Raw baseline** (no speculation) | **71 tok/s** generation (`llama bench` tg128) · **3877 tok/s** prompt processing (pp512) |

## Results summary

Consolidated numbers from every run so far. Machine-readable copy:
[`results_summary.csv`](results_summary.csv) (three labeled tables in one file).
Analysis and best findings: [`Insights.md`](Insights.md).

### Speed sweep — DFlash vs baseline

aiperf synthetic sweep (ISL = OSL, greedy, concurrency 1, 10 requests per size);
artifacts in `artifacts/{base,dflash}/speed/` and `reasoning/dflash/speed/`
(the latter = DFlash with reasoning on).

| Context (ISL=OSL) | base tok/s | DFlash tok/s | DFlash reasoning tok/s | **Speedup** | base ITL (ms) | DFlash ITL (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 67.62 | 97.13 | 96.61 | **1.44×** | 14.20 | 9.62 |
| 4 096 | 67.53 | 182.04 | 166.64 | **2.70×** | 14.46 | 5.08 |
| 12 288 | 64.78 | 220.12 | 240.41 | **3.40×** | 15.11 | 4.17 |
| 36 864 | 61.47 | **273.04** | — | **4.44×** | 15.91 | 3.26 |
| 98 304 | — | — | 241.44 | — | — | — |

The speedup **grows with context**: the baseline degrades from 67.6 → 61.5 tok/s
while DFlash climbs from 97 → 273 tok/s.

### Accuracy runs — pass@1 (temperature 0)

| Run (server alias) | Benchmark | N | Correct | Unparsed | pass@1 | gen tok/s | Valid? |
|---|---|---:|---:|---:|---:|---:|---|
| base (`qwen3.6-27B`) | math_500 | 200 | 49 | 0 | 24.50% | 71.37 | ⚠️ no — failed run (132 errors, 68/200 completed) |
| DFlash (`qwen36-dflash15`) | math_500 | 50 | 41 | 0 | 82.00% | 219.53 | ✅ yes (small N=50) |
| DFlash (`qwen36-dflash15`) | gsm8k | 50 | 25 | 50 | 50.00% | 187.91 | ❌ no — all unparsed (reasoning trap) |
| DFlash (`qwen36-dflash15`) | lcb-codegeneration | 3 | 0 | 0 | 0.00% | 145.15 | ❌ no — N=3 smoke test |
| DFlash (`qwen36-dflash15reason`) | math_500 | 200 | **144** | 0 | **72.00%** | 242.07 | ✅ yes — clean full run |
| DFlash (`qwen36-dflash15reason`) | gsm8k | 200 | 21 | 200 | 10.50% | 205.47 | ❌ no — all unparsed (reasoning trap) |

Aliases read from each run's `profile_export_aiperf.json`. Headline: the only
clean full-size MATH-500 run is **`qwen36-dflash15reason` = 72.00% (144/200,
0 unparsed)** — measured *while generating at 242 tok/s*. The baseline run
failed (132 connection errors), so there is **no valid baseline accuracy yet**;
rerun it before quoting any accuracy gap.

### Leaderboard runs (`benchmark/leaderboard_runs.csv`)

Fixed Fibonacci prompt, 10 requests per run, `n_max` = `--spec-draft-n-max`;
speedup vs the baseline row (70.34 tok/s). Separate measurement path from the
sweep above — see [Speed leaderboard](#speed-leaderboard-benchmarkleaderboardpy).

| Run | Type | n_max | avg tok/s | median | accept % | Speedup |
|---|---|---:|---:|---:|---:|---:|
| qwen3.6-27B | baseline | — | 70.34 | 70.38 | — | 1.00× |
| qwen36-dflash2 | dflash | 2 | 141.01 | 141.12 | 91.1 | 2.00× |
| qwen36-dflash4 | dflash | 4 | 178.34 | 179.65 | 80.4 | 2.54× |
| qwen36-dflash8 | dflash | 8 | 236.15 | 239.06 | 63.1 | 3.36× |
| **qwen36-dflash12** | dflash | 12 | **256.01** | 258.54 | 50.5 | **3.64×** |
| qwen36-dflash15 | dflash | 15 | 253.21 | 256.35 | 42.8 | 3.60× |
| qwen36-dflash15reason | dflash | 15 | 224.18 | 225.30 | 36.4 | 3.19× |
| qwen3.6-27b-mtp2 | mtp | 2 | 142.68 | 142.48 | 88.5 | 2.03× |
| qwen3.6-27b-mtp4 | mtp | 4 | 167.47 | 167.70 | 79.8 | 2.38× |
| qwen3.6-27b-mtp8 | mtp | 8 | 190.17 | 191.48 | 57.9 | 2.70× |
| qwen3.6-27b-mtp12 | mtp | 12 | 179.03 | 180.29 | 43.3 | 2.55× |
| qwen3.6-27b-mtp15 | mtp | 15 | 167.00 | 168.25 | 36.0 | 2.37× |
| qwen3.6-27b-mtp15reason | mtp | 15 | 163.12 | 163.60 | 34.4 | 2.32× |

Best config: **dflash12** — throughput peaks at n_max=12 and dips slightly at
15, and DFlash beats MTP at every draft length.

## Choosing a DFlash draft GGUF

This setup pairs the [unsloth Q4_K_XL target](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF)
with a **Q8_0 draft from a different repo** — that's fine, and here's what actually
matters when picking a drafter:

- **Quantization does NOT need to match.** The draft only *proposes* tokens; the
  target *verifies* every one. At temperature 0 the output is lossless — identical
  to what the target would produce alone — so a bad draft can only cost speed,
  never quality. A Q8_0 draft is a good default: it's ~1.9 GB, so the extra
  precision is nearly free and keeps acceptance high.
- **The base model / vocabulary MUST match.** Both GGUFs must derive from
  Qwen3.6-27B (same tokenizer, same token IDs); llama.cpp checks this at load.
- **The draft GGUF must report arch `dflash`.** Only conversions made with merged
  llama.cpp ([PR #22105](https://github.com/ggml-org/llama.cpp/pull/22105)) work.
  Pre-merge conversions report `dflash-draft` and fail with
  `unknown model architecture: 'dflash-draft'`.

Surveyed drafter repos for Qwen3.6-27B (arch checked via the HF API):

| Repo | Arch | Verdict |
|---|---|---|
| [Alittlehammmer/Qwen3.6-27B-DFlash-GGUF-llama.cpp](https://huggingface.co/Alittlehammmer/Qwen3.6-27B-DFlash-GGUF-llama.cpp) | `dflash` | ✅ used here (Q8_0) |
| [williamliao/qwen3.6-27B-DFlash-GGUF](https://huggingface.co/williamliao/qwen3.6-27B-DFlash-GGUF) | `dflash` | ✅ alternative |
| [jojohai/Qwen3.6-27B-DFlash-GGUF](https://huggingface.co/jojohai/Qwen3.6-27B-DFlash-GGUF) | `dflash` | ✅ IQ4_XS only |
| [spiritbuun/Qwen3.6-27B-DFlash-GGUF](https://huggingface.co/spiritbuun/Qwen3.6-27B-DFlash-GGUF) and similar (`Anbeeld`, `Radamanthys11`, …) | `dflash-draft` | ❌ pre-merge, fails to load |
| [z-lab/Qwen3.6-27B-DFlash](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) | safetensors | original source — convert yourself with current `convert_hf_to_gguf.py` |

Check a repo **before** downloading: open
`https://huggingface.co/api/models/<repo>` and look at `gguf.architecture` —
it must say `dflash`. Swap drafters via the `LLAMA_DRAFT_REPO` /
`LLAMA_DRAFT_FILE` env vars in [docker/docker-compose.yaml](docker/docker-compose.yaml).

## Speed leaderboard (`benchmark/leaderboard.py`)

A live tok/s benchmark that hits one running llama.cpp server, sends the same
prompt N times, and keeps a persistent leaderboard in
[`benchmark/leaderboard_runs.csv`](benchmark/leaderboard_runs.csv). Runs are
grouped by **run name** (auto-detected from the server's model alias), and the
board shows the best average for each name.

### Run it

Start one server (baseline, DFlash, or MTP), then:

```bash
# auto-detect the server on ports 8000/8001 and the run name from its alias
uv run python benchmark/leaderboard.py

# or target a specific server / settings
uv run python benchmark/leaderboard.py --port 8001 --runs 10 --max-tokens 1500
```

Common flags:

| Flag | Default | Meaning |
|---|---|---|
| `--port` / `--url` | auto (probes 8000, then 8001) | Which server to benchmark |
| `--run-name` | server's model alias | Leaderboard group name (e.g. `qwen36-dflash15`) |
| `--runs` | `10` | How many requests to send |
| `--max-tokens` | `1500` | Output tokens per request (OpenAI API) |
| `--prompt` | Fibonacci code prompt | The prompt sent every run — **keep it fixed** when comparing servers |
| `--history-file` | `benchmark/leaderboard_runs.csv` | Where results are appended |
| `--show-top` | `20` | Leaderboard rows to print |

Each invocation appends one row and reprints the leaderboard sorted by avg
tok/s.

### What `accept%` means

`accept%` is the **speculative-decoding draft acceptance rate** for the run —
how often the draft model's guessed tokens were accepted by the target model.
It comes from llama.cpp's per-response `timings` block (`draft_n_accepted /
draft_n`), pooled across all N requests in that run (token-weighted, not the
mean of per-request percentages):

```
accept% = 100 * total_draft_tokens_accepted / total_draft_tokens
```

- **Higher is better** — more accepted drafts means more tokens produced per
  target-model step, i.e. more speedup. It is *not* a quality metric: at
  greedy decoding (temperature 0) speculative decoding is output-lossless
  regardless of acceptance, so accept% only affects speed, never correctness.
- **`-`** means the server reported no draft stats — i.e. a **baseline**
  (non-speculative) server, or an old leaderboard row saved before this column
  existed. The baseline runs the target model with **no DFlash draft and no
  MTP** (~71 tok/s raw generation, `llama bench` tg128 on the RTX PRO 6000
  Blackwell); it's the reference point the speculative speedups are measured
  against, so `-` there is expected, not a bug.
- Acceptance is **prompt-dependent**: code prompts accept far more drafts than
  free-form prose. Compare accept% only across runs that used the same prompt.

The raw totals are stored in the `draft_tokens` and `draft_accepted` CSV
columns if you want to aggregate acceptance across multiple runs yourself.



## Intelligence check — aiperf accuracy benchmarks (dflash vs mtp)

Answers *"is speculative decoding hurting the model's intelligence?"* At
temperature 0, DFlash/MTP verify every drafted token against the target model,
so pass@1 should be **identical** across servers — any real gap is a bug in the
spec path, not the model.

Two benchmarks run on the installed aiperf **as-is**, with tiny one-time
downloads:

| `--accuracy-benchmark` | What it is | Download |
|---|---|---|
| `gsm8k` | **GSM8K** — grade-school math word problems: multi-step arithmetic with short numeric answers. The classic easy-reasoning check. Its lighteval loader **caps output at 256 tokens**, which makes it incompatible with reasoning mode (see below). | ~6 MB |
| `math_500` | **MATH-500** — a 500-problem subset of the MATH competition dataset (AMC/AIME level), spread across 7 subjects (Algebra, Counting & Probability, Geometry, Intermediate Algebra, Number Theory, Prealgebra, Precalculus). Harder, and **not** output-capped. | ~0.5 MB |

**Why these two:** both ship with aiperf's accuracy mode out of the box
(lighteval-aligned loaders with deterministic graders), download in seconds, and
bracket difficulty from easy arithmetic up to competition math. Since greedy
speculative decoding is output-lossless, the goal isn't a state-of-the-art score
— it's to confirm DFlash/MTP land on the **same** pass@1 as the plain baseline.

### Why gsm8k needs `math_500` (the reasoning trap)

These servers run with **reasoning on** (the DFlash alias is
`qwen36-dflash15reason`), so the model emits a `<think>…</think>` trace before
its final answer. gsm8k's loader caps generation at **256 tokens** — far less
than a reasoning trace needs — so the response is cut off mid-`<think>` before
any answer appears. The grader then extracts nothing: **every response comes
back unparsed**. Measured on DFlash: 200/200 unparsed → **10.50%** (pure noise),
and aiperf itself prints *"every accuracy response was unparsed."*

`math_500` has no such cap (outputs averaged ~9,500 tokens, some hitting the
32,768-token context limit), so the trace finishes, the boxed answer is emitted,
and it parses cleanly → **72.00%, 0 unparsed**.

**Rule:** with reasoning on, only benchmark tasks that allow long outputs — run
`math_500` and skip gsm8k (gsm8k would need a separate non-reasoning server
config to be meaningful).

### Run it (one server at a time — they share the GPU)

Everything is wrapped in [`benchmark/intelligence_sweep.sh`](benchmark/intelligence_sweep.sh):
it starts the right container, waits for `/health`, auto-detects the model alias
from the live server (aliases change as the compose file is tweaked), runs the
benchmark(s) named in `BENCHES`, and prints the pass@1 rows at the end.

> **First run:** make the script executable — `chmod +x benchmark/intelligence_sweep.sh`
> (a fresh clone doesn't preserve the execute bit).

```bash
# quick smoke test (5 problems, ~15 s) — math_500 only:
N=5 BENCHES=math_500 ./benchmark/intelligence_sweep.sh dflash

# full reasoning run — math_500 ONLY (the part that works with --reasoning on);
# one server at a time, stop each before starting the next:
BENCHES=math_500 ./benchmark/intelligence_sweep.sh dflash
docker compose -f docker/docker-compose.yaml stop llamacpp_dflash

BENCHES=math_500 ./benchmark/intelligence_sweep.sh mtp     # same port as dflash (8001)
docker compose -f docker/docker-compose.yaml stop llama_cpp_qwen36_mtp

BENCHES=math_500 ./benchmark/intelligence_sweep.sh base    # baseline, port 8000

# --- DON'T run gsm8k with reasoning on: the 256-token cap truncates <think>
# --- before any answer -> 100% unparsed, ~10% (noise). The script default is
# --- BENCHES="gsm8k math_500", so ALWAYS pass BENCHES=math_500 explicitly.
# N=5 ./benchmark/intelligence_sweep.sh dflash    # runs gsm8k too — DON'T
# ./benchmark/intelligence_sweep.sh dflash        # runs gsm8k too — DON'T
```

Results land in `artifacts/{dflash,mtp,base}/accuracy/<bench>/`. Two flag gotchas
are baked into the script (learned the hard way on aiperf 0.11.0):

- **`sequential`, not `shuffle`** — accuracy mode rejects shuffle. Sequential
  also guarantees every server sees the exact same first-N problems.
- **`--request-count` must equal `--num-dataset-entries`** — without it aiperf
  loops the dataset and grades every problem twice.

A full 200-problem `math_500` run is ~10–15 min per server (long answers); the
`N=5` smoke test is ~15 s.

### Read the results

Each run writes `artifacts/<tag>/accuracy/<bench>/accuracy_results.csv`, and the
script greps the summary line for you. To pull every pass@1 row yourself:

```bash
grep OVERALL artifacts/*/accuracy/math_500/accuracy_results.csv
```

The `OVERALL` row is `OVERALL,correct,total,unparsed,accuracy` (accuracy is
0–1). **`unparsed` must be 0** — anything higher means the grader couldn't
extract answers (truncated or malformed output), so the accuracy figure is
meaningless, not a real score.

Measured pass@1 so far (temperature 0, 200 problems, `math_500`):

| Server | alias | `math_500` pass@1 | unparsed | notes |
|---|---|---|---|---|
| **DFlash** | `qwen36-dflash15reason` | **72.00%** (144/200) | 0 | clean |
| baseline | `qwen3.6-27B` | 24.50% (49/200) | 0 | ⚠️ **not comparable** — this run hit 132 connection errors and only 68/200 requests completed; rerun before trusting |
| MTP | — | not yet measured | — | run `BENCHES=math_500 … mtp` |
| gsm8k (reasoning on) | any | 10.50% | 200/200 | broken — see the reasoning trap above |

That baseline 24.50% is **not** a real accuracy gap versus DFlash's 72% — it's a
failed run (errored requests are graded as wrong). At temperature 0 speculative
decoding is output-lossless, so a clean baseline should land on the **same** 72%.
Rerun the baseline (restart the container, drop concurrency) before comparing.

## Speed tests — how to speed sweep

[`benchmark/speed_sweep.sh`](benchmark/speed_sweep.sh) is a synthetic throughput
sweep (no dataset downloads): input = output tokens, greedy, concurrency 1,
across sizes **512 / 4096 / 12288 / 36864**. It starts the container, waits for
`/health`, auto-detects the alias, runs `aiperf profile` per size, then reads
draft acceptance from a completion's `timings` block into `acceptance.txt`.

> **First run:** make the script executable — `chmod +x benchmark/speed_sweep.sh`
> (a fresh clone doesn't preserve the execute bit).

```bash
# full sweep (all four sizes), one server at a time:
./benchmark/speed_sweep.sh dflash
docker compose -f docker/docker-compose.yaml stop llamacpp_dflash

./benchmark/speed_sweep.sh mtp     # port 8001
docker compose -f docker/docker-compose.yaml stop llama_cpp_qwen36_mtp

./benchmark/speed_sweep.sh base    # baseline, port 8000

# just one size (e.g. re-run only the high-context point):
./benchmark/speed_sweep.sh dflash 36864
# or a couple of mid points:
./benchmark/speed_sweep.sh dflash 4096 12288
```

Results land in `artifacts/{dflash,mtp,base}/speed/isl<N>_osl<N>/`, plus a
per-size `acceptance.txt`. Passing a size that isn't one of the four fails fast
with a clear message.


## References

- DFlash project page — https://z-lab.ai/pHardware: single RTX PRO 6000 Blackwell, Qwen3.6-27B UD-Q4_K_XL target + DFlash Q8_0 draft,
rojects/dflash/
- DFlash paper (arXiv) — https://arxiv.org/pdf/2602.06036
- llama.cpp DFlash merge (PR #22105) — https://github.com/ggml-org/llama.cpp/pull/22105#event-27298914025
- z-lab DFlash model collection — https://huggingface.co/collections/z-lab/dflash
- DFlash GGUFs on Hugging Face — https://huggingface.co/models?search=dflash
- r/LocalLLaMA discussion — https://www.reddit.com/r/LocalLLaMA/comments/1uhx862/dflash_support_merged_into_llamacpp/