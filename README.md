# DFlash

Qwen3.6-27B with **DFlash speculative decoding** on llama.cpp, benchmarked for
both **speed** (tok/s across context sizes) and **intelligence** (pass@1 on math
benchmarks) against a plain baseline and an MTP setup. Because greedy speculative
decoding is output-lossless, the questions these benchmarks answer are: *how much
faster is DFlash, and does it cost any accuracy?* Full setup, the "what broke and
how I fixed it" log, and the config knobs live in [`DFLASH.md`](DFLASH.md).

## YouTube

LOCAL AI / SPECULATIVE DECODING SERIES:

- Up to 6x Faster AI? DFlash Explained, Deployed & Benchmarked on Qwen 3.6 27B. Lamma.cpp! https://www.youtube.com/watch?v=TUdihA_dJjo 

## Hardware

All numbers here are measured on a single card:

| | |
|---|---|
| **GPU** | NVIDIA RTX PRO 6000 Blackwell Workstation Edition (Blackwell, compute capability 12.0) |
| **VRAM** | 97 GB - target + draft both fit on one card |
| **Host** | Zen4 CPU |
| **llama.cpp** | `ghcr.io/ggml-org/llama.cpp:server-cuda13` (CUDA 13) |
| **GPU layout** | single GPU, `-ngl -1`, no tensor split |
| **Raw baseline** (no speculation) | **71 tok/s** generation (`llama bench` tg128) · **3877 tok/s** prompt processing (pp512) |

## Results summary

Consolidated numbers from the current runs. Machine-readable copy:
[`results_summary.csv`](results_summary.csv) (labeled tables in one file); the
charts below are generated from it by
[`benchmark/plot_results.py`](benchmark/plot_results.py) - regenerate with
`uv run python benchmark/plot_results.py`. Analysis and best findings:
[`Insights.md`](Insights.md).

### Intelligence - MATH-500 pass@1, base vs DFlash (N=100)

![Same answers, 3.75× faster](assets/accuracy_vs_speed.png)

Both servers answered the **same first 100 MATH-500 problems** (sequential
sampling, seed 42), greedy (temperature 0), reasoning off, `LLAMA_CTX=32768`,
one server on the GPU at a time. Zero request errors and zero unparsed answers
on both sides - the first fully valid accuracy pair in this repo.

| | base (`qwen3.6-27B`) | DFlash (`qwen36-dflash10`) |
|---|---:|---:|
| pass@1 | **87/100 (87.0%)** | **86/100 (86.0%)** |
| gen tok/s during the run | 72.06 | **270.47 (3.75×)** |
| wall-clock for 100 problems | 24.6 min | **7.6 min** |
| avg output tokens | 1,046 | 1,095 |

![Accuracy by subject](assets/accuracy_by_subject.png)

Per-subject scores are **identical in 6 of 7 subjects** - the entire 1-point
gap is a single Prealgebra problem. That is the expected result: greedy
speculative decoding verifies every drafted token against the target model, so
it is output-lossless; a 1/100 difference is within numerical-batching noise.
**DFlash costs no measurable accuracy and generated 3.75× faster while being
measured.** Commands and validity rules: [Intelligence check](#intelligence-check--aiperf-accuracy-benchmark-math-500).

### Speed sweep - DFlash vs baseline

![DFlash pulls away as context grows](assets/speed_vs_context.png)

aiperf synthetic sweep (ISL = OSL, greedy, concurrency 1, 3-30 requests per
size - fewer as size grows); artifacts in `artifacts/{base,dflash}/speed/` and
`reasoning/dflash/speed/` (the latter = DFlash with reasoning on).
[How the synthetic benchmark works](#how-the-synthetic-benchmark-works-islosl-greedy).

| Context (ISL=OSL) | base tok/s | DFlash tok/s | DFlash reasoning tok/s | **Speedup** | base ITL (ms) | DFlash ITL (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 67.62 | 97.13 | 96.61 | **1.44×** | 14.20 | 9.62 |
| 4 096 | 67.53 | 182.04 | 166.64 | **2.70×** | 14.46 | 5.08 |
| 12 288 | 64.78 | 220.12 | 240.41 | **3.40×** | 15.11 | 4.17 |
| 36 864 | 61.47 | **273.04** | - | **4.44×** | 15.91 | 3.26 |
| 98 304 | - | - | 241.44 | - | - | - |

The speedup **grows with context**: the baseline degrades from 67.6 → 61.5 tok/s
while DFlash climbs from 97 → 273 tok/s.

### Leaderboard runs (`benchmark/leaderboard_runs.csv`)

Fixed Fibonacci prompt, 10 requests per run, `n_max` = `--spec-draft-n-max`;
speedup vs the baseline row (70.34 tok/s). Separate measurement path from the
sweep above - see [Speed leaderboard](#speed-leaderboard-benchmarkleaderboardpy).

| Run | Type | n_max | avg tok/s | median | accept % | Speedup |
|---|---|---:|---:|---:|---:|---:|
| qwen36-dflash12 | dflash | 12 | 256.01 | 258.54 | 50.5 | 3.64x |
| qwen36-dflash15 | dflash | 15 | 253.21 | 256.35 | 42.8 | 3.60x |
| qwen36-dflash8 | dflash | 8 | 236.15 | 239.06 | 63.1 | 3.36x |
| qwen36-dflash15reason | dflash | 15 | 224.18 | 225.30 | 36.4 | 3.19x |
| qwen3.6-27b-mtp8 | mtp | 8 | 190.17 | 191.48 | 57.9 | 2.70x |
| qwen3.6-27b-mtp12 | mtp | 12 | 179.03 | 180.29 | 43.3 | 2.55x |
| qwen36-dflash4 | dflash | 4 | 178.34 | 179.65 | 80.4 | 2.54x |
| qwen3.6-27b-mtp4 | mtp | 4 | 167.47 | 167.70 | 79.8 | 2.38x |
| qwen3.6-27b-mtp15 | mtp | 15 | 167.00 | 168.25 | 36.0 | 2.37x |
| qwen3.6-27b-mtp15reason | mtp | 15 | 163.12 | 163.60 | 34.4 | 2.32x |
| qwen3.6-27b-mtp2 | mtp | 2 | 142.68 | 142.48 | 88.5 | 2.03x |
| qwen36-dflash2 | dflash | 2 | 141.01 | 141.12 | 91.1 | 2.00x |
| qwen3.6-27B | baseline | - | 70.34 | 70.38 | - | 1.00x |

Best config: **dflash12** - throughput peaks at n_max=12 and dips slightly at
15, and DFlash beats MTP at every draft length.

## Choosing a DFlash draft GGUF

This setup pairs the [unsloth Q4_K_XL target](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF)
with a **Q8_0 draft from a different repo** - that's fine, and here's what actually
matters when picking a drafter:

- **Quantization does NOT need to match.** The draft only *proposes* tokens; the
  target *verifies* every one. At temperature 0 the output is lossless - identical
  to what the target would produce alone - so a bad draft can only cost speed,
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
| [z-lab/Qwen3.6-27B-DFlash](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) | safetensors | original source - convert yourself with current `convert_hf_to_gguf.py` |

Check a repo **before** downloading: open
`https://huggingface.co/api/models/<repo>` and look at `gguf.architecture` -
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
| `--prompt` | Fibonacci code prompt | The prompt sent every run - **keep it fixed** when comparing servers |
| `--history-file` | `benchmark/leaderboard_runs.csv` | Where results are appended |
| `--show-top` | `20` | Leaderboard rows to print |

Each invocation appends one row and reprints the leaderboard sorted by avg
tok/s.

### What `accept%` means

`accept%` is the **speculative-decoding draft acceptance rate** for the run -
how often the draft model's guessed tokens were accepted by the target model.
It comes from llama.cpp's per-response `timings` block (`draft_n_accepted /
draft_n`), pooled across all N requests in that run (token-weighted, not the
mean of per-request percentages):

```
accept% = 100 * total_draft_tokens_accepted / total_draft_tokens
```

- **Higher is better** - more accepted drafts means more tokens produced per
  target-model step, i.e. more speedup. It is *not* a quality metric: at
  greedy decoding (temperature 0) speculative decoding is output-lossless
  regardless of acceptance, so accept% only affects speed, never correctness.
- **`-`** means the server reported no draft stats - i.e. a **baseline**
  (non-speculative) server, or an old leaderboard row saved before this column
  existed. The baseline runs the target model with **no DFlash draft and no
  MTP** (~71 tok/s raw generation, `llama bench` tg128 on the RTX PRO 6000
  Blackwell); it's the reference point the speculative speedups are measured
  against, so `-` there is expected, not a bug.
- Acceptance is **prompt-dependent**: code prompts accept far more drafts than
  free-form prose. Compare accept% only across runs that used the same prompt.

The raw totals are stored in the `draft_tokens` and `draft_accepted` CSV
columns if you want to aggregate acceptance across multiple runs yourself.



## Intelligence check - aiperf accuracy benchmark (MATH-500)

Answers *"is speculative decoding hurting the model's intelligence?"* At
temperature 0, DFlash/MTP verify every drafted token against the target model,
so pass@1 should be **identical** across servers - any real gap is a bug in the
spec path, not the model.

### Why only MATH-500

The accuracy comparison uses **MATH-500 only** (`--accuracy-benchmark
math_500`): a 500-problem subset of the MATH competition dataset (AMC/AIME
level) across 7 subjects, a ~0.5 MB one-time download, deterministic
boxed-answer grading, and **no output-length cap**. The other aiperf accuracy
benchmarks were dropped deliberately:

- **`gsm8k`** - aiperf's lighteval loader hard-caps generation at **256
  tokens**. The model's step-by-step answers routinely run past that, so
  responses are truncated before the final answer and the grader extracts
  nothing. Measured here: **100% unparsed → a ~10% "score" that is pure
  noise.** A harness limitation, not a model result.
- **`lcb-codegeneration` (LiveCodeBench)** - grading requires sandboxed
  execution of generated code against hidden test suites, and needs a large N
  to mean anything. A 5-problem smoke test scored 0/5 with multi-thousand-token
  solutions - no signal either way - and a meaningful N costs hours per server.
  Out of scope for this comparison.

MATH-500 alone answers the question being asked. Greedy speculative decoding
either reproduces the target model's outputs or it doesn't - any accuracy
effect would be task-independent, so one uncapped, deterministically-graded
benchmark is enough. Measured at N=100, base and DFlash differ by exactly one
problem (87 vs 86).

### Run it (the exact commands behind the results above)

Everything is wrapped in [`benchmark/intelligence_sweep.sh`](benchmark/intelligence_sweep.sh):
it starts the right container, waits for `/health`, auto-detects the model alias
from the live server (aliases change as the compose file is tweaked), runs the
benchmark(s) named in `BENCHES`, and prints the pass@1 rows at the end.

> **First run:** make the script executable - `chmod +x benchmark/intelligence_sweep.sh`
> (a fresh clone doesn't preserve the execute bit).

```bash
cd benchmark

# 0. one server on the GPU at a time - stop everything first
docker compose -f ../docker/docker-compose.yaml stop

# 1. smoke test the baseline (5 problems, ~1 min)
LLAMA_CTX=32768 N=5 BENCHES=math_500 ./intelligence_sweep.sh base

# 2. full baseline run (100 problems, ~25 min)
LLAMA_CTX=32768 N=100 BENCHES=math_500 ./intelligence_sweep.sh base
docker compose -f ../docker/docker-compose.yaml stop llamacpp_baseline

# 3. full DFlash run (same 100 problems, ~8 min)
LLAMA_CTX=32768 N=100 BENCHES=math_500 ./intelligence_sweep.sh dflash
docker compose -f ../docker/docker-compose.yaml stop llamacpp_dflash

# 4. compare
grep -H OVERALL ../artifacts/{base,dflash}/accuracy/math_500/accuracy_results.csv
```

Setup rules that made this work (the first baseline attempt died with 132
connection errors before these):

- **`LLAMA_CTX=32768`, not the 262k default.** The full-size f16 KV cache is
  the prime suspect for the earlier server instability, and 32k is plenty for
  reasoning-off answers (avg ~1,050 tokens).
- **One container on the GPU at a time.** All services use
  `restart: unless-stopped`, so a leftover container can come back and fight
  for VRAM mid-run - `docker compose stop` everything first.
- **Reasoning off** (the compose default). With reasoning on, answers average
  ~9,500 tokens and the 71 tok/s baseline would need ~7 hours for a 200-problem
  run; reasoning-off finishes N=100 in 24.6 min (base) / 7.6 min (DFlash).
- Two aiperf flag gotchas baked into the script (aiperf 0.11.0):
  `--dataset-sampling-strategy sequential` (accuracy mode rejects shuffle, and
  sequential + a fixed seed guarantees every server sees the exact same
  problems) and `--request-count` equal to `--num-dataset-entries` (otherwise
  aiperf loops the dataset and grades every problem twice).

### Read the results

Each run writes `artifacts/<tag>/accuracy/math_500/accuracy_results.csv`, and
the script greps the summary line for you. The `OVERALL` row is
`OVERALL,correct,total,unparsed,accuracy` (accuracy is 0-1).

Measured pass@1 (temperature 0, same first 100 MATH-500 problems, reasoning off):

| Server | alias | pass@1 | unparsed | gen tok/s |
|---|---|---|---:|---:|
| baseline | `qwen3.6-27B` | **87.00%** (87/100) | 0 | 72.06 |
| DFlash | `qwen36-dflash10` | **86.00%** (86/100) | 0 | **270.47** |

## Speed tests - how to speed sweep

[`benchmark/speed_sweep.sh`](benchmark/speed_sweep.sh) is a synthetic throughput
sweep (no dataset downloads): input = output tokens, greedy, concurrency 1,
across sizes **512 / 4096 / 12288 / 36864**. It starts the container, waits for
`/health`, auto-detects the alias, runs `aiperf profile` per size, then reads
draft acceptance from a completion's `timings` block into `acceptance.txt`.

> **First run:** make the script executable - `chmod +x benchmark/speed_sweep.sh`
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

### How the synthetic benchmark works (ISL/OSL, greedy)

**ISL** = input sequence length (prompt tokens), **OSL** = output sequence
length (generated tokens). The sweep sets them equal at each point
(`isl512_osl512` … `isl36864_osl36864`), so "context size" is a single knob:
at the 36K point the model reads a 36,864-token prompt, then generates 36,864
more tokens on top of it.

**Where the prompts come from.** No dataset downloads - aiperf synthesizes
prompts locally from a text corpus it ships (Shakespeare), tokenized with the
model's own tokenizer (`--tokenizer Qwen/Qwen3.6-27B`, so token counts are
exact). Each request gets a random corpus window of **exactly N tokens**
(`--synthetic-input-tokens-stddev 0` → no length variation; `--random-seed 42`
→ reproducible, and base/dflash/mtp all see the same inputs). The content is
nonsense mid-sentence Shakespeare on purpose: only the token count matters.
Every run saves its generated prompts in `inputs.json` for inspection.

**How the output length is forced.** A model would normally answer a
Shakespeare fragment in a few hundred tokens and stop. Two flags prevent that:

- `--extra-inputs ignore_eos:true` - the server keeps generating even when the
  model emits its end-of-sequence token
- `--extra-inputs min_tokens:$N` with `--output-tokens-mean $N` - pins
  generation to exactly N tokens

So every request produces exactly N output tokens regardless of content. This
is the opposite of the accuracy runs (where hitting EOS naturally is *desired*
- that's the benign "OSL mismatch" warning there, which would be a real
problem here).

**Why greedy.** `temperature:0, top_p:1.0, top_k:1` - always pick the single
most likely next token. Two reasons:

1. **Determinism** - same prompt, same output, run to run; the numbers are
   reproducible.
2. **It's the lossless regime for speculative decoding** - at temperature 0
   the target verifies each drafted token against its own argmax, so DFlash's
   output is the target model's output. The speed comparison is
   apples-to-apples: the same tokens get produced, only the machinery differs.
   Sampling would also change draft acceptance rates and muddy the comparison.

**The measurement discipline.**

- `--concurrency 1` - one request in flight at a time: this measures
  **single-user** generation speed (the ITL / tok/s per user in the speed
  table), with no batching effects.
- Warmup requests (`WARM`: 1-2 per size) are sent first and discarded - first
  requests pay one-time costs (CUDA graph capture, cache allocation) that
  would pollute the numbers.
- Measured request counts shrink with size (`REQS`: 30 → 10 → 5 → 3) because a
  single 36K request takes ~20 minutes on the baseline.

**Why the sweep tells the DFlash story.** Every generated token attends over
an ever-longer KV cache, so per-token cost grows with context. The baseline
pays that cost once per token (67.6 → 61.5 tok/s as context grows). DFlash
verifies a batch of ~10 drafted tokens in one target pass, amortizing that
expensive long-context attention across multiple tokens - which is exactly why
its curve climbs (97 → 273 tok/s) and the speedup grows from 1.44× to 4.44×.

## References

- DFlash project page - https://z-lab.ai/projects/dflash/: single RTX PRO 6000 Blackwell, Qwen3.6-27B UD-Q4_K_XL target + DFlash Q8_0 draft,
rojects/dflash/
- DFlash paper (arXiv) - https://arxiv.org/pdf/2602.06036
- llama.cpp DFlash merge (PR #22105) - https://github.com/ggml-org/llama.cpp/pull/22105#event-27298914025
- z-lab DFlash model collection - https://huggingface.co/collections/z-lab/dflash
- DFlash GGUFs on Hugging Face - https://huggingface.co/models?search=dflash
- r/LocalLLaMA discussion - https://www.reddit.com/r/LocalLLaMA/comments/1uhx862/dflash_support_merged_into_llamacpp/
