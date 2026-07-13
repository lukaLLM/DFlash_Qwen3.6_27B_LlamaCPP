# bench_ngram - iterative-coding speed bench

Headless benchmark ([bench_ngram.py](bench_ngram.py), no UI) that measures
whether the n-gram drafters actually help when an LLM works on a code base.

The workload: 18 fixed prompts sent as ONE cumulative conversation, in two
phases:

- **build (1-9)**: the model creates a Gradio chat app and extends the SAME
  code base feature by feature (history, settings panel, stats, styling...).
- **maint (10-18)**: realistic vibe-coding follow-ups on the finished app,
  each forcing a different overlap profile:
  - *full re-emit* - "show me the complete final file" (max verbatim overlap)
  - *docstrings+types* / *renames* - mechanical whole-file rewrites with tiny
    deltas (this is where n-gram accept % should peak)
  - *bugfix report* - vague user bug report, find-and-fix, full file again
  - *client refactor* - restructure into a LlamaClient class (moved code =
    still lots of verbatim spans)
  - *pytest tests* - new code that quotes existing names/signatures
  - *export feature* - small feature added to the existing file
  - *review+fix* - review then re-emit fixed file
  - *write readme* - prose control turn (little code overlap)

The model keeps re-emitting and modifying its own prior code every turn - the
exact repeated-text scenario the n-gram lookup drafters target, and the maint
phase mirrors how real coding sessions actually look. Measured per turn from
the llama.cpp `timings` block: TTFT, decode t/s, prompt t/s, generated tokens,
draft tokens, draft accepted, accept %. Aggregates are reported overall AND
per phase (stats.json `aggregates_by_phase`; the comparison table has
`maint t/s` / `m.accept` columns) - expect the ngram advantage to show up
mostly in the maint columns.

## Setup + run (uv)

No setup step: the script carries PEP 723 inline metadata
(`dependencies = ["openai>=1.90,<3"]`), so uv creates an isolated env for it
on first run - the repo venv is not touched.

You start/swap the docker services yourself - one at a time (they share the
GPU and host port 8001):

```bash
# from the repo root
docker compose -f docker/docker-compose.yaml up -d llamacpp_baseline      # no spec decoding
uv run benchmark/bench_ngram.py --baseline    # targets port 8000, reference decode t/s

docker compose -f docker/docker-compose.yaml up -d llamacpp_dflash        # DFlash only
uv run benchmark/bench_ngram.py

docker compose -f docker/docker-compose.yaml up -d llamacpp_dflash_ngram  # DFlash + ngram
uv run benchmark/bench_ngram.py
```

The baseline run gives the no-speculation reference speed; its accept columns
show `-` (the server reports no draft stats). All three land in the same
comparison table.

At the end of every run the script prints the per-turn table, the aggregates,
and a comparison table across ALL saved runs - so after the second run you
immediately see dflash vs dflash+ngram side by side.

Options - by default NO sampler params are sent, so the run uses the
server/model defaults (we bench the server exactly as configured in docker).
The flags exist as optional overrides only:

```
--url URL             server base url        (default http://localhost:8001, env LLAMA_BENCH_URL)
--baseline            bench the baseline server instead (http://localhost:8000,
                      or $LLAMA_BASELINE_HOST_PORT); --url takes precedence
--tag TAG             folder-name suffix - use for LLAMA_SPEC_TYPE ablations,
                      which all share the qwen36-dflash-ngram alias
--temperature / --top-p / --top-k / --min-p / --repeat-penalty / --max-tokens
                      optional overrides, only sent when set (the non-OpenAI
                      ones go through extra_body); unset = server default
--system TEXT         optional system prompt
```

stats.json records which overrides (if any) were sent under
`settings.overrides`, so runs stay comparable.

Ablation example (matching the docker-compose comment):

```bash
LLAMA_SPEC_TYPE=draft-dflash docker compose -f docker/docker-compose.yaml up -d llamacpp_dflash_ngram
uv run benchmark/bench_ngram.py --tag control
LLAMA_SPEC_TYPE=draft-dflash,ngram-mod,ngram-map-k4v docker compose -f docker/docker-compose.yaml up -d llamacpp_dflash_ngram
uv run benchmark/bench_ngram.py --tag full-stack
```

## Output

Each run gets its own folder:

```
benchmark/bench_results/<timestamp>_<model-alias>[_<tag>]/
  stats.json        per-turn rows + aggregates + settings + server info
  transcript.json   the full cumulative conversation
  responses/01.md   the model's evolving app code, one file per turn
  ...
  responses/09.md
```

Ctrl+C mid-run saves what completed with `"aborted": true` in stats.json.
A failed turn (server died) also aborts and is recorded in its row's `error`.

## Reading the numbers

- **decode t/s** = `timings.predicted_per_second` - headline generation speed.
  The overall figure is `Σ predicted_n / Σ predicted_ms` across turns, NOT a
  mean of per-turn rates, so short turns do not skew it.
- **speedup** (comparison table) = run decode t/s / reference decode t/s. The
  reference is the newest run without draft stats (= your latest `--baseline`
  run), or pass `--ref RUN_ID` explicitly. The reference row reads 1.00x.
- **e2e t/s / end-to-end** = `Σ predicted_n / Σ wall-clock` per request -
  includes prefill wait and network, i.e. the speed you actually feel across
  the whole session (always below decode t/s).
- **speculation** = `spec_share_pct` (share of output tokens that came from
  accepted drafts - the "free" tokens), `draft_rejected` (drafted tokens the
  target threw away), and `draft_overhead` (drafted tokens per output token -
  how aggressively the stack drafts). Compare dflash vs dflash+ngram here: the
  ngram stack should raise the share on maint turns without exploding the
  overhead.
- **ctx scaling** = decode t/s over the first 3 vs last 3 turns - shows how
  each method degrades as the conversation (KV cache) grows.
- **prompt t/s** = prefill speed. `cache_prompt: true` is set, so later turns
  only process the new tokens (`prompt_n` stays small); conversation size is
  tracked via `usage.prompt_tokens` instead ("ctx used" column).
- **draft accept** = `100 * draft_n_accepted / draft_n` - same formula as
  `leaderboard.py`. `-` means the server reported no draft stats (baseline,
  no speculative decoding).
- **TTFT** = client-measured time from request start to first streamed token.

## Why not aiperf?

aiperf cannot see llama.cpp's `draft_n` / `draft_n_accepted`, and the draft
acceptance delta is the whole point of the dflash vs dflash+ngram comparison.
The native `timings` block already gives decode/prefill throughput and token
counts, and TTFT is measured client-side. The aiperf harness in this folder
remains the standardized tool for load/latency sweeps.

## Why not llama.cpp's official speed-bench?

llama.cpp ships its own speculative-decoding benchmark,
[tools/server/bench/speed-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/bench/speed-bench/README.md),
which does report accept rate and baseline-vs-spec speedups on the NVIDIA
SPEED-Bench dataset. We deliberately skip it: full runs are long/expensive,
and its prompts are single-turn dataset questions - no multi-turn support -
so they never exercise the real "model edits its own code" workload that the
n-gram drafters target. This bench measures exactly that scenario instead.
