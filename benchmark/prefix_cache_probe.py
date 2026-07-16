#!/usr/bin/env python3
"""Multi-turn prefix cache probe for llama.cpp servers (base / DFlash / MTP).

Answers one question the speed sweep and leaderboard cannot: when a chat
conversation grows turn by turn, does the server reuse the KV cache for the
shared prefix, or does it reprocess the whole history on every turn?

Method: send a conversation with a long opening message, then keep appending
short follow-up turns. For each response, read llama.cpp's `timings` block:

  - usage.prompt_tokens = full prompt length this request
  - timings.prompt_n    = prompt tokens the server actually processed
  - timings.prompt_ms   = time spent on prompt processing (server-side TTFT)

A working prefix cache shows prompt_n << prompt_tokens on turns 2+ (only the
new suffix is processed). A cache miss shows prompt_n ~= prompt_tokens, i.e.
full reprocessing, and prompt_ms grows with history length.

Usage (start one server first, same as leaderboard.py):

    uv run python benchmark/prefix_cache_probe.py
    uv run python benchmark/prefix_cache_probe.py --port 8001 --turns 5
    uv run python benchmark/prefix_cache_probe.py --no-cache   # force-miss control run
"""

from __future__ import annotations

import argparse
import json
import time

import requests

CANDIDATE_PORTS = (8000, 8001)

# One paragraph of filler, repeated to build a long opening message. Content
# is irrelevant, only the token count matters (same idea as the aiperf sweep).
FILLER_PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog while the miller grinds "
    "wheat by the river and the ferryman waits for passengers under the old "
    "stone bridge near the market square where traders argue about the price "
    "of salt, wool, timber, iron, and grain from the northern valleys. "
)


def detect_server(host: str, port: int | None, url: str | None) -> str:
    if url:
        return url.rstrip("/")
    if port is not None:
        return f"http://{host}:{port}"
    for candidate in CANDIDATE_PORTS:
        base = f"http://{host}:{candidate}"
        try:
            if requests.get(f"{base}/health", timeout=3).status_code < 500:
                return base
        except Exception:
            continue
    raise SystemExit(f"No server found on {host} ports {CANDIDATE_PORTS}.")


def detect_model(base_url: str) -> str:
    data = requests.get(f"{base_url}/v1/models", timeout=10).json()
    return data["data"][0]["id"]


def build_opener(target_tokens: int) -> str:
    # ~0.75 words per token is close enough; the exact count does not matter,
    # only that turns 2+ carry a long shared prefix.
    words_needed = int(target_tokens * 0.75)
    para_words = len(FILLER_PARAGRAPH.split())
    body = FILLER_PARAGRAPH * (words_needed // para_words + 1)
    return (
        "Here is a document to keep in context for the rest of this chat:\n\n"
        + body
        + "\n\nAcknowledge in one short sentence."
    )


def chat_turn(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    cache_prompt: bool,
) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        # llama.cpp extension: per-request prompt cache switch (default true)
        "cache_prompt": cache_prompt,
    }
    start = time.perf_counter()
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=900,
    )
    response.raise_for_status()
    wall_s = time.perf_counter() - start
    body = response.json()
    return {
        "content": body["choices"][0]["message"]["content"] or "",
        "prompt_tokens": body.get("usage", {}).get("prompt_tokens"),
        "timings": body.get("timings") or {},
        "wall_s": wall_s,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--url", default=None)
    parser.add_argument("--turns", type=int, default=4, help="Total chat turns to send.")
    parser.add_argument(
        "--opener-tokens",
        type=int,
        default=8000,
        help="Approximate token length of the first user message.",
    )
    parser.add_argument("--max-tokens", type=int, default=200, help="Answer budget per turn.")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Send cache_prompt:false on every request (control run showing a full miss).",
    )
    args = parser.parse_args()

    base_url = detect_server(args.host, args.port, args.url)
    model = detect_model(base_url)
    cache_prompt = not args.no_cache
    print(f"server={base_url} model={model} cache_prompt={cache_prompt}")

    messages: list[dict[str, str]] = [
        {"role": "user", "content": build_opener(args.opener_tokens)}
    ]
    follow_ups = [
        "Summarize the document in one sentence.",
        "What animals are mentioned? Answer briefly.",
        "What goods are traded? Answer briefly.",
        "Where does the ferryman wait? Answer briefly.",
        "Name one structure mentioned. Answer briefly.",
    ]

    header = f"{'turn':>4} {'prompt_tok':>10} {'processed':>10} {'cached':>8} {'prompt_ms':>10} {'wall_s':>7}  verdict"
    print(header)
    print("-" * len(header))

    verdicts: list[str] = []
    for turn in range(1, args.turns + 1):
        result = chat_turn(base_url, model, messages, args.max_tokens, cache_prompt)
        timings = result["timings"]
        prompt_tokens = result["prompt_tokens"]
        prompt_n = timings.get("prompt_n")
        prompt_ms = timings.get("prompt_ms")

        if prompt_tokens is None or prompt_n is None:
            print(f"{turn:>4}  server returned no usage/timings block, cannot judge caching")
            verdict = "unknown"
        else:
            cached = prompt_tokens - prompt_n
            if turn == 1:
                verdict = "baseline (cold prefill)"
            elif prompt_n <= 0.2 * prompt_tokens:
                verdict = "HIT (prefix reused)"
            else:
                verdict = "MISS (history reprocessed)"
            ms = f"{prompt_ms:.0f}" if prompt_ms is not None else "-"
            print(
                f"{turn:>4} {prompt_tokens:>10} {prompt_n:>10} {cached:>8} {ms:>10} "
                f"{result['wall_s']:>7.2f}  {verdict}"
            )
        verdicts.append(verdict)

        messages.append({"role": "assistant", "content": result["content"]})
        if turn <= len(follow_ups):
            messages.append({"role": "user", "content": follow_ups[turn - 1]})
        else:
            messages.append({"role": "user", "content": f"Reply with the number {turn}."})

    later_turns = verdicts[1:]
    if later_turns and all(v.startswith("HIT") for v in later_turns):
        print("\nRESULT: prefix cache WORKS on this server - turns 2+ only processed the new suffix.")
    elif any(v.startswith("MISS") for v in later_turns):
        print("\nRESULT: prefix cache MISSED - this server reprocessed conversation history.")
    else:
        print("\nRESULT: inconclusive (no timings data).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
