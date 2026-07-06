#!/usr/bin/env python3
"""Single-endpoint benchmark with persistent CSV leaderboard history."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import requests
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

pyfiglet: Any | None
try:
    import pyfiglet as pyfiglet
except ImportError:
    pyfiglet = None


console = Console()

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
# baseline listens on 8000; dflash and mtp are mutually-exclusive alternates
# on 8001 (see docker/docker-compose.yaml). Probed in this order when no
# --port/--url is given.
CANDIDATE_PORTS = (8000, 8001)
DEFAULT_HISTORY = Path(__file__).with_name("leaderboard_runs.csv")
APIMode = Literal["auto", "completion", "openai"]
YOUTUBE_CHANNEL = "@LukaszGawendaAI"
YOUTUBE_URL = "www.youtube.com/@LukaszGawendaAI"
DEFAULT_CONTEXT_SIZE = 3000
DEFAULT_CONTEXT_RESERVE = 128
DEFAULT_GENERATION_TOKENS = 1500
CSV_FIELDS = [
    "run_name",
    "api",
    "model",
    "runs",
    "avg_tps",
    "median_tps",
    "min_tps",
    "max_tps",
    "total_tokens",
    "total_elapsed_s",
    "samples",
    "sample_tokens",
    "sample_elapsed_s",
    "prompt_preview",
    "draft_tokens",
    "draft_accepted",
]


@dataclass
class RunMetrics:
    run_name: str
    api: str
    model: str | None
    samples: list[float] = field(default_factory=list)
    sample_tokens: list[int] = field(default_factory=list)
    sample_elapsed_s: list[float] = field(default_factory=list)
    avg_tps: float = 0.0
    median_tps: float = 0.0
    min_tps: float = 0.0
    max_tps: float = 0.0
    runs: int = 0
    total_tokens: int = 0
    total_elapsed_s: float = 0.0
    prompt_preview: str = ""
    # Speculative decoding totals from llama.cpp's per-response `timings`
    # (draft_n / draft_n_accepted). Both stay 0 on non-spec (baseline) servers.
    draft_tokens: int = 0
    draft_accepted: int = 0


def accept_pct(row: RunMetrics) -> str:
    if row.draft_tokens <= 0:
        return "-"
    return f"{100 * row.draft_accepted / row.draft_tokens:.0f}%"


def parse_stream_line(line: bytes) -> dict[str, Any] | None:
    text = line.decode("utf-8", errors="ignore").strip()
    if not text:
        return None
    if text.startswith("data:"):
        text = text[5:].strip()
    if text == "[DONE]":
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return cast(dict[str, Any], parsed)
    return None


def is_premature_stream_end(exc: Exception) -> bool:
    text = str(exc).lower()
    return "response ended prematurely" in text or "incomplete read" in text


def clip_text(text: str, max_len: int) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3] + "..."


def big_number(value: float, color: str = "green") -> Text:
    number = f"{value:.1f}"
    if pyfiglet:
        rendered = pyfiglet.figlet_format(number, font="small")
    else:
        rendered = f"\n{number}\n"
    return Text(rendered, style=f"bold {color}")


def wrap_text(text: str, width: int) -> list[str]:
    if not text:
        return ["Waiting for benchmark output..."]

    lines: list[str] = []
    for raw_line in text.splitlines() or [text]:
        line = raw_line.rstrip()
        if not line:
            lines.append("")
            continue
        while len(line) > width:
            lines.append(line[:width])
            line = line[width:]
        lines.append(line)
    return lines


def trim_output(text: str, max_lines: int, max_width: int) -> str:
    lines = wrap_text(text, max_width)
    return "\n".join(lines[-max_lines:])


def estimate_token_count(text: str) -> int:
    """Estimate generated tokens when OpenAI streaming chunks do not expose counts."""
    if not text:
        return 0

    word_like = len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))
    char_like = max(1, round(len(text.encode("utf-8")) / 4))
    return max(word_like, char_like)


def auto_generation_tokens(prompt: str, context_size: int, reserve_tokens: int) -> int:
    prompt_tokens = estimate_token_count(prompt)
    return max(1, context_size - prompt_tokens - reserve_tokens)


def resolve_base_url(url: str | None, host: str, port: int) -> str:
    if url:
        return url
    return f"http://{host}:{port}"


def check_server(url: str, timeout: float) -> bool:
    probe_paths = ("/health", "/v1/models", "/")
    for path in probe_paths:
        try:
            response = requests.get(f"{url.rstrip('/')}{path}", timeout=timeout)
            return response.status_code < 500
        except Exception:
            continue
    return False


def detect_api(base_url: str, api_mode: APIMode, timeout: float) -> str:
    if api_mode != "auto":
        return api_mode
    try:
        response = requests.get(f"{base_url.rstrip('/')}/v1/models", timeout=timeout)
        if response.status_code == 200:
            return "openai"
    except Exception:
        pass
    return "completion"


def detect_model(base_url: str, model: str | None, timeout: float) -> str:
    if model:
        return model
    response = requests.get(f"{base_url.rstrip('/')}/v1/models", timeout=timeout)
    response.raise_for_status()
    data = response.json()
    entries = data.get("data", [])
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("No models returned from /v1/models.")
    model_id = entries[0].get("id")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError("Could not detect model id from /v1/models.")
    return model_id


def stream_completion(
    base_url: str,
    prompt: str,
    n_predict: int,
    seed: int,
    timeout: float,
    on_progress: Callable[[str, int, float], None] | None = None,
) -> tuple[float, str, int, float, int, int]:
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "top_k": 1,
        "seed": seed,
        "ignore_eos": True,
        "cache_prompt": False,
        "stream": True,
    }

    start = time.perf_counter()
    output = ""
    token_count = 0
    previous_server_tokens = 0
    final_tps = 0.0
    draft_tokens = 0
    draft_accepted = 0

    response = requests.post(
        f"{base_url.rstrip('/')}/completion",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        stream=True,
        timeout=timeout,
    )
    response.raise_for_status()

    try:
        for raw_line in response.iter_lines():
            obj = parse_stream_line(raw_line)
            if not obj:
                continue
            content = obj.get("content", "")
            timings = obj.get("timings") or {}
            if timings.get("predicted_per_second"):
                final_tps = float(timings["predicted_per_second"])
            if timings.get("draft_n"):
                draft_tokens = int(timings["draft_n"])
                draft_accepted = int(timings.get("draft_n_accepted") or 0)
            server_tokens = obj.get("tokens_predicted") or obj.get("predicted_n") or timings.get(
                "predicted_n"
            )
            if content:
                output += content
                if isinstance(server_tokens, int) and server_tokens > previous_server_tokens:
                    token_count = server_tokens
                    previous_server_tokens = server_tokens
                else:
                    token_count += 1
                if on_progress:
                    elapsed = max(time.perf_counter() - start, 0.001)
                    on_progress(output, token_count, elapsed)
    except Exception as exc:
        if token_count <= 0 or not is_premature_stream_end(exc):
            raise

    elapsed = max(time.perf_counter() - start, 0.001)
    tps = final_tps if final_tps > 0 else (token_count / elapsed if token_count > 0 else 0.0)
    return tps, output, token_count, elapsed, draft_tokens, draft_accepted


def stream_openai_chat(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    seed: int,
    timeout: float,
    on_progress: Callable[[str, int, float], None] | None = None,
) -> tuple[float, str, int, float, int, int]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        # llama.cpp extension: ship its `timings` block in stream chunks so we
        # can read draft_n / draft_n_accepted (speculative decoding stats).
        "timings_per_token": True,
    }

    start = time.perf_counter()
    output = ""
    token_count = 0
    server_token_count = 0
    draft_tokens = 0
    draft_accepted = 0

    response = requests.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        stream=True,
        timeout=timeout,
    )
    response.raise_for_status()

    try:
        for raw_line in response.iter_lines():
            obj = parse_stream_line(raw_line)
            if not obj:
                continue

            timings = obj.get("timings")
            if isinstance(timings, dict) and timings.get("draft_n"):
                draft_tokens = int(timings["draft_n"])
                draft_accepted = int(timings.get("draft_n_accepted") or 0)

            usage = obj.get("usage")
            if isinstance(usage, dict):
                completion_tokens = usage.get("completion_tokens")
                if isinstance(completion_tokens, int) and completion_tokens > 0:
                    server_token_count = completion_tokens
                    token_count = completion_tokens
                    if on_progress:
                        elapsed = max(time.perf_counter() - start, 0.001)
                        on_progress(output, token_count, elapsed)

            choices = obj.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta")
                    if not isinstance(delta, dict):
                        continue
                    content = delta.get("content", "")
                    if isinstance(content, str) and content:
                        output += content
                        if server_token_count == 0:
                            token_count = estimate_token_count(output)
                        if on_progress:
                            elapsed = max(time.perf_counter() - start, 0.001)
                            on_progress(output, token_count, elapsed)
    except Exception as exc:
        if not output or not is_premature_stream_end(exc):
            raise

    elapsed = max(time.perf_counter() - start, 0.001)
    if server_token_count > 0:
        token_count = server_token_count
    else:
        token_count = estimate_token_count(output)
    tps = token_count / elapsed if token_count > 0 else 0.0
    return tps, output, token_count, elapsed, draft_tokens, draft_accepted


def summarize_samples(samples: list[float]) -> tuple[float, float, float, float]:
    if not samples:
        return 0.0, 0.0, 0.0, 0.0
    return (
        statistics.mean(samples),
        statistics.median(samples),
        min(samples),
        max(samples),
    )


def serialize_run(run: RunMetrics) -> dict[str, str]:
    return {
        "run_name": run.run_name,
        "api": run.api,
        "model": run.model or "",
        "runs": str(run.runs),
        "avg_tps": f"{run.avg_tps:.10f}",
        "median_tps": f"{run.median_tps:.10f}",
        "min_tps": f"{run.min_tps:.10f}",
        "max_tps": f"{run.max_tps:.10f}",
        "total_tokens": str(run.total_tokens),
        "total_elapsed_s": f"{run.total_elapsed_s:.10f}",
        "samples": json.dumps(run.samples),
        "sample_tokens": json.dumps(run.sample_tokens),
        "sample_elapsed_s": json.dumps(run.sample_elapsed_s),
        "prompt_preview": run.prompt_preview,
        "draft_tokens": str(run.draft_tokens),
        "draft_accepted": str(run.draft_accepted),
    }


def _safe_json_list(raw: str, cast_fn: Callable[[Any], Any]) -> list[Any]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    result: list[Any] = []
    for item in parsed:
        try:
            result.append(cast_fn(item))
        except Exception:
            continue
    return result


def read_history(path: Path) -> list[RunMetrics]:
    if not path.exists():
        return []

    if path.suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            return []
        rows: list[RunMetrics] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                rows.append(
                    RunMetrics(
                        run_name=str(item.get("run_name", "")),
                        api=str(item.get("api", "")),
                        model=(str(item.get("model")) if item.get("model") else None),
                        samples=_safe_json_list(json.dumps(item.get("samples", [])), float),
                        sample_tokens=_safe_json_list(
                            json.dumps(item.get("sample_tokens", [])),
                            int,
                        ),
                        sample_elapsed_s=_safe_json_list(
                            json.dumps(item.get("sample_elapsed_s", [])),
                            float,
                        ),
                        avg_tps=float(item.get("avg_tps", 0) or 0),
                        median_tps=float(item.get("median_tps", 0) or 0),
                        min_tps=float(item.get("min_tps", 0) or 0),
                        max_tps=float(item.get("max_tps", 0) or 0),
                        runs=int(item.get("runs", 0) or 0),
                        total_tokens=int(item.get("total_tokens", 0) or 0),
                        total_elapsed_s=float(item.get("total_elapsed_s", 0) or 0),
                        prompt_preview=str(item.get("prompt_preview", "")),
                        draft_tokens=int(item.get("draft_tokens", 0) or 0),
                        draft_accepted=int(item.get("draft_accepted", 0) or 0),
                    )
                )
            except (TypeError, ValueError):
                continue
        return rows

    csv_rows: list[RunMetrics] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for item in reader:
            if not item:
                continue
            samples = _safe_json_list(item.get("samples", ""), float)
            sample_tokens = _safe_json_list(item.get("sample_tokens", ""), int)
            sample_elapsed_s = _safe_json_list(item.get("sample_elapsed_s", ""), float)

            try:
                csv_rows.append(
                    RunMetrics(
                        run_name=item.get("run_name", ""),
                        api=item.get("api", ""),
                        model=item.get("model") or None,
                        samples=samples,
                        sample_tokens=sample_tokens,
                        sample_elapsed_s=sample_elapsed_s,
                        avg_tps=float(item.get("avg_tps", "0") or 0),
                        median_tps=float(item.get("median_tps", "0") or 0),
                        min_tps=float(item.get("min_tps", "0") or 0),
                        max_tps=float(item.get("max_tps", "0") or 0),
                        runs=int(item.get("runs", "0") or 0),
                        total_tokens=int(item.get("total_tokens", "0") or 0),
                        total_elapsed_s=float(item.get("total_elapsed_s", "0") or 0),
                        prompt_preview=item.get("prompt_preview", ""),
                        draft_tokens=int(item.get("draft_tokens", "0") or 0),
                        draft_accepted=int(item.get("draft_accepted", "0") or 0),
                    )
                )
            except ValueError:
                continue
    return csv_rows


def append_history(path: Path, run: RunMetrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # If the file predates a CSV_FIELDS change, rewrite it with the current
    # header (old rows get default values for the new columns).
    if path.suffix != ".json" and path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as handle:
            existing_fields = next(csv.reader(handle), [])
        if existing_fields != CSV_FIELDS:
            old_rows = read_history(path)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()
                for old_run in old_rows:
                    writer.writerow(serialize_run(old_run))
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(serialize_run(run))


def compress_leaderboard(rows: list[RunMetrics]) -> list[RunMetrics]:
    by_name: dict[str, RunMetrics] = {}
    for row in rows:
        current = by_name.get(row.run_name)
        if current is None or row.avg_tps > current.avg_tps:
            by_name[row.run_name] = row
    best = list(by_name.values())
    best.sort(key=lambda item: item.avg_tps, reverse=True)
    return best


def build_live_panel(
    run_name: str,
    api: str,
    model: str | None,
    url: str,
    run_no: int,
    total_runs: int,
    latest_tps: float,
    latest_tokens: int,
    latest_elapsed: float,
    samples: list[float],
    latest_output: str,
    prompt: str,
    board_rows: list[RunMetrics],
    generation_budget: int,
    context_size: int,
    output_lines: int,
    prompt_lines: int,
    show_top: int,
    draft_tokens: int = 0,
    draft_accepted: int = 0,
) -> Group:
    avg, median, min_tps, max_tps = summarize_samples(samples)
    live_tps = latest_tokens / latest_elapsed if latest_elapsed > 0 else 0.0
    last_line = (
        f"{latest_tokens} tokens in {latest_elapsed:.2f}s"
        if latest_elapsed > 0 and latest_tokens > 0
        else "-"
    )

    terminal_width = console.size.width
    side_width = max(42, terminal_width // 2 - 4)
    output_width = max(36, side_width - 10)
    prompt_width = max(60, terminal_width - 12)

    header = Table.grid(expand=True)
    header.add_column(ratio=1)
    header.add_column(width=18)
    header.add_row(
        Text(f"{run_name} Single-Port Benchmark", style="bold white"),
        Align.right(Text(f"{avg:.2f} avg", style="bold green")),
    )

    meta = Text()
    meta.append("api: ", style="dim")
    meta.append(api, style="bold cyan")
    meta.append(" | model: ", style="dim")
    meta.append(model or "-", style="white")
    meta.append(" | url: ", style="dim")
    meta.append(url, style="white")
    meta.append(" | ctx: ", style="dim")
    meta.append(str(context_size), style="bold white")
    meta.append(" | gen: ", style="dim")
    meta.append(str(generation_budget), style="bold white")
    header.add_row(meta, Text(""))

    promo = Text()
    promo.append("Channel: ", style="dim")
    promo.append(YOUTUBE_CHANNEL, style="bold magenta")
    promo.append(" | ", style="dim")
    promo.append(YOUTUBE_URL, style="bold blue")
    header.add_row(promo, Text(""))

    stats = Table.grid(expand=True)
    stats.add_column(justify="left")
    stats.add_column(justify="right")
    stats.add_row("run", f"{run_no}/{total_runs}")
    stats.add_row("latest tok/s", f"{latest_tps:.2f}")
    stats.add_row("live tok/s", f"{live_tps:.2f}")
    stats.add_row("avg tok/s", f"{avg:.2f}")
    stats.add_row("median tok/s", f"{median:.2f}")
    stats.add_row("min/max", f"{min_tps:.2f} / {max_tps:.2f}")
    stats.add_row("last run", last_line)
    if draft_tokens > 0:
        stats.add_row(
            "draft accept",
            f"{draft_accepted}/{draft_tokens} ({100 * draft_accepted / draft_tokens:.0f}%)",
        )

    output_panel = Panel(
        Text(
            trim_output(latest_output, max_lines=output_lines, max_width=output_width),
            style="white",
        ),
        title="Live output",
        border_style="dim",
        box=box.ROUNDED,
        height=output_lines + 3,
    )

    run_panel = Panel(
        Group(
            Align.center(big_number(live_tps if live_tps > 0 else (latest_tps if latest_tps > 0 else avg), "green")),
            stats,
            output_panel,
        ),
        title="[bold green]Current Run[/bold green]",
        border_style="green",
        box=box.ROUNDED,
    )

    prompt_panel = Panel(
        Text(trim_output(prompt, max_lines=prompt_lines, max_width=prompt_width), style="white"),
        title=f"Prompt ({prompt_lines} lines)",
        border_style="white",
        box=box.ROUNDED,
    )

    board = compress_leaderboard(board_rows)
    board_table = Table(title=f"Leaderboard (Top {show_top})", box=box.SIMPLE_HEAVY)
    board_table.add_column("rank", justify="right")
    board_table.add_column("run")
    board_table.add_column("avg tok/s", justify="right")
    board_table.add_column("accept%", justify="right")
    board_table.add_column("tokens", justify="right")
    board_table.add_column("seconds", justify="right")
    board_table.add_column("runs", justify="right")

    for idx, row in enumerate(board[:show_top], start=1):
        board_table.add_row(
            str(idx),
            row.run_name,
            f"{row.avg_tps:.2f}",
            accept_pct(row),
            str(row.total_tokens),
            f"{row.total_elapsed_s:.2f}",
            str(row.runs),
        )

    samples_table = Table(title="Samples", box=box.MINIMAL_HEAVY_HEAD)
    samples_table.add_column("#", justify="right")
    samples_table.add_column("tok/s", justify="right")
    recent = samples[-8:]
    start_index = len(samples) - len(recent) + 1
    for idx, value in enumerate(recent, start=start_index):
        samples_table.add_row(str(idx), f"{value:.2f}")

    right_panel = Panel(
        Group(board_table, samples_table),
        title="[bold cyan]History[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
    )

    columns = Table.grid(expand=True)
    columns.add_column(ratio=3)
    columns.add_column(ratio=2)
    columns.add_row(run_panel, right_panel)

    return Group(Panel(header, border_style="white", box=box.ROUNDED), prompt_panel, columns)


def print_leaderboard(history_rows: list[RunMetrics], limit: int) -> None:
    board = compress_leaderboard(history_rows)
    table = Table(title="Leaderboard (best avg tok/s per run name)")
    table.add_column("rank", justify="right")
    table.add_column("run")
    table.add_column("avg tok/s", justify="right")
    table.add_column("median", justify="right")
    table.add_column("accept%", justify="right")
    table.add_column("tokens", justify="right")
    table.add_column("seconds", justify="right")
    table.add_column("runs", justify="right")

    for idx, row in enumerate(board[:limit], start=1):
        table.add_row(
            str(idx),
            row.run_name,
            f"{row.avg_tps:.2f}",
            f"{row.median_tps:.2f}",
            accept_pct(row),
            str(row.total_tokens),
            f"{row.total_elapsed_s:.2f}",
            str(row.runs),
        )

    console.print(table)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run benchmark against one local server and append results to CSV leaderboard."
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Server host (default: {DEFAULT_HOST})")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=(
            "Server port. If omitted (with --url also omitted), probes "
            f"{CANDIDATE_PORTS} in order and uses whichever responds."
        ),
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Full server URL. Overrides --host and --port when provided.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Name for this benchmark run group. If omitted, auto-detected from "
            "the server's model alias (example: qwen36-dflash)."
        ),
    )
    parser.add_argument(
        "--api",
        choices=["auto", "completion", "openai"],
        default="auto",
        help="API format to use.",
    )
    parser.add_argument("--model", default=None, help="OpenAI model id. Auto-detected when omitted.")
    parser.add_argument("--runs", type=int, default=10, help="How many requests to send.")
    parser.add_argument(
        "--n-predict",
        type=int,
        default=DEFAULT_GENERATION_TOKENS,
        help=f"For /completion API (default: {DEFAULT_GENERATION_TOKENS}).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_GENERATION_TOKENS,
        help=f"For OpenAI API (default: {DEFAULT_GENERATION_TOKENS}).",
    )
    parser.add_argument(
        "--use-full-context",
        action="store_true",
        help="Use context_size - estimated_prompt_tokens - reserve as generation budget.",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=DEFAULT_CONTEXT_SIZE,
        help=f"Server context window from Docker config (default: {DEFAULT_CONTEXT_SIZE}).",
    )
    parser.add_argument(
        "--context-reserve",
        type=int,
        default=DEFAULT_CONTEXT_RESERVE,
        help=f"Tokens reserved so prompt + output stays under context (default: {DEFAULT_CONTEXT_RESERVE}).",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Seed for generation.")
    parser.add_argument("--timeout", type=float, default=900.0, help="Request timeout in seconds.")
    parser.add_argument(
        "--history-file",
        default=str(DEFAULT_HISTORY),
        help=f"Path to leaderboard history CSV (default: {DEFAULT_HISTORY}).",
    )
    parser.add_argument("--show-top", type=int, default=20, help="How many leaderboard rows to print.")
    parser.add_argument("--output-lines", type=int, default=16, help="Visible lines in output panel.")
    parser.add_argument("--prompt-lines", type=int, default=5, help="Visible lines in prompt panel.")
    parser.add_argument(
        "--prompt",
        default=(
            "Write a Python program to find the nth Fibonacci number using recursion. "
            "Explain briefly and include code."
        ),
    )
    args = parser.parse_args()

    if args.url or args.port is not None:
        base_url = resolve_base_url(args.url, args.host, args.port or DEFAULT_PORT)
        if not check_server(base_url, timeout=5.0):
            raise SystemExit(
                f"Server is not reachable at {base_url}. Start the container first "
                "and confirm the published port."
            )
    else:
        base_url = None
        for candidate_port in CANDIDATE_PORTS:
            candidate_url = f"http://{args.host}:{candidate_port}"
            if check_server(candidate_url, timeout=3.0):
                base_url = candidate_url
                break
        if base_url is None:
            raise SystemExit(
                f"No server found on {args.host} ports {CANDIDATE_PORTS}. "
                "Start a container first, or pass --port/--url explicitly."
            )
        console.print(Text(f"Auto-detected server at {base_url}", style="dim"))

    api = detect_api(base_url, args.api, timeout=10.0)

    # Best-effort alias lookup, independent of --api: used for naming even
    # when benchmarking via the raw /completion endpoint.
    detected_alias: str | None = None
    try:
        detected_alias = detect_model(base_url, args.model, timeout=10.0)
    except Exception:
        pass

    model: str | None = detected_alias if api == "openai" else None
    run_name = args.run_name or detected_alias
    if run_name is None:
        raise SystemExit(
            f"Could not auto-detect a model alias from {base_url}/v1/models, and no "
            "--run-name was given. Pass --run-name explicitly."
        )
    if not args.run_name:
        console.print(Text(f"Auto-detected run name: {run_name}", style="dim"))
    args.run_name = run_name

    full_context_budget = auto_generation_tokens(
        args.prompt,
        context_size=args.context_size,
        reserve_tokens=args.context_reserve,
    )
    n_predict = full_context_budget if args.use_full_context else args.n_predict
    max_tokens = full_context_budget if args.use_full_context else args.max_tokens

    samples: list[float] = []
    sample_tokens: list[int] = []
    sample_elapsed_s: list[float] = []
    draft_tokens_total = 0
    draft_accepted_total = 0
    latest_output = ""
    latest_tps = 0.0
    latest_tokens = 0
    latest_elapsed = 0.0
    history_path = Path(args.history_file)
    history = read_history(history_path)

    def make_preview_run(
        include_in_progress: bool,
        current_tokens: int,
        current_elapsed: float,
    ) -> RunMetrics:
        preview_samples = list(samples)
        preview_tokens = list(sample_tokens)
        preview_elapsed = list(sample_elapsed_s)

        if include_in_progress and current_tokens > 0 and current_elapsed > 0:
            live_tps = current_tokens / current_elapsed
            preview_samples.append(live_tps)
            preview_tokens.append(current_tokens)
            preview_elapsed.append(current_elapsed)

        avg, median, min_tps, max_tps = summarize_samples(preview_samples)
        return RunMetrics(
            run_name=args.run_name,
            api=api,
            model=model,
            samples=preview_samples,
            sample_tokens=preview_tokens,
            sample_elapsed_s=preview_elapsed,
            avg_tps=avg,
            median_tps=median,
            min_tps=min_tps,
            max_tps=max_tps,
            runs=len(preview_samples),
            total_tokens=sum(preview_tokens),
            total_elapsed_s=sum(preview_elapsed),
            prompt_preview=clip_text(args.prompt, 160),
            draft_tokens=draft_tokens_total,
            draft_accepted=draft_accepted_total,
        )

    def render(run_no: int, board_rows: list[RunMetrics]) -> Group:
        return build_live_panel(
            run_name=args.run_name,
            api=api,
            model=model,
            url=base_url,
            run_no=run_no,
            total_runs=args.runs,
            latest_tps=latest_tps,
            latest_tokens=latest_tokens,
            latest_elapsed=latest_elapsed,
            samples=samples,
            latest_output=latest_output,
            prompt=args.prompt,
            board_rows=board_rows,
            generation_budget=max_tokens if api == "openai" else n_predict,
            context_size=args.context_size,
            output_lines=args.output_lines,
            prompt_lines=args.prompt_lines,
            show_top=args.show_top,
            draft_tokens=draft_tokens_total,
            draft_accepted=draft_accepted_total,
        )

    with Live(render(0, history), refresh_per_second=5, console=console, screen=True) as live:
        for run_no in range(1, args.runs + 1):

            def progress_cb(output: str, token_count: int, elapsed: float) -> None:
                nonlocal latest_output, latest_tokens, latest_elapsed
                latest_output = output
                latest_tokens = token_count
                latest_elapsed = elapsed
                preview_run = make_preview_run(
                    include_in_progress=True,
                    current_tokens=token_count,
                    current_elapsed=elapsed,
                )
                live.update(render(run_no, history + [preview_run]))

            if api == "completion":
                tps, output, token_count, elapsed, run_draft, run_accepted = stream_completion(
                    base_url,
                    args.prompt,
                    n_predict,
                    args.seed,
                    args.timeout,
                    on_progress=progress_cb,
                )
            else:
                assert model is not None
                tps, output, token_count, elapsed, run_draft, run_accepted = stream_openai_chat(
                    base_url,
                    model,
                    args.prompt,
                    max_tokens,
                    args.seed,
                    args.timeout,
                    on_progress=progress_cb,
                )

            latest_tps = tps
            latest_tokens = token_count
            latest_elapsed = elapsed
            latest_output = output

            samples.append(tps)
            sample_tokens.append(token_count)
            sample_elapsed_s.append(elapsed)
            draft_tokens_total += run_draft
            draft_accepted_total += run_accepted

            preview_run = make_preview_run(
                include_in_progress=False,
                current_tokens=0,
                current_elapsed=0.0,
            )
            live.update(render(run_no, history + [preview_run]))

    avg, median, min_tps, max_tps = summarize_samples(samples)

    run = RunMetrics(
        run_name=args.run_name,
        api=api,
        model=model,
        samples=samples,
        sample_tokens=sample_tokens,
        sample_elapsed_s=sample_elapsed_s,
        avg_tps=avg,
        median_tps=median,
        min_tps=min_tps,
        max_tps=max_tps,
        runs=len(samples),
        total_tokens=sum(sample_tokens),
        total_elapsed_s=sum(sample_elapsed_s),
        prompt_preview=clip_text(args.prompt, 160),
        draft_tokens=draft_tokens_total,
        draft_accepted=draft_accepted_total,
    )

    append_history(history_path, run)
    history.append(run)

    draft_note = (
        f" | draft {run.draft_accepted}/{run.draft_tokens} accepted ({accept_pct(run)})"
        if run.draft_tokens > 0
        else ""
    )
    console.print()
    console.print(
        Text(
            (
                f"Saved run '{args.run_name}' to {history_path} | avg {avg:.2f} tok/s "
                f"| total {run.total_tokens} tokens in {run.total_elapsed_s:.2f}s{draft_note}"
            ),
            style="bold cyan",
        )
    )
    print_leaderboard(history, limit=args.show_top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
