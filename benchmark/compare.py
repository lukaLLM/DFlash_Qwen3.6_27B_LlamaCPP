#!/usr/bin/env python3
"""Side-by-side terminal benchmark: Qwen 3.6 baseline vs Qwen 3.6 + DFlash."""

from __future__ import annotations

import argparse
import csv
import statistics
import threading
import time
from _thread import LockType
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from leaderboard import (
    APIMode,
    auto_generation_tokens,
    big_number,
    check_server,
    detect_api,
    detect_model,
    stream_completion,
    stream_openai_chat,
    trim_output,
)
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
# baseline listens on 8000; dflash and mtp are mutually-exclusive alternates
# on 8001 (see docker/docker-compose.yaml).
DEFAULT_LEFT_PORT = 8000
DEFAULT_RIGHT_PORT = 8001
DEFAULT_CONTEXT_SIZE = 32768
DEFAULT_CONTEXT_RESERVE = 128
DEFAULT_GENERATION_TOKENS = 4096
DEFAULT_RESULTS_CSV = Path(__file__).with_name("comparison_runs.csv")
YOUTUBE_CHANNEL = "@LukaszGawendaAI"
YOUTUBE_URL = "www.youtube.com/@LukaszGawendaAI"
CSV_FIELDS = [
    "session_name",
    "mode",
    "run_index",
    "tokens",
    "elapsed_s",
    "tok_per_s",
]
RunStatus = Literal["ready", "running", "error"]


@dataclass(frozen=True)
class EndpointConfig:
    """Resolved endpoint settings used by one side of the comparison."""

    name: str
    url: str
    api: str
    model: str | None
    color: str


@dataclass(frozen=True)
class EndpointSnapshot:
    """Thread-safe point-in-time view of benchmark state."""

    name: str
    url: str
    color: str
    status: RunStatus
    output: str
    error: str | None
    run_index: int
    completed: int
    token_count: int
    elapsed: float
    last_tokens: int
    last_elapsed: float
    last_tps: float
    samples: tuple[float, ...]

    @property
    def live_tps(self) -> float:
        """Return current in-flight tok/s."""
        if self.elapsed <= 0 or self.token_count <= 0:
            return 0.0
        return self.token_count / self.elapsed

    @property
    def avg_tps(self) -> float:
        """Return completed-run average tok/s."""
        if not self.samples:
            return 0.0
        return statistics.mean(self.samples)

    @property
    def display_tps(self) -> float:
        """Return the best tok/s value for the large dashboard number."""
        if self.live_tps > 0:
            return self.live_tps
        if self.last_tps > 0:
            return self.last_tps
        return self.avg_tps


@dataclass
class EndpointState:
    """Mutable benchmark state guarded for concurrent UI reads."""

    config: EndpointConfig
    output: str = ""
    error: str | None = None
    running: bool = False
    run_index: int = 0
    elapsed: float = 0.0
    token_count: int = 0
    last_tokens: int = 0
    last_elapsed: float = 0.0
    last_tps: float = 0.0
    samples: list[float] = field(default_factory=list)
    lock: LockType = field(default_factory=threading.Lock)

    def start_run(self) -> None:
        """Reset in-flight fields and mark a request as running."""
        with self.lock:
            self.output = ""
            self.error = None
            self.running = True
            self.elapsed = 0.0
            self.token_count = 0
            self.last_tokens = 0
            self.last_elapsed = 0.0
            self.last_tps = 0.0
            self.run_index += 1

    def update_progress(self, output: str, token_count: int, elapsed: float) -> None:
        """Store streaming progress for the live UI."""
        with self.lock:
            self.output = output[-12000:]
            self.token_count = token_count
            self.elapsed = elapsed

    def finish_run(
        self,
        tps: float,
        output: str,
        token_count: int,
        elapsed: float,
        error: str | None = None,
    ) -> None:
        """Store final request metrics."""
        with self.lock:
            self.output = output[-12000:]
            self.error = error
            self.running = False
            self.token_count = token_count
            self.elapsed = elapsed
            self.last_tokens = token_count
            self.last_elapsed = elapsed
            self.last_tps = tps
            if error is None and tps > 0:
                self.samples.append(tps)

    def snapshot(self) -> EndpointSnapshot:
        """Return a stable copy for rendering and CSV writes."""
        with self.lock:
            status: RunStatus = "running" if self.running else "ready"
            if self.error:
                status = "error"
            return EndpointSnapshot(
                name=self.config.name,
                url=self.config.url,
                color=self.config.color,
                status=status,
                output=self.output,
                error=self.error,
                run_index=self.run_index,
                completed=len(self.samples),
                token_count=self.token_count,
                elapsed=self.elapsed,
                last_tokens=self.last_tokens,
                last_elapsed=self.last_elapsed,
                last_tps=self.last_tps,
                samples=tuple(self.samples),
            )


def resolve_url(url: str | None, host: str, port: int) -> str:
    """Resolve a CLI URL/host/port pair into a base URL."""
    if url:
        return url
    return f"http://{host}:{port}"


def ascii_heading(text: str, color: str, max_width: int) -> Text:
    """Render a compact figlet heading when it fits the terminal."""
    if pyfiglet and max_width >= 54:
        rendered = pyfiglet.figlet_format(text.upper(), font="small")
        lines = [line.rstrip() for line in rendered.splitlines() if line.strip()]
        if lines and max(len(line) for line in lines) <= max_width:
            return Text("\n".join(lines), style=f"bold {color}")
    return Text(text.upper(), style=f"bold {color}")


def append_csv_row(csv_file: Path, session_name: str, snapshot: EndpointSnapshot) -> None:
    """Append one completed request to the comparison CSV."""
    if snapshot.last_tokens <= 0 or snapshot.last_elapsed <= 0:
        return

    csv_file.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not csv_file.exists() or csv_file.stat().st_size == 0
    with csv_file.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(
            {
                "session_name": session_name,
                "mode": snapshot.name,
                "run_index": snapshot.run_index,
                "tokens": snapshot.last_tokens,
                "elapsed_s": f"{snapshot.last_elapsed:.6f}",
                "tok_per_s": f"{snapshot.last_tps:.6f}",
            }
        )


def run_endpoint_request(
    state: EndpointState,
    prompt: str,
    n_predict: int,
    max_tokens: int,
    seed: int,
    timeout: float,
) -> None:
    """Run one streaming request against one endpoint."""
    state.start_run()
    config = state.config

    def progress_cb(output: str, token_count: int, elapsed: float) -> None:
        state.update_progress(output, token_count, elapsed)

    try:
        if config.api == "completion":
            tps, output, token_count, elapsed = stream_completion(
                config.url,
                prompt,
                n_predict,
                seed,
                timeout,
                on_progress=progress_cb,
            )
        else:
            if config.model is None:
                msg = "OpenAI-compatible endpoint needs a model id."
                raise RuntimeError(msg)
            tps, output, token_count, elapsed = stream_openai_chat(
                config.url,
                config.model,
                prompt,
                max_tokens,
                seed,
                timeout,
                on_progress=progress_cb,
            )
        state.finish_run(tps, output, token_count, elapsed)
    except Exception as exc:
        snapshot = state.snapshot()
        state.finish_run(
            0.0,
            snapshot.output,
            snapshot.token_count,
            max(snapshot.elapsed, 0.001),
            error=str(exc),
        )


def make_stats(snapshot: EndpointSnapshot, total_runs: int) -> Table:
    """Build the small status table for one panel."""
    last_line = "-"
    if snapshot.last_tokens > 0 and snapshot.last_elapsed > 0:
        last_line = f"{snapshot.last_tokens} in {snapshot.last_elapsed:.1f}s"

    stats = Table.grid(expand=True)
    stats.add_column(justify="left")
    stats.add_column(justify="right")
    stats.add_row("status", snapshot.status)
    stats.add_row("run", f"{min(snapshot.run_index, total_runs)}/{total_runs}")
    stats.add_row("completed", str(snapshot.completed))
    stats.add_row("live tok/s", f"{snapshot.live_tps:.2f}")
    stats.add_row("avg tok/s", f"{snapshot.avg_tps:.2f}")
    stats.add_row("last run", last_line)
    return stats


def make_endpoint_panel(
    snapshot: EndpointSnapshot,
    total_runs: int,
    panel_width: int,
    output_lines: int,
) -> Panel:
    """Build one side of the comparison dashboard."""
    output_width = max(34, panel_width - 10)

    if snapshot.error:
        output_text = Text(snapshot.error, style="bold red")
    else:
        output_text = Text(
            trim_output(snapshot.output, max_lines=output_lines, max_width=output_width),
            style=snapshot.color,
        )

    output_panel = Panel(
        output_text,
        title="Live output",
        border_style=snapshot.color,
        box=box.ROUNDED,
        height=output_lines + 3,
    )

    body = Group(
        make_stats(snapshot, total_runs),
        Align.center(big_number(snapshot.display_tps, snapshot.color)),
        Align.center(Text("tok/sec", style=f"bold {snapshot.color}")),
        output_panel,
    )

    return Panel(
        body,
        title=f"[bold {snapshot.color}]{snapshot.name}[/bold {snapshot.color}]",
        border_style=snapshot.color,
        box=box.ROUNDED,
    )


def make_header(
    left: EndpointConfig,
    right: EndpointConfig,
    session_name: str,
    generation_budget: int,
    context_size: int,
    left_snap: EndpointSnapshot | None = None,
    right_snap: EndpointSnapshot | None = None,
) -> Panel:
    """Build the channel and quick-info header."""
    header = Table.grid(expand=True)
    header.add_column(ratio=1)
    header.add_column(justify="right")

    brand = Text()
    brand.append(YOUTUBE_CHANNEL, style="bold magenta")
    brand.append(" | ", style="dim")
    brand.append(YOUTUBE_URL, style="bold blue")
    header.add_row(brand, Text("DFlash benchmark lab", style="bold white"))

    info = Text()
    info.append("session: ", style="dim")
    info.append(session_name, style="bold white")
    info.append(" | ctx: ", style="dim")
    info.append(str(context_size), style="bold white")
    info.append(" | gen: ", style="dim")
    info.append(str(generation_budget), style="bold white")
    header.add_row(info, Text(f"{left.url}  vs  {right.url}", style="white"))

    models = Text()
    models.append("left: ", style="dim")
    models.append(left.model or left.api, style=left.color)
    models.append(" | right: ", style="dim")
    models.append(right.model or right.api, style=right.color)

    speedup_text = Text()
    if left_snap and right_snap and left_snap.avg_tps > 0 and right_snap.avg_tps > 0:
        speedup = right_snap.avg_tps / left_snap.avg_tps
        if speedup >= 1.0:
            speedup_text.append(f"{right_snap.name}", style=f"bold {right.color}")
            speedup_text.append(" is ", style="white")
            speedup_text.append(f"{speedup:.2f}x faster", style="bold yellow")
            speedup_text.append(" than ", style="white")
            speedup_text.append(f"{left_snap.name}", style=f"bold {left.color}")
            speedup_text.append(" (avg)", style="dim")
        else:
            inv = left_snap.avg_tps / right_snap.avg_tps
            speedup_text.append(f"{left_snap.name}", style=f"bold {left.color}")
            speedup_text.append(" is ", style="white")
            speedup_text.append(f"{inv:.2f}x faster", style="bold yellow")
            speedup_text.append(" than ", style="white")
            speedup_text.append(f"{right_snap.name}", style=f"bold {right.color}")
            speedup_text.append(" (avg)", style="dim")
    else:
        speedup_text.append("speed: calculating...", style="dim")

    header.add_row(models, speedup_text)

    return Panel(header, border_style="white", box=box.ROUNDED)


def make_ui(
    left_state: EndpointState,
    right_state: EndpointState,
    session_name: str,
    prompt: str,
    total_runs: int,
    generation_budget: int,
    context_size: int,
    output_lines: int,
    prompt_lines: int,
) -> Group:
    """Build the full live dashboard."""
    left = left_state.snapshot()
    right = right_state.snapshot()
    terminal_width = console.size.width
    panel_width = max(48, (terminal_width - 8) // 2)
    prompt_width = max(70, terminal_width - 12)

    prompt_panel = Panel(
        Text(trim_output(prompt, max_lines=prompt_lines, max_width=prompt_width), style="white"),
        title=f"Prompt ({prompt_lines} lines)",
        border_style="white",
        box=box.ROUNDED,
    )

    panels = Table.grid(expand=True)
    panels.add_column(ratio=1)
    panels.add_column(width=2)
    panels.add_column(ratio=1)
    panels.add_row(
        make_endpoint_panel(left, total_runs, panel_width, output_lines),
        Text(""),
        make_endpoint_panel(right, total_runs, panel_width, output_lines),
    )

    return Group(
        make_header(
            left_state.config,
            right_state.config,
            session_name,
            generation_budget,
            context_size,
            left_snap=left,
            right_snap=right,
        ),
        prompt_panel,
        panels,
    )


def refresh_until_done(
    live: Live,
    threads: list[threading.Thread],
    render: Callable[[], Group],
) -> None:
    """Refresh the live display until all worker threads finish."""
    while any(thread.is_alive() for thread in threads):
        live.update(render())
        time.sleep(0.12)
    for thread in threads:
        thread.join()
    live.update(render())


def resolve_endpoint(
    name: str | None,
    url: str,
    api_mode: APIMode,
    model: str | None,
    color: str,
    timeout: float,
) -> EndpointConfig:
    """Resolve API mode, model id, and display name for an endpoint.

    `name` is the label shown in the dashboard/CSV; when None it is
    auto-detected from the server's model alias so whichever container
    (baseline/dflash/mtp) is actually running gets labeled correctly.
    """
    if not check_server(url, timeout=5.0):
        raise SystemExit(f"No server reachable at {url}. Start the container first.")

    api = detect_api(url, api_mode, timeout=timeout)

    # Best-effort alias lookup, independent of --api, used for naming even
    # when the endpoint is benchmarked via the raw /completion endpoint.
    detected_alias: str | None = None
    try:
        detected_alias = detect_model(url, model, timeout=timeout)
    except Exception:
        pass

    detected_model = detected_alias if api == "openai" else model
    display_name = name or detected_alias or url
    return EndpointConfig(
        name=display_name,
        url=url,
        api=api,
        model=detected_model,
        color=color,
    )


def print_final(left: EndpointSnapshot, right: EndpointSnapshot, csv_path: Path) -> None:
    """Print the final comparison summary."""
    left_avg = left.avg_tps
    right_avg = right.avg_tps
    speedup = right_avg / left_avg if left_avg > 0 else 0.0

    table = Table(title="Final result", box=box.ROUNDED)
    table.add_column("side")
    table.add_column("avg tok/s", justify="right")
    table.add_column("runs", justify="right")
    table.add_column("last tok/s", justify="right")
    table.add_row(left.name, f"{left_avg:.2f}", str(left.completed), f"{left.last_tps:.2f}")
    table.add_row(right.name, f"{right_avg:.2f}", str(right.completed), f"{right.last_tps:.2f}")
    console.print(table)
    console.print(Text(f"{right.name} / {left.name}: {speedup:.2f}x", style="bold cyan"))
    console.print(Text(f"Saved per-request rows to: {csv_path}", style="bold cyan"))


def positive_int(value: str) -> int:
    """Parse a positive integer CLI value."""
    parsed = int(value)
    if parsed < 1:
        msg = "value must be >= 1"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def main() -> int:
    """Run the side-by-side benchmark."""
    parser = argparse.ArgumentParser(
        description="Compare Qwen 3.6 baseline vs DFlash in a side-by-side terminal dashboard."
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Server host ({DEFAULT_HOST}).")
    parser.add_argument("--left-port", type=int, default=DEFAULT_LEFT_PORT)
    parser.add_argument("--right-port", type=int, default=DEFAULT_RIGHT_PORT)
    parser.add_argument("--left-url", "--baseline-url", dest="left_url")
    parser.add_argument("--right-url", "--dflash-url", dest="right_url")
    parser.add_argument(
        "--left-name", default=None, help="Display name. Auto-detected from the server's alias if omitted."
    )
    parser.add_argument(
        "--right-name", default=None, help="Display name. Auto-detected from the server's alias if omitted."
    )
    parser.add_argument(
        "--left-model", default=None, help="OpenAI model id. Auto-detected from /v1/models if omitted."
    )
    parser.add_argument(
        "--right-model", default=None, help="OpenAI model id. Auto-detected from /v1/models if omitted."
    )
    parser.add_argument("--api", choices=["auto", "completion", "openai"], default="auto")
    parser.add_argument("--left-api", choices=["auto", "completion", "openai"])
    parser.add_argument("--right-api", choices=["auto", "completion", "openai"])
    parser.add_argument("--runs", type=positive_int, default=10)
    parser.add_argument("--n-predict", type=positive_int, default=DEFAULT_GENERATION_TOKENS)
    parser.add_argument("--max-tokens", type=positive_int, default=DEFAULT_GENERATION_TOKENS)
    parser.add_argument(
        "--use-full-context",
        action="store_true",
        help="Use context_size - estimated_prompt_tokens - reserve as generation budget.",
    )
    parser.add_argument("--context-size", type=positive_int, default=DEFAULT_CONTEXT_SIZE)
    parser.add_argument("--context-reserve", type=positive_int, default=DEFAULT_CONTEXT_RESERVE)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output-lines", type=positive_int, default=16)
    parser.add_argument("--prompt-lines", type=positive_int, default=4)
    parser.add_argument(
        "--session-name",
        default=f"compare-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Session name saved in CSV rows.",
    )
    parser.add_argument(
        "--csv-file",
        default=str(DEFAULT_RESULTS_CSV),
        help=f"CSV output path (default: {DEFAULT_RESULTS_CSV}).",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run one endpoint after the other instead of in parallel.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Explain speculative decoding in large language models by creating a clear "
            "8-frame animation storyboard. For each frame, include: frame title, visual "
            "scene description, short on-screen text, narration, and transition. Compare "
            "plain autoregressive decoding with draft-based speculative decoding. Keep it "
            "beginner-friendly, accurate, and suitable for a 60-second educational animation."
        ),
    )
    args = parser.parse_args()

    left_url = resolve_url(args.left_url, args.host, args.left_port)
    right_url = resolve_url(args.right_url, args.host, args.right_port)
    left_api = args.left_api or args.api
    right_api = args.right_api or args.api

    full_context_budget = auto_generation_tokens(
        args.prompt,
        context_size=args.context_size,
        reserve_tokens=args.context_reserve,
    )
    n_predict = full_context_budget if args.use_full_context else args.n_predict
    max_tokens = full_context_budget if args.use_full_context else args.max_tokens
    left_config = resolve_endpoint(
        args.left_name,
        left_url,
        left_api,
        args.left_model,
        "green",
        timeout=10.0,
    )
    right_config = resolve_endpoint(
        args.right_name,
        right_url,
        right_api,
        args.right_model,
        "red",
        timeout=10.0,
    )
    generation_budget = (
        max_tokens
        if left_config.api == "openai" or right_config.api == "openai"
        else n_predict
    )

    left_state = EndpointState(left_config)
    right_state = EndpointState(right_config)
    csv_path = Path(args.csv_file)

    def render() -> Group:
        return make_ui(
            left_state,
            right_state,
            args.session_name,
            args.prompt,
            args.runs,
            generation_budget,
            args.context_size,
            args.output_lines,
            args.prompt_lines,
        )

    with Live(render(), refresh_per_second=8, screen=True, console=console) as live:
        for _ in range(args.runs):
            left_thread = threading.Thread(
                target=run_endpoint_request,
                args=(left_state, args.prompt, n_predict, max_tokens, args.seed, args.timeout),
            )
            right_thread = threading.Thread(
                target=run_endpoint_request,
                args=(right_state, args.prompt, n_predict, max_tokens, args.seed, args.timeout),
            )

            if args.sequential:
                left_thread.start()
                refresh_until_done(live, [left_thread], render)
                append_csv_row(csv_path, args.session_name, left_state.snapshot())

                right_thread.start()
                refresh_until_done(live, [right_thread], render)
                append_csv_row(csv_path, args.session_name, right_state.snapshot())
            else:
                left_thread.start()
                right_thread.start()
                refresh_until_done(live, [left_thread, right_thread], render)
                append_csv_row(csv_path, args.session_name, left_state.snapshot())
                append_csv_row(csv_path, args.session_name, right_state.snapshot())

    print_final(left_state.snapshot(), right_state.snapshot(), csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
