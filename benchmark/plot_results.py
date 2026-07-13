#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib"]
# ///
"""Render the README charts from results_summary.csv.

Reads the labeled tables out of benchmark/results_summary.csv (the `# TABLE n:`
blocks) and writes five PNGs to assets/:

  speed_vs_context.png    - gen tok/s vs context size, base vs DFlash vs +ngram
  accuracy_vs_speed.png   - MATH-500 (N=500) pass@1 + gen tok/s during that run
  accuracy_by_subject.png - per-subject correct counts, base vs DFlash (N=500)
  ngram_ablation.png      - bench_ngram iterative-coding decode t/s per stack
  lcb_accuracy.png        - official LiveCodeBench pass@1, base vs DFlash

Regenerate after editing the CSV:  uv run benchmark/plot_results.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

REPO = Path(__file__).resolve().parent.parent
CSV_FILE = Path(__file__).resolve().parent / "results_summary.csv"
ASSETS = REPO / "assets"

# Palette: DFlash is the story (blue), the ngram stack is the twist (green),
# base is the neutral reference (gray). Identity never rides on color alone:
# every chart carries a legend or named axis ticks plus direct value labels.
BLUE = "#2a78d6"  # DFlash
GREEN = "#1f9d63"  # DFlash + ngram stack
GRAY = "#898781"  # base
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

DPI = 200


def load_tables(path: Path) -> dict[str, list[dict[str, str]]]:
    """Parse the `# TABLE n:` blocks into {table_id: rows-as-dicts}."""
    tables: dict[str, list[dict[str, str]]] = {}
    current: str | None = None
    header: list[str] | None = None
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if line.startswith("# TABLE"):
            current = line.removeprefix("# TABLE").split(":")[0].strip().rstrip(",")
            header = None
            tables[current] = []
            continue
        if not line or current is None:
            continue
        cells = next(csv.reader([line]))
        if header is None:
            header = [c.strip() for c in cells]
            continue
        row = {h: c.strip() for h, c in zip(header, cells)}
        if any(row.values()):
            tables[current].append(row)
    return tables


def style_axes(ax, y_grid: bool = True, x_grid: bool = False) -> None:
    """Recessive chrome: hairline grid, single baseline, muted ticks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(1)
    if y_grid:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    if x_grid:
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.spines["bottom"].set_visible(False)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelcolor=MUTED, length=0, labelsize=9)


def px_to_data(ax, px: float) -> tuple[float, float]:
    """Convert a pixel length to (x, y) data units. Axes limits must be final."""
    inv = ax.transData.inverted()
    x0, y0 = inv.transform((0, 0))
    x1, y1 = inv.transform((px, px))
    return abs(x1 - x0), abs(y1 - y0)


def rounded_bar(ax, x: float, height: float, width: float, color: str) -> None:
    """Vertical bar: 4px-rounded data end, square at the baseline."""
    rx, ry = px_to_data(ax, 4 * DPI / 100)
    rx = min(rx, width / 2)
    ry = min(ry, height / 2)
    x0, x1, h = x - width / 2, x + width / 2, height
    verts = [
        (x0, 0), (x0, h - ry), (x0, h), (x0 + rx, h),
        (x1 - rx, h), (x1, h), (x1, h - ry), (x1, 0), (x0, 0),
    ]
    codes = [
        MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
        MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, linewidth=0))


def rounded_bar_h(ax, y: float, length: float, thickness: float, color: str) -> None:
    """Horizontal bar: 4px-rounded data end (right), square at the baseline."""
    rx, ry = px_to_data(ax, 4 * DPI / 100)
    rx = min(rx, length / 2)
    ry = min(ry, thickness / 2)
    y0, y1, w = y - thickness / 2, y + thickness / 2, length
    verts = [
        (0, y0), (w - rx, y0), (w, y0), (w, y0 + ry),
        (w, y1 - ry), (w, y1), (w - rx, y1), (0, y1), (0, y0),
    ]
    codes = [
        MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
        MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, linewidth=0))


def fmt_ctx(tokens: int) -> str:
    return f"{tokens // 1024}K" if tokens >= 1024 else str(tokens)


def plot_speed_vs_context(tables: dict) -> None:
    rows = [r for r in tables["1"] if r["base_tps"] and r["dflash_tps"]]
    ctx = [int(r["context_tokens"]) for r in rows]
    base = [float(r["base_tps"]) for r in rows]
    dflash = [float(r["dflash_tps"]) for r in rows]
    ngram = [float(r["ngram_tps"]) if r.get("ngram_tps") else None for r in rows]
    speedup = [float(r["speedup_ngram_vs_base"]) if r.get("speedup_ngram_vs_base")
               else None for r in rows]
    xs = range(len(ctx))

    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)

    series = (
        (base, GRAY, "base"),
        (dflash, BLUE, "DFlash"),
        (ngram, GREEN, "DFlash + ngram"),
    )
    for ys, color, label in series:
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color,
                linewidth=2, solid_capstyle="round", solid_joinstyle="round",
                label=label, zorder=3)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o", color=color,
                markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)

    for x, y, s in zip(xs, ngram, speedup):
        if y is not None and s is not None:
            ax.annotate(f"{s:.1f}×", (x, y), xytext=(0, 12),
                        textcoords="offset points", ha="center",
                        fontsize=10, color=INK_2, fontweight="bold")
    for ys in (base, dflash, ngram):
        if ys[-1] is not None:
            ax.annotate(f"{ys[-1]:.0f}", (len(ctx) - 1, ys[-1]), xytext=(14, -3),
                        textcoords="offset points", fontsize=10, color=INK)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([fmt_ctx(c) for c in ctx])
    ax.set_xlim(-0.35, len(ctx) - 0.45)
    ax.set_ylim(0, 380)
    ax.set_xlabel("context size (input = output tokens)", color=MUTED, fontsize=9)
    ax.set_ylabel("generation tok/s (E2E per user)", color=MUTED, fontsize=9)
    ax.set_title("Speculative decoding vs context size",
                 color=INK, fontsize=13, fontweight="bold", loc="left", pad=16)
    ax.text(0, 1.02, "aiperf synthetic sweep, greedy, concurrency 1 — Qwen3.6-27B, RTX PRO 6000 · ×= ngram speedup vs base",
            transform=ax.transAxes, fontsize=9, color=INK_2)
    ax.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=INK_2)

    fig.tight_layout()
    fig.savefig(ASSETS / "speed_vs_context.png", facecolor=SURFACE)
    plt.close(fig)


def plot_accuracy_vs_speed(tables: dict) -> None:
    # full-set rows only (the CSV also carries the older N=100 pair)
    acc = {r["run"]: r for r in tables["2"] if r["n_problems"] == "500"}
    base, dflash = acc["base"], acc["dflash"]
    n = int(base["n_problems"])
    ratio = float(dflash["gen_tps"]) / float(base["gen_tps"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.2), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(f"Same answers — {ratio:.1f}× faster", x=0.06, ha="left",
                 color=INK, fontsize=14, fontweight="bold")
    fig.text(0.06, 0.895,
             f"MATH-500, all {n} problems, greedy (temperature 0), reasoning off",
             fontsize=9, color=INK_2)

    panels = (
        (ax1, f"pass@1 (N={n})", "accuracy_pct", 100, "{:.1f}%"),
        (ax2, "gen tok/s during the run", "gen_tps", 300, "{:.0f}"),
    )
    for ax, title, key, ymax, fmt in panels:
        style_axes(ax)
        ax.set_xlim(-0.7, 1.7)
        ax.set_ylim(0, ymax)
        fig.canvas.draw()  # finalize transforms for px-accurate rounding
        for x, (row, color) in enumerate(((base, GRAY), (dflash, BLUE))):
            val = float(row[key])
            rounded_bar(ax, x, val, 0.22, color)
            ax.annotate(fmt.format(val), (x, val), xytext=(0, 6),
                        textcoords="offset points", ha="center",
                        fontsize=12, color=INK, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["base", "DFlash"], fontsize=10)
        ax.set_title(title, color=INK_2, fontsize=10, pad=10)

    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(ASSETS / "accuracy_vs_speed.png", facecolor=SURFACE)
    plt.close(fig)


def plot_accuracy_by_subject(tables: dict) -> None:
    rows = tables["2B"]
    subjects = [r["subject"] for r in rows]
    totals = [int(r["total"]) for r in rows]
    base = [int(r["base_correct"]) for r in rows]
    dflash = [int(r["dflash_correct"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax, y_grid=False, x_grid=True)

    thickness, gap = 0.30, 0.06
    ys = range(len(subjects))
    ax.set_ylim(len(subjects) - 0.4, -0.6)
    ax.set_xlim(0, max(totals) * 1.12)
    fig.canvas.draw()
    for y, (b, d) in enumerate(zip(base, dflash)):
        rounded_bar_h(ax, y - (thickness + gap) / 2, b, thickness, GRAY)
        rounded_bar_h(ax, y + (thickness + gap) / 2, d, thickness, BLUE)
        ax.annotate(str(b), (b, y - (thickness + gap) / 2), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
        ax.annotate(str(d), (d, y + (thickness + gap) / 2), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)

    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{s}  ({t})" for s, t in zip(subjects, totals)],
                       fontsize=9, color=INK_2)
    ax.set_xlabel("problems correct", color=MUTED, fontsize=9)
    ax.set_title("Within 2 problems in every subject",
                 color=INK, fontsize=13, fontweight="bold", loc="left", pad=16)
    ax.text(0, 1.02, "MATH-500 full set (N=500) — subject totals in parentheses",
            transform=ax.transAxes, fontsize=9, color=INK_2)
    ax.legend(handles=[
        plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=GRAY, label="base"),
        plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=BLUE, label="DFlash"),
    ], loc="lower right", frameon=False, fontsize=10, labelcolor=INK_2)

    fig.tight_layout()
    fig.savefig(ASSETS / "accuracy_by_subject.png", facecolor=SURFACE)
    plt.close(fig)


def plot_ngram_ablation(tables: dict) -> None:
    rows = tables["6"]
    names = [r["run"] for r in rows]
    overall = [float(r["decode_tps"]) for r in rows]
    maint = [float(r["maint_tps"]) for r in rows]
    speedup = [float(r["speedup_vs_base"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax, y_grid=False, x_grid=True)

    thickness, gap = 0.30, 0.06
    ax.set_ylim(len(names) - 0.4, -0.6)
    ax.set_xlim(0, max(maint) * 1.22)
    fig.canvas.draw()
    for y, (o, m, s) in enumerate(zip(overall, maint, speedup)):
        rounded_bar_h(ax, y - (thickness + gap) / 2, o, thickness, BLUE)
        rounded_bar_h(ax, y + (thickness + gap) / 2, m, thickness, GREEN)
        ax.annotate(f"{o:.0f} · {s:.1f}×", (o, y - (thickness + gap) / 2),
                    xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK, fontweight="bold")
        ax.annotate(f"{m:.0f}", (m, y + (thickness + gap) / 2), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)

    ax.set_yticks(list(range(len(names))))
    ax.set_yticklabels(names, fontsize=9, color=INK_2)
    ax.set_xlabel("decode tok/s (× = speedup vs baseline)", color=MUTED, fontsize=9)
    ax.set_title("n-gram lookup doubles DFlash on iterative coding",
                 color=INK, fontsize=13, fontweight="bold", loc="left", pad=16)
    ax.text(0, 1.02,
            "bench_ngram — 18-turn cumulative coding session; maint = turns 10-18, editing the existing code",
            transform=ax.transAxes, fontsize=9, color=INK_2)
    ax.legend(handles=[
        plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=BLUE, label="whole session"),
        plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=GREEN, label="maint phase"),
    ], loc="upper right", frameon=False, fontsize=10, labelcolor=INK_2)

    fig.tight_layout()
    fig.savefig(ASSETS / "ngram_ablation.png", facecolor=SURFACE)
    plt.close(fig)


def plot_lcb_accuracy(tables: dict) -> None:
    acc = {r["run"]: r for r in tables["4"]}
    base, dflash = acc["base"], acc["dflash"]
    n = int(base["n_problems"])

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.set_xlim(-0.7, 1.7)
    ax.set_ylim(0, 30)
    fig.canvas.draw()
    for x, (row, color) in enumerate(((base, GRAY), (dflash, BLUE))):
        val = float(row["pass_at_1_pct"])
        rounded_bar(ax, x, val, 0.22, color)
        ax.annotate(f"{val:.1f}%", (x, val), xytext=(0, 20),
                    textcoords="offset points", ha="center",
                    fontsize=12, color=INK, fontweight="bold")
        ax.annotate(f"{row['correct']}/{n} correct", (x, val), xytext=(0, 6),
                    textcoords="offset points", ha="center",
                    fontsize=9, color=INK_2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["base", "DFlash"], fontsize=10)
    ax.set_title("LiveCodeBench: parity (within noise)",
                 color=INK, fontsize=13, fontweight="bold", loc="left", pad=16)
    ax.text(0, 1.02, f"official LCB harness · release_v5 · Aug-2024 ({n} problems) · n=1, temp 0",
            transform=ax.transAxes, fontsize=8.5, color=INK_2)
    ax.set_ylabel("pass@1 (%)", color=MUTED, fontsize=9)

    fig.tight_layout()
    fig.savefig(ASSETS / "lcb_accuracy.png", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    tables = load_tables(CSV_FILE)
    for tid in ("1", "2", "2B", "4", "6"):
        if tid not in tables or not tables[tid]:
            raise SystemExit(f"results_summary.csv: TABLE {tid} missing or empty")
    ASSETS.mkdir(exist_ok=True)
    plot_speed_vs_context(tables)
    plot_accuracy_vs_speed(tables)
    plot_accuracy_by_subject(tables)
    plot_ngram_ablation(tables)
    plot_lcb_accuracy(tables)
    for name in ("speed_vs_context", "accuracy_vs_speed", "accuracy_by_subject",
                 "ngram_ablation", "lcb_accuracy"):
        print(f"wrote assets/{name}.png")


if __name__ == "__main__":
    main()
