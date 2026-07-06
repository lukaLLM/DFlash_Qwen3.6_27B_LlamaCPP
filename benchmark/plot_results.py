#!/usr/bin/env python3
"""Render the README charts from results_summary.csv.

Reads the labeled tables out of results_summary.csv (the `# TABLE n:` blocks)
and writes three PNGs to assets/:

  speed_vs_context.png    - gen tok/s vs context size, base vs DFlash
  accuracy_vs_speed.png   - MATH-500 pass@1 + gen tok/s during that run
  accuracy_by_subject.png - per-subject correct counts, base vs DFlash

Regenerate after editing the CSV:  uv run python benchmark/plot_results.py
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
CSV_FILE = REPO / "results_summary.csv"
ASSETS = REPO / "assets"

# Palette: DFlash is the story (blue), base is the neutral reference (gray).
# Identity never rides on color alone: every chart carries a legend or named
# axis ticks plus direct value labels.
BLUE = "#2a78d6"  # DFlash
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
    speedup = [float(r["speedup_dflash_vs_base"]) for r in rows]
    xs = range(len(ctx))

    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)

    for ys, color, label in ((base, GRAY, "base"), (dflash, BLUE, "DFlash")):
        ax.plot(xs, ys, color=color, linewidth=2, solid_capstyle="round",
                solid_joinstyle="round", label=label, zorder=3)
        ax.plot(xs, ys, "o", color=color, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)

    for x, y, s in zip(xs, dflash, speedup):
        ax.annotate(f"{s:.1f}×", (x, y), xytext=(0, 12),
                    textcoords="offset points", ha="center",
                    fontsize=10, color=INK_2, fontweight="bold")
    ax.annotate(f"{dflash[-1]:.0f}", (len(ctx) - 1, dflash[-1]), xytext=(14, -3),
                textcoords="offset points", fontsize=10, color=INK)
    ax.annotate(f"{base[-1]:.0f}", (len(ctx) - 1, base[-1]), xytext=(14, -3),
                textcoords="offset points", fontsize=10, color=INK)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([fmt_ctx(c) for c in ctx])
    ax.set_xlim(-0.35, len(ctx) - 0.45)
    ax.set_ylim(0, 320)
    ax.set_xlabel("context size (input = output tokens)", color=MUTED, fontsize=9)
    ax.set_ylabel("generation tok/s", color=MUTED, fontsize=9)
    ax.set_title("DFlash pulls away as context grows",
                 color=INK, fontsize=13, fontweight="bold", loc="left", pad=16)
    ax.text(0, 1.02, "aiperf synthetic sweep, greedy, concurrency 1 — Qwen3.6-27B, RTX PRO 6000",
            transform=ax.transAxes, fontsize=9, color=INK_2)
    ax.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=INK_2)

    fig.tight_layout()
    fig.savefig(ASSETS / "speed_vs_context.png", facecolor=SURFACE)
    plt.close(fig)


def plot_accuracy_vs_speed(tables: dict) -> None:
    acc = {r["run"]: r for r in tables["2"]}
    base, dflash = acc["base"], acc["dflash"]
    n = int(base["n_problems"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.2), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle("Same answers — 3.75× faster", x=0.06, ha="left",
                 color=INK, fontsize=14, fontweight="bold")
    fig.text(0.06, 0.895,
             f"MATH-500, first {n} problems, greedy (temperature 0), reasoning off",
             fontsize=9, color=INK_2)

    panels = (
        (ax1, f"pass@1 (N={n})", "accuracy_pct", 100, "{:.0f}%"),
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
    ax.set_xlim(0, 27.5)
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
    ax.set_title("Identical in 6 of 7 subjects",
                 color=INK, fontsize=13, fontweight="bold", loc="left", pad=16)
    ax.text(0, 1.02, "MATH-500 N=100 — subject totals in parentheses",
            transform=ax.transAxes, fontsize=9, color=INK_2)
    ax.legend(handles=[
        plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=GRAY, label="base"),
        plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=BLUE, label="DFlash"),
    ], loc="lower right", frameon=False, fontsize=10, labelcolor=INK_2)

    fig.tight_layout()
    fig.savefig(ASSETS / "accuracy_by_subject.png", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    tables = load_tables(CSV_FILE)
    for tid in ("1", "2", "2B"):
        if tid not in tables or not tables[tid]:
            raise SystemExit(f"results_summary.csv: TABLE {tid} missing or empty")
    ASSETS.mkdir(exist_ok=True)
    plot_speed_vs_context(tables)
    plot_accuracy_vs_speed(tables)
    plot_accuracy_by_subject(tables)
    for name in ("speed_vs_context", "accuracy_vs_speed", "accuracy_by_subject"):
        print(f"wrote assets/{name}.png")


if __name__ == "__main__":
    main()
