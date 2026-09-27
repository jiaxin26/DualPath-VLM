#!/usr/bin/env python3
"""
Figure 5: Cross-benchmark generalization heatmap
==================================================
Visualizes Table 8: relative hallucination reduction across 5 models × 6 benchmarks.

Layout: 5 rows (models) × 6 columns (POPE-adv/pop/rand + AMBER-exist/attr/rel)
  Cell values = relative hallucination reduction (positive = improvement)
  Diverging colormap: green = improvement, gray = negligible, red = worsening
  Column groups labeled POPE / AMBER above the heatmap

Auto-loads from results/multi_benchmark/summary_table.json if available.

Usage:
  python plot_cross_benchmark.py
  python plot_cross_benchmark.py --output figures/fig5_cross_benchmark.pdf
  python plot_cross_benchmark.py --results results/
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle


# ── Data from Table 8 ───────────────────────────────────────────
# Convention: positive = improvement (hallucination reduced)
#             negative = worsening (hallucination increased)
#             0        = negligible change (|ΔH| < 1pp)

MODELS = [
    "Qwen3-VL-8B",
    "LLaVA-v1.6-7B",
    "Llama-3.2-V-11B",
    "InternVL3-8B",
    "InternVL3-14B",
]

BENCHMARKS = [
    "POPE-adv",
    "POPE-pop",
    "POPE-rand",
    "AMBER-exist",
    "AMBER-attr",
    "AMBER-rel",
]

# 5 × 6 matrix of relative hallucination reduction (%)
# Sign convention: positive = improvement, negative = worsening
TABLE8_FALLBACK = np.array([
    # POPE-adv  POPE-pop  POPE-rand  AMBER-ex  AMBER-attr  AMBER-rel
    [   52,        53,       33,         9,         0,         34],   # Qwen3-VL-8B
    [   27,        22,        0,        14,         0,          7],   # LLaVA
    [   64,        65,       60,        15,        17,         39],   # InternVL3-8B
    [   76,        69,       80,        38,        27,         47],   # Llama-3.2-V-11B
    [   32,        48,       25,       -15,         0,        -19],   # InternVL3-14B
])


def load_from_summary(results_dir):
    """Try to load from results/multi_benchmark/summary_table.json."""
    path = os.path.join(results_dir, "multi_benchmark", "summary_table.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return None

    # Map summary keys to display names
    key_map = {
        "qwen": "Qwen3-VL-8B",
        "llava": "LLaVA-v1.6-7B",
        "llama": "Llama-3.2-V-11B",
        "internvl8b": "InternVL3-8B",
        "internvl14b": "InternVL3-14B",
    }
    bench_map = {
        "pope_adversarial": "POPE-adv",
        "pope_popular": "POPE-pop",
        "pope_random": "POPE-rand",
        "amber_existence": "AMBER-exist",
        "amber_attribute": "AMBER-attr",
        "amber_relation": "AMBER-rel",
    }

    matrix = np.full((len(MODELS), len(BENCHMARKS)), np.nan)
    for model_key, model_data in data.items():
        if model_key not in key_map:
            continue
        display = key_map[model_key]
        if display not in MODELS:
            continue
        i = MODELS.index(display)
        for bench_key, r in model_data.items():
            if bench_key not in bench_map:
                continue
            j = BENCHMARKS.index(bench_map[bench_key])
            bh = r.get("baseline_halluc", 0)
            th = r.get("targeted_halluc", 0)

            # Absolute change in percentage points
            delta_pp = (bh - th) * 100

            # Floor: if absolute change < 1pp, treat as negligible (leave as NaN → "—")
            # This prevents small-denominator artifacts where e.g. 0.8% → 1.2%
            # computes to a misleading -50% relative change.
            if abs(delta_pp) < 0.5:
                continue

            if bh > 0:
                # Relative reduction: positive = improvement
                rel = (bh - th) / bh * 100
                matrix[i, j] = rel

    if np.isnan(matrix).all():
        return None
    print(f"  Loaded results from {path}")
    return matrix


def make_plot(output, results_dir=None):
    """Generate the cross-benchmark heatmap."""

    # Load real data if available
    matrix = TABLE8_FALLBACK.astype(float)
    if results_dir:
        loaded = load_from_summary(results_dir)
        if loaded is not None:
            matrix = loaded
        else:
            print("  No summary file found — using Table 8 fallback data")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
    })

    # ── Diverging colormap: red (bad) → white (zero) → green (good) ──
    # Custom muted palette
    colors_neg = ["#8b1a1a", "#c0392b", "#e6a89c"]   # red shades
    color_zero = "#f5f5f5"                            # near-white gray
    colors_pos = ["#c8e6c9", "#5cb85c", "#1a6f3a"]   # green shades

    cmap = LinearSegmentedColormap.from_list(
        "transfer", colors_neg + [color_zero] + colors_pos, N=256)
    cmap.set_bad(color=color_zero)  # NaN cells (negligible change) render as gray

    # Norm: center at 0, asymmetric range to accommodate -25 to +85
    vmin, vmax = -30, 85
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(5.8, 4.2))

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    # ── Cell annotations ──
    for i in range(len(MODELS)):
        for j in range(len(BENCHMARKS)):
            v = matrix[i, j]
            if np.isnan(v) or abs(v) < 1:
                text = "—"
                color = "#888"
            else:
                text = f"{int(round(v)):+d}%"
                # White text on dark cells, dark text on light cells
                if v > 50 or v < -15:
                    color = "white"
                else:
                    color = "#222"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    # ── Axis labels ──
    # ax.set_xticks(np.arange(len(BENCHMARKS)))
    # ax.set_xticklabels(BENCHMARKS, rotation=0, fontsize=9)
    xlabels = [
        "POPE\nadv",
        "POPE\npop",
        "POPE\nrand",
        "AMBER\nexist",
        "AMBER\nattr",
        "AMBER\nrel",
    ]

    ax.set_xticks(np.arange(len(BENCHMARKS)))
    ax.set_xticklabels(xlabels, rotation=0, fontsize=8, linespacing=0.9)
    ax.set_yticks(np.arange(len(MODELS)))
    ax.set_yticklabels(MODELS, fontsize=8)

    # Move x-axis ticks to top? No — keep at bottom but add column groups above
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=0, pad=4)

    # ── Group separator (vertical line between POPE and AMBER) ──
    ax.axvline(2.5, color="white", linewidth=2.5, zorder=10)

    # ── Cell borders (thin white lines) ──
    for i in range(len(MODELS) + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.8, zorder=5)
    for j in range(len(BENCHMARKS) + 1):
        ax.axvline(j - 0.5, color="white", linewidth=0.8, zorder=5)

    # # ── Colorbar ──
    # cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    # cbar.set_label("Hallucination reduction (relative, %)", fontsize=9)
    # cbar.ax.tick_params(labelsize=8)
    # cbar.set_ticks([-30, -15, 0, 15, 30, 50, 70, 85])

    # Frame
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Reserve room at bottom for column group headers
    plt.subplots_adjust(left=0.18, right=0.92, top=0.90, bottom=0.22)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"  Saved figure to {output}")

    if output.endswith(".pdf"):
        png_path = output.replace(".pdf", ".png")
        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        print(f"  Saved preview to {png_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Cross-benchmark heatmap")
    parser.add_argument("--output", type=str,
                        default="figures/fig5_cross_benchmark.pdf",
                        help="Output figure path")
    parser.add_argument("--results", type=str, default=None,
                        help="Auto-load from results dir")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    make_plot(args.output, results_dir=args.results)


if __name__ == "__main__":
    main()