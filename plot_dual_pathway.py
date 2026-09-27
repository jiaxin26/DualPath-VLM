#!/usr/bin/env python3
"""
Figure: Dual-Pathway Discovery Across 5 Models
================================================
Visualizes the component-level activation patching results from Table 2,
showing the dual-pathway organization shared across all five VLMs.

Layout: 5 columns of small multiples (one per model)
  x-axis: normalized depth ∈ [0, 1]
  y-axis: Cohen's d effect size
  blue circle/square : grounding pathway (d > 0)
  red circle/square  : hallucination pathway (d < 0)
  ○ MLP, □ Attn

Auto-loads from results/<model>_component/circuit.json if available;
otherwise falls back to the published Table 2 highlights.

Usage:
  python plot_dual_pathway.py
  python plot_dual_pathway.py --results results/
  python plot_dual_pathway.py --output figures/fig2_dual_pathway.pdf
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ── Fallback data: Table 2 highlights (5 components per model) ───
# Used when circuit.json files are not available.
# Each entry: (depth_pct, cohen_d, comp_type)

TABLE2_FALLBACK = {
    "Qwen3-VL-8B": {
        "n_layers": 36,
        "n_grnd_total": 14,
        "n_halluc_total": 12,
        "components": [
            (0,   +0.94, "mlp"),    # L0.mlp
            (23,  +0.48, "mlp"),    # L8.mlp
            (60,  +0.78, "attn"),   # L21.attn
            (14,  -0.51, "mlp"),    # L5.mlp
            (100, -0.63, "mlp"),    # L35.mlp
        ],
    },
    "LLaVA-v1.6-7B": {
        "n_layers": 32,
        "n_grnd_total": 18,
        "n_halluc_total": 30,
        "components": [
            (97,  +0.42, "mlp"),    # L30.mlp
            (94,  +0.52, "mlp"),    # L29.mlp
            (100, -0.63, "attn"),   # L31.attn
            (77,  -0.58, "mlp"),    # L24.mlp
            (19,  -0.59, "mlp"),    # L6.mlp
        ],
    },
    "Llama-3.2-V-11B": {
        "n_layers": 40,
        "n_grnd_total": 11,
        "n_halluc_total": 20,
        "components": [
            (8,   +0.91, "attn"),   # L3.attn (cross-attn!)
            (3,   +0.86, "mlp"),    # L1.mlp
            (28,  +0.69, "mlp"),    # L11.mlp
            (41,  -0.89, "mlp"),    # L16.mlp
            (18,  -0.77, "mlp"),    # L7.mlp
        ],
    },
    "InternVL3-8B": {
        "n_layers": 28,
        "n_grnd_total": 21,
        "n_halluc_total": 6,
        "components": [
            (56,  +0.57, "attn"),   # L15.attn
            (70,  +1.22, "mlp"),    # L19.mlp
            (74,  +0.66, "attn"),   # L20.attn
            (100, -0.87, "mlp"),    # L27.mlp
            (33,  -0.44, "mlp"),    # L9.mlp
        ],
    },
    "InternVL3-14B": {
        "n_layers": 48,
        "n_grnd_total": 29,
        "n_halluc_total": 16,
        "components": [
            (0,   +0.61, "mlp"),    # L0.mlp
            (19,  +0.54, "mlp"),    # L9.mlp
            (4,   -0.73, "attn"),   # L2.attn
            (9,   -0.58, "mlp"),    # L4.mlp
            (32,  -0.46, "attn"),   # L15.attn
        ],
    },
}

EXPLICIT_PATHS = {
    "Qwen3-VL-8B":     "qwen3vl_v2/circuit.json",
    "LLaVA-v1.6-7B":   "llava_v2/circuit.json",
    "Llama-3.2-V-11B": "llama32v_v2/circuit.json",
    "InternVL3-8B":    "internvl3_8b_v2/circuit.json",
    "InternVL3-14B":   "internvl3_14b_v2/circuit.json",
}

# Map directory name candidates → display name
RESULTS_DIR_MAP = {
    "qwen": "Qwen3-VL-8B",
    "qwen3vl_component": "Qwen3-VL-8B",
    "qwen_component": "Qwen3-VL-8B",
    "llava": "LLaVA-v1.6-7B",
    "llava_component": "LLaVA-v1.6-7B",
    "internvl8b": "InternVL3-8B",
    "internvl3_8b": "InternVL3-8B",
    "internvl_component": "InternVL3-8B",
    "internvl3_8b_v2": "InternVL3-8B",
    "llama": "Llama-3.2-V-11B",
    "llama_component": "Llama-3.2-V-11B",
    "llama32v_v2": "Llama-3.2-V-11B",
    "internvl14b": "InternVL3-14B",
    "internvl3_14b": "InternVL3-14B",
    "internvl3_14b_v2": "InternVL3-14B",
}

# Display order
MODEL_ORDER = [
    "Qwen3-VL-8B",
    "LLaVA-v1.6-7B",
    "Llama-3.2-V-11B",
    "InternVL3-8B",
    "InternVL3-14B",
]


def load_from_circuit_files(results_dir):
    found = {}
    for display, rel in EXPLICIT_PATHS.items():
        path = os.path.join(results_dir, rel)
        if not os.path.isfile(path):
            print(f"  [skip] {display}: {path} not found")
            continue
        with open(path) as f:
            data = json.load(f)
        comps = []
        for c in data.get("components", []):
            if c.get("type") not in ("attn", "mlp"):
                continue
            d, layer = c.get("cohen_d"), c.get("layer")
            if d is None or layer is None:
                continue
            comps.append((layer, float(d), c["type"]))
        if not comps:
            continue
        n_layers = data.get("n_layers") or (max(c[0] for c in comps) + 1)
        comps_norm = [(l / max(n_layers - 1, 1) * 100, d, t) for l, d, t in comps]
        found[display] = {
            "n_layers": n_layers,
            "n_grnd_total":   sum(1 for _, d, _ in comps_norm if d > 0),
            "n_halluc_total": sum(1 for _, d, _ in comps_norm if d < 0),
            "components": comps_norm,
        }
        print(f"  Loaded {len(comps_norm)} components for {display} from {path}")
    return found


def make_plot(output, results_dir=None):
    """Generate the dual-pathway figure."""

    # Load real data if available; fall back to Table 2 highlights
    data = dict(TABLE2_FALLBACK)
    if results_dir:
        loaded = load_from_circuit_files(results_dir)
        for name, d in loaded.items():
            data[name] = d
        if not loaded:
            print("  No circuit.json files found — using Table 2 fallback data")
    else:
        print("  Using Table 2 fallback data (pass --results to load real circuits)")

    # Style
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.linewidth": 0.8,
    })

    # Colors
    # GRND_COLOR = "#1f77b4"   # blue
    # HALL_COLOR = "#d62728"   # red

    # GRND_COLOR = "#2B6CB0"   # blue
    # HALL_COLOR = "#C0563F"   # red

    # GRND_COLOR = "#4F8DFF"   # blue
    # HALL_COLOR = "#F28C28"   # red
    GRND_COLOR = "#3BABE8"   # blue
    HALL_COLOR = "#C90000"   # red

    # 5 subplots side by side
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.8),
                              sharey=True, gridspec_kw={"wspace": 0.12})

    for ax, name in zip(axes, MODEL_ORDER):
        if name not in data:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title(name, fontsize=10)
            continue

        info = data[name]
        components = info["components"]

        # Separate by pathway and type for legend
        grnd_mlp  = [(d, c) for d, c, t in components if c > 0 and t == "mlp"]
        grnd_attn = [(d, c) for d, c, t in components if c > 0 and t == "attn"]
        hall_mlp  = [(d, c) for d, c, t in components if c < 0 and t == "mlp"]
        hall_attn = [(d, c) for d, c, t in components if c < 0 and t == "attn"]

        # Zero line
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="-", alpha=0.5,
                   zorder=1)

        # Light shading: top half = grounding region, bottom half = halluc region
        ax.axhspan(0, 2, alpha=0.04, color=GRND_COLOR, zorder=0)
        ax.axhspan(-2, 0, alpha=0.04, color=HALL_COLOR, zorder=0)

        # Plot points
        marker_size = 70

        if grnd_mlp:
            xs, ys = zip(*grnd_mlp)
            ax.scatter(xs, ys, s=marker_size, marker="o",
                       color=GRND_COLOR, edgecolors="black", linewidths=0.6,
                       alpha=0.85, zorder=5)
        if grnd_attn:
            xs, ys = zip(*grnd_attn)
            ax.scatter(xs, ys, s=marker_size, marker="s",
                       color=GRND_COLOR, edgecolors="black", linewidths=0.6,
                       alpha=0.85, zorder=5)
        if hall_mlp:
            xs, ys = zip(*hall_mlp)
            ax.scatter(xs, ys, s=marker_size, marker="o",
                       color=HALL_COLOR, edgecolors="black", linewidths=0.6,
                       alpha=0.85, zorder=5)
        if hall_attn:
            xs, ys = zip(*hall_attn)
            ax.scatter(xs, ys, s=marker_size, marker="s",
                       color=HALL_COLOR, edgecolors="black", linewidths=0.6,
                       alpha=0.85, zorder=5)

        # Title with model name + counts
        n_g = info["n_grnd_total"]
        n_h = info["n_halluc_total"]
        ax.set_title(f"{name}\n"
                     f"$n_\\mathrm{{grnd}}={n_g}$, $n_\\mathrm{{hall}}={n_h}$",
                     fontsize=11)

        # X-axis
        ax.set_xlim(-5, 105)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=10)
        ax.set_xlabel("Normalized depth (%)", fontsize=12)

        # Y-axis (only on leftmost)
        ax.set_ylim(-1.8, 1.8)
        ax.set_yticks([-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5])
        ax.tick_params(axis="y", labelsize=10)

        ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)

    axes[0].set_ylabel(r"Effect size $d$", fontsize=12)

    # # ── Annotations: top-left and bottom-right show pathway labels ──
    # axes[0].text(0.03, 0.93, "grounding", transform=axes[0].transAxes,
    #              fontsize=10, color=GRND_COLOR, fontweight="bold",
    #              ha="left", va="top")
    # axes[0].text(0.03, 0.07, "hallucination", transform=axes[0].transAxes,
    #              fontsize=10, color=HALL_COLOR, fontweight="bold",
    #              ha="left", va="bottom")

    # ── Shared legend (above titles) ──
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=GRND_COLOR, markeredgecolor="black",
               markersize=9, label="Grounding (MLP)", linewidth=0),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=GRND_COLOR, markeredgecolor="black",
               markersize=9, label="Grounding (Attn)", linewidth=0),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=HALL_COLOR, markeredgecolor="black",
               markersize=9, label="Hallucination (MLP)", linewidth=0),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=HALL_COLOR, markeredgecolor="black",
               markersize=9, label="Hallucination (Attn)", linewidth=0),
    ]
    fig.legend(handles=legend_elements,
               loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=4, fontsize=11, frameon=True, framealpha=0.95,
               edgecolor="lightgray")

    # Reserve room: top for legend, sides minimal, bottom for x-labels
    plt.subplots_adjust(top=0.78, bottom=0.18, left=0.05, right=0.99,
                        wspace=0.12)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"  Saved figure to {output}")

    if output.endswith(".pdf"):
        png_path = output.replace(".pdf", ".png")
        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        print(f"  Saved preview to {png_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Dual-pathway discovery figure")
    parser.add_argument("--output", type=str, default="figures/fig2_dual_pathway.pdf",
                        help="Output figure path")
    parser.add_argument("--results", type=str, default=None,
                        help="Auto-load components from circuit.json files")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    make_plot(args.output, results_dir=args.results)


if __name__ == "__main__":
    main()
