#!/usr/bin/env python3
"""
Figure 4: Pareto plot of intervention effectiveness
====================================================
Plots accuracy cost vs. hallucination reduction across models and configurations.

X-axis: ΔAccuracy (percentage points)
Y-axis: Relative hallucination reduction (%)

Each model contributes:
  - One filled marker = optimal configuration (from Table 6)
  - LLaVA additionally shows hollow markers = alternative top-k values (Table 7)

Usage:
  python plot_pareto.py                                # default output
  python plot_pareto.py --output figures/fig4.pdf     # specify path
  python plot_pareto.py --results results/            # auto-load from results
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ── Data from Table 6 (Intervention results, POPE-adversarial) ───
# Format: model_name → (ΔAcc_pp, RelReduction_pct, marker, color)

OPTIMAL = {
    "Qwen3-VL-8B":     (-1.0, 52, "o", "#1f77b4"),
    "LLaVA-v1.6-7B":   (+0.0, 40, "*", "#d62728"),  # top-10 selection
    "Llama-3.2-V-11B": (-0.2, 76, "o", "#9467bd"),
    "InternVL3-8B":    (-2.0, 64, "o", "#708238"),
    "InternVL3-14B":   (-0.3, 42, "o", "#ff7f0e"),
}

# LLaVA top-k from Table 10 (s=0.5, 400-sample held-out)
# (k, ΔAcc_pp, RelReduction_pct)
LLAVA_TOPK = [
    (8,  +0.5,  40),
    (10, +0.0,  40),  # selected — drawn as filled star via OPTIMAL
    (15, +0.3,  20),
    (20, -4.5,  30),
    (30, -9.0,  70),
]


def load_topk_from_results(results_dir):
    """Try loading actual LLaVA top-k data from result JSON files."""
    candidates = [
        os.path.join(results_dir, "llava_topk", "topk_ablation_results.json"),
        os.path.join(results_dir, "intervention_llava_topk", "topk_ablation_results.json"),
        os.path.join(results_dir, "llava_intervention", "topk_ablation_results.json"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        for bench, bdata in data.items():
            if not isinstance(bdata, dict) or "baseline" not in bdata:
                continue
            bl = bdata["baseline"]
            ba, bh = bl.get("accuracy", 0), bl.get("halluc_rate", 0)
            topk_data = []
            for k_str, v in bdata.items():
                if not k_str.startswith("top_"):
                    continue
                k = int(k_str.split("_")[1])
                da_pp = (v["accuracy"] - ba) * 100
                dh_pp = (v["halluc_rate"] - bh) * 100
                rel = -(dh_pp / (bh * 100)) * 100 if bh > 0 else 0
                topk_data.append((k, da_pp, rel))
            if topk_data:
                print(f"  Loaded LLaVA top-k from {path}")
                return sorted(topk_data)
    return None


def make_plot(output, results_dir=None):
    topk_data = LLAVA_TOPK
    if results_dir:
        loaded = load_topk_from_results(results_dir)
        if loaded:
            topk_data = loaded

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.linewidth": 0.8,
    })

    fig, ax = plt.subplots(figsize=(6.5, 4))

    # ── Shaded "acceptable accuracy cost" region (≤2pp drop) ──
    ax.axvspan(-2, 100, alpha=0.07, color="#708238", zorder=0)
    ax.axvline(-2, color="#708238", linewidth=0.8, linestyle=":",
               alpha=0.7, zorder=1)
    ax.axvline(0, color="gray", linewidth=0.6, linestyle="--",
               alpha=0.6, zorder=1)

    # ── LLaVA top-k alternatives (hollow stars) ──
    llava_color = OPTIMAL["LLaVA-v1.6-7B"][3]
    for k, da, rel in topk_data:
        if k == 10:  # selected config — drawn as filled star
            continue
        ax.scatter(da, rel, s=85, marker="*",
                   facecolors="none", edgecolors=llava_color,
                   linewidths=1.0, alpha=0.6, zorder=3)
        # Label k value beside each candidate
        label_offsets = {
            8:  (0.25,  0.0),
            15: (0.25,  0.0),
            20: (0.25,  0.0),
            30: (0.25,  0.0),
        }
        dx, dy = label_offsets.get(k, (0.20, 0))
        ax.annotate(f"$k$={k}", (da, rel),
                    xytext=(da + dx, rel + dy),
                    fontsize=7.5, color=llava_color, alpha=0.75,
                    va="center")

    # ── Optimal configurations (filled markers) ──
    for name, (da, rel, marker, color) in OPTIMAL.items():
        size = 340 if marker == "*" else 170
        ax.scatter(da, rel, s=size, marker=marker,
                   color=color, edgecolors="black", linewidths=1.0,
                   zorder=10)

        offsets = {
            "Qwen3-VL-8B":     (0.30,  0),
            "LLaVA-v1.6-7B":   (0.40, -5),
            "Llama-3.2-V-11B": (0.30,  0),
            "InternVL3-8B":    (0.30,  0),
            "InternVL3-14B":   (0.30,  4),
        }
        ha_align = {}
        dx, dy = offsets.get(name, (0.30, 0))
        ha = ha_align.get(name, "left")
        ax.annotate(name, (da, rel),
                    xytext=(da + dx, rel + dy),
                    fontsize=10, fontweight="bold",
                    ha=ha, va="center",
                    color=color, zorder=11)

    # # ── Budget region label ──
    # ax.text(2.7, 5, r"$\Delta$Acc $\geq -2$ pp",
    #         fontsize=8.5, color="#2a7a2a",
    #         ha="right", va="bottom", style="italic")

    # ── Axes ──
    ax.set_xlabel(r"Accuracy change ($\Delta$Acc)",
                fontsize=11)
    ax.set_ylabel("Hallucination reduction", fontsize=11)
    ax.set_xlim(-11, 3)
    ax.set_ylim(0, 90)
    ax.set_xticks([-10, -8, -6, -4, -2, 0, 2])
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="gray", markeredgecolor="black",
               markersize=11, label="Uniform scaling", linewidth=0),
        Line2D([0], [0], marker="*", color="w",
               markerfacecolor=OPTIMAL["LLaVA-v1.6-7B"][3],
               markeredgecolor="black",
               markersize=14, label="LLaVA top-$k$", linewidth=0),
        Line2D([0], [0], marker="*", color="w",
               markerfacecolor="none",
               markeredgecolor=OPTIMAL["LLaVA-v1.6-7B"][3],
               markersize=12, label="LLaVA top-$k$ alternatives", linewidth=0),
    ]
    ax.legend(handles=legend_elements, loc="lower left",
              fontsize=9, frameon=True, framealpha=0.95,
              edgecolor="lightgray")

    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"  Saved figure to {output}")

    if output.endswith(".pdf"):
        png_path = output.replace(".pdf", ".png")
        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        print(f"  Saved preview to {png_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot intervention Pareto frontier")
    parser.add_argument("--output", type=str, default="figures/fig4_pareto.pdf",
                        help="Output figure path (PDF or PNG)")
    parser.add_argument("--results", type=str, default=None,
                        help="Optional: load LLaVA top-k from results dir")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    make_plot(args.output, results_dir=args.results)


if __name__ == "__main__":
    main()