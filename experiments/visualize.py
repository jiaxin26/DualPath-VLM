"""
Visualization: Patching Effect Heatmaps
========================================

Generates publication-quality figures from patching results.

Usage:
  PYTHONPATH=. python experiments/visualize.py \
    --results results/qwen3vl_head/ \
    --output figures/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_component_heatmap(comp_correct: np.ndarray,
                           comp_halluc: np.ndarray,
                           model_name: str,
                           save_path: str):
    """Side-by-side heatmaps: component effects for correct vs hallucination.

    comp_correct, comp_halluc: (n_layers, 2) where [:, 0]=attn, [:, 1]=mlp
    """
    n_layers = comp_correct.shape[0]
    diff = comp_correct - comp_halluc

    fig, axes = plt.subplots(1, 3, figsize=(14, max(6, n_layers * 0.25)),
                              gridspec_kw={"width_ratios": [1, 1, 1]})

    vmax = max(abs(comp_correct).max(), abs(comp_halluc).max(), 1e-6)
    dmax = max(abs(diff).max(), 1e-6)

    labels = ["Attn", "MLP"]
    layer_labels = [str(i) for i in range(n_layers)]

    for ax_idx, (data, title, vm) in enumerate([
        (comp_correct, "Correct samples", vmax),
        (comp_halluc, "Hallucination samples", vmax),
        (diff, "Differential (correct − halluc)", dmax),
    ]):
        ax = axes[ax_idx]
        cmap = "RdBu_r" if ax_idx == 2 else "YlOrRd"
        vmin = -vm if ax_idx == 2 else 0
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vm,
                       interpolation="nearest")
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Layer" if ax_idx == 0 else "", fontsize=10)
        ax.set_yticks(range(0, n_layers, max(1, n_layers // 15)))
        ax.set_yticklabels(
            [layer_labels[i] for i in range(0, n_layers, max(1, n_layers // 15))],
            fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Component-Level Patching Effects — {model_name}",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_head_heatmap(head_correct: np.ndarray,
                      head_halluc: np.ndarray,
                      model_name: str,
                      save_path: str):
    """Head-level heatmaps: (n_layers, n_heads)."""
    n_layers, n_heads = head_correct.shape
    diff = head_correct - head_halluc

    fig, axes = plt.subplots(1, 3, figsize=(20, max(6, n_layers * 0.3)),
                              gridspec_kw={"width_ratios": [1, 1, 1]})

    vmax = max(abs(head_correct).max(), abs(head_halluc).max(), 1e-6)
    dmax = max(abs(diff).max(), 1e-6)

    for ax_idx, (data, title, vm) in enumerate([
        (head_correct, "Correct samples", vmax),
        (head_halluc, "Hallucination samples", vmax),
        (diff, "Differential (correct − halluc)", dmax),
    ]):
        ax = axes[ax_idx]
        cmap = "RdBu_r" if ax_idx == 2 else "YlOrRd"
        vmin = -vm if ax_idx == 2 else 0
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vm,
                       interpolation="nearest")
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlabel("Head", fontsize=10)
        ax.set_ylabel("Layer" if ax_idx == 0 else "", fontsize=10)
        ax.set_xticks(range(0, n_heads, max(1, n_heads // 8)))
        ax.set_yticks(range(0, n_layers, max(1, n_layers // 15)))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Head-Level Patching Effects — {model_name}",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_layer_profile_comparison(circuits: list, save_path: str):
    """Overlay depth-normalized layer profiles across architectures."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#534AB7", "#D85A30", "#1D9E75", "#D4537E"]

    for i, (name, profile) in enumerate(circuits):
        n = len(profile)
        x = np.linspace(0, 1, n)
        c = colors[i % len(colors)]
        ax.plot(x, profile, label=name, linewidth=2, color=c)
        ax.fill_between(x, 0, profile, alpha=0.1, color=c)

    ax.set_xlabel("Relative depth (0 = first layer, 1 = last)", fontsize=11)
    ax.set_ylabel("Aggregate |differential effect|", fontsize=11)
    ax.set_title("Layer Profile Comparison Across Architectures", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def visualize_results(result_dir: str, output_dir: str):
    """Generate all figures from a single experiment's results."""
    os.makedirs(output_dir, exist_ok=True)

    # Load summary
    summary_path = os.path.join(result_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        model_name = summary.get("model", "Unknown").split("/")[-1]
    else:
        model_name = Path(result_dir).name

    # Component heatmaps
    comp_c = _try_load(os.path.join(result_dir, "comp_correct.npy"))
    comp_h = _try_load(os.path.join(result_dir, "comp_halluc.npy"))
    if comp_c is not None and comp_h is not None:
        plot_component_heatmap(
            comp_c, comp_h, model_name,
            os.path.join(output_dir, "component_heatmap.png"))

    # Head heatmaps
    head_c = _try_load(os.path.join(result_dir, "head_correct.npy"))
    head_h = _try_load(os.path.join(result_dir, "head_halluc.npy"))
    if head_c is not None and head_h is not None:
        plot_head_heatmap(
            head_c, head_h, model_name,
            os.path.join(output_dir, "head_heatmap.png"))


def _try_load(path):
    return np.load(path) if os.path.exists(path) else None


def main():
    parser = argparse.ArgumentParser(description="Visualize patching results")
    parser.add_argument("--results", nargs="+", required=True,
                        help="Result directories to visualize")
    parser.add_argument("--output", type=str, default="figures/")
    args = parser.parse_args()

    for rd in args.results:
        name = Path(rd).name
        out = os.path.join(args.output, name)
        visualize_results(rd, out)

    # If multiple results, do profile comparison
    if len(args.results) > 1:
        circuits = []
        for rd in args.results:
            cpath = os.path.join(rd, "circuit.json")
            if os.path.exists(cpath):
                with open(cpath) as f:
                    data = json.load(f)
                comp_c = _try_load(os.path.join(rd, "comp_correct.npy"))
                comp_h = _try_load(os.path.join(rd, "comp_halluc.npy"))
                if comp_c is not None and comp_h is not None:
                    diff = comp_c - comp_h
                    profile = np.abs(diff).sum(axis=1)
                    circuits.append((data["model_name"], profile))

        if len(circuits) >= 2:
            plot_layer_profile_comparison(
                circuits, os.path.join(args.output, "layer_profile_comparison.png"))


if __name__ == "__main__":
    main()
