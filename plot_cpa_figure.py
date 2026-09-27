import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from pathlib import Path


# ============================================================
# Data from Appendix Table 10
# ============================================================

DATA = {
    "Qwen3-VL": {
        "grounding": {
            "k": 14,
            "correct": -13.89,
            "halluc": +10.03,
            "d_correct": -1.74,
            "d_halluc": +0.89,
            "diff_d": -2.45,
        },
        "halluc_pw": {
            "k": 12,
            "correct": -0.13,
            "halluc": +1.39,
            "d_correct": -0.01,
            "d_halluc": +0.12,
            "diff_d": -0.14,
            "diff_p": 0.385,
        },
    },
    "LLaVA-v1.6": {
        "grounding": {
            "k": 18,
            "correct": -5.49,
            "halluc": +4.11,
            "d_correct": -0.93,
            "d_halluc": +0.68,
            "diff_d": -1.60,
        },
        "halluc_pw": {
            "k": 30,
            "correct": +3.30,
            "halluc": -7.51,
            "d_correct": +0.18,
            "d_halluc": -0.40,
            "diff_d": +0.59,
        },
    },
    "Llama-3.2": {
        "grounding": {
            "k": 11,
            "correct": -1.27,
            "halluc": +0.77,
            "d_correct": -1.09,
            "d_halluc": +0.64,
            "diff_d": -1.72,
        },
        "halluc_pw": {
            "k": 20,
            "correct": +1.03,
            "halluc": -1.03,
            "d_correct": +0.51,
            "d_halluc": -0.48,
            "diff_d": +0.99,
        },
    },
    "InternVL3-8B": {
        "grounding": {
            "k": 21,
            "correct": -11.08,
            "halluc": +3.60,
            "d_correct": -1.33,
            "d_halluc": +0.47,
            "diff_d": -1.83,
        },
        "halluc_pw": {
            "k": 6,
            "correct": +0.86,
            "halluc": +0.34,
            "d_correct": +0.35,
            "d_halluc": +0.12,
            "diff_d": +0.20,
            "diff_p": 0.206,
        },
    },
    "InternVL3-14B": {
        "grounding": {
            "k": 29,
            "correct": -15.56,
            "halluc": +6.43,
            "d_correct": -1.16,
            "d_halluc": +0.84,
            "diff_d": -2.02,
        },
        "halluc_pw": {
            "k": 16,
            "correct": +6.81,
            "halluc": -0.86,
            "d_correct": +0.82,
            "d_halluc": -0.14,
            "diff_d": +1.04,
        },
    },
}


MODEL_ORDER = [
    "Qwen3-VL",
    "LLaVA-v1.6",
    "Llama-3.2",
    "InternVL3-8B",
    "InternVL3-14B",
]


MCOLORS = {
    "Qwen3-VL": "#4C78A8",
    "LLaVA-v1.6": "#E07B54",
    "Llama-3.2": "#A06AAE",
    "InternVL3-8B": "#59A87A",
    "InternVL3-14B": "#C89C32",
}


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _style_axis(ax):
    ax.axvline(0, color="#777777", linewidth=1.25, zorder=1)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#BFBFBF")

    ax.tick_params(axis="y", length=0, colors="#333333")
    ax.tick_params(axis="x", colors="#444444")


def _plot_dumbbell_panel(
    ax,
    pathway_key,
    title,
    title_color,
    show_ylabels=True,
    x_min=-18,
    x_max=12,
):
    n_models = len(MODEL_ORDER)
    y_lookup = {
        model: n_models - 1 - i
        for i, model in enumerate(MODEL_ORDER)
    }

    for model in MODEL_ORDER:
        d = DATA[model][pathway_key]

        x_correct = d["correct"]
        x_halluc = d["halluc"]
        y = y_lookup[model]

        color = MCOLORS[model]
        is_sig = d.get("diff_p", 0.001) < 0.05

        alpha = 0.95 if is_sig else 0.42
        linestyle = "-" if is_sig else "--"
        linewidth = 2.5 if is_sig else 2.0

        # Significant differentials get a directional arrow.
        # n.s. differentials get a plain dashed connector — the direction is
        # statistically meaningless so we don't draw an arrowhead that the
        # eye would over-interpret. Avoids the "color blob" problem at tiny
        # magnitudes where shrunk arrows mush into the endpoint markers.
        if is_sig:
            arrow = FancyArrowPatch(
                (x_correct, y),
                (x_halluc, y),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=linewidth,
                linestyle=linestyle,
                color=color,
                alpha=alpha,
                shrinkA=8,
                shrinkB=8,
                zorder=2,
            )
            ax.add_patch(arrow)
        else:
            ax.plot(
                [x_correct, x_halluc],
                [y, y],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
                solid_capstyle="round",
                dash_capstyle="round",
                zorder=2,
            )

        # Correct: open circle
        ax.scatter(
            x_correct,
            y,
            s=78,
            marker="o",
            facecolor="white",
            edgecolor=color,
            linewidth=2.0,
            alpha=alpha,
            zorder=3,
        )

        # Hallucinating: filled circle
        ax.scatter(
            x_halluc,
            y,
            s=78,
            marker="o",
            facecolor=color,
            edgecolor="white",
            linewidth=0.8,
            alpha=alpha,
            zorder=4,
        )

    ax.set_title(title, color=title_color, pad=8, fontweight="semibold")

    ax.set_xlim(x_min, x_max)
    ax.set_xticks([-15, -10, -5, 0, 5, 10])
    ax.set_ylim(-0.6, n_models - 0.4)

    ax.set_yticks([y_lookup[m] for m in MODEL_ORDER])
    if show_ylabels:
        ax.set_yticklabels(MODEL_ORDER)
    else:
        ax.tick_params(labelleft=False)

    _style_axis(ax)


def plot_cpa_figure(save_path="figures/fig3_cpa_interaction_dumbbell.pdf"):
    x_min, x_max = -18, 12

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.0),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    fig.patch.set_facecolor("white")

    _plot_dumbbell_panel(
        axes[0],
        pathway_key="grounding",
        title="Grounding pathway",
        title_color="#2B6CB0",
        show_ylabels=True,
        x_min=x_min,
        x_max=x_max,
    )

    _plot_dumbbell_panel(
        axes[1],
        pathway_key="halluc_pw",
        title="Hallucination pathway",
        title_color="#C0563F",
        show_ylabels=False,
        x_min=x_min,
        x_max=x_max,
    )

    axes[0].set_xlabel("CPA interaction")
    axes[1].set_xlabel("CPA interaction")

    condition_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="#444444",
            markeredgewidth=2.0,
            markersize=7,
            label="Correct",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#444444",
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=7,
            label="Hallucinating",
        ),
    ]

    style_handles = [
        Line2D(
            [0],
            [0],
            color="#666666",
            linestyle="-",
            linewidth=2.5,
            label="p < 0.05",
        ),
        Line2D(
            [0],
            [0],
            color="#666666",
            linestyle="--",
            linewidth=2.5,
            alpha=0.55,
            label="n.s.",
        ),
    ]

    fig.legend(
        handles=condition_handles + style_handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.06),
        frameon=False,
        handlelength=1.8,
        columnspacing=1.6,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.91])

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for ext in [".pdf", ".png"]:
        out = str(save_path.with_suffix(ext))
        fig.savefig(
            out,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        print(f"Saved: {out}")

    plt.close(fig)


if __name__ == "__main__":
    plot_cpa_figure()