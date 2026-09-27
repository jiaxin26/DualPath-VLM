"""
Summarize Multi-Benchmark Intervention Results
================================================

Aggregates results from run_multi_benchmark.sh into a single
summary table suitable for the paper.

Outputs:
  1. JSON summary (machine-readable)
  2. LaTeX table (copy-paste into paper)
  3. Console table (for quick review)

Usage:
  PYTHONPATH=. python experiments/summarize_multi_benchmark.py \
    --results-dir results/multi_benchmark/ \
    --output results/multi_benchmark/summary_table.json
"""

import argparse
import json
import os
from pathlib import Path


MODEL_DISPLAY = {
    "qwen": "Qwen3-VL-8B",
    "llava": "LLaVA-v1.6-7B",
    "internvl8b": "InternVL3-8B",
    "llama": "Llama-3.2-V-11B",
    "internvl14b": "InternVL3-14B",
}

BENCH_DISPLAY = {
    "pope_adversarial": "POPE-adv",
    "pope_popular": "POPE-pop",
    "pope_random": "POPE-rand",
    "amber_all": "AMBER-all",
    "amber_existence": "AMBER-exist",
    "amber_attribute": "AMBER-attr",
    "amber_relation": "AMBER-rel",
}


def load_results(results_dir: str) -> dict:
    """Walk results directory and collect all intervention results."""
    summary = {}

    for model_key in MODEL_DISPLAY:
        model_dir = os.path.join(results_dir, model_key)
        if not os.path.isdir(model_dir):
            continue

        summary[model_key] = {}

        for bench_dir in sorted(Path(model_dir).iterdir()):
            if not bench_dir.is_dir():
                continue

            bench_name = bench_dir.name
            result_file = bench_dir / "intervention_results.json"
            topk_file = bench_dir / "topk_ablation_results.json"

            if result_file.exists():
                with open(result_file) as f:
                    data = json.load(f)
                # Extract the first benchmark entry
                for bname, bdata in data.items():
                    baseline = bdata.get("baseline", {})
                    # Find the targeted result (first one with "targeted" in key)
                    targeted = None
                    for k, v in bdata.items():
                        if "targeted" in k:
                            targeted = v
                            break
                    summary[model_key][bench_name] = {
                        "baseline_acc": baseline.get("accuracy", 0),
                        "baseline_halluc": baseline.get("halluc_rate", 0),
                        "targeted_acc": targeted.get("accuracy", 0) if targeted else 0,
                        "targeted_halluc": targeted.get("halluc_rate", 0) if targeted else 0,
                    }

            elif topk_file.exists():
                with open(topk_file) as f:
                    data = json.load(f)
                for bname, bdata in data.items():
                    baseline = bdata.get("baseline", {})
                    # Find best top-k result (lowest halluc_rate with acc >= baseline - 2%)
                    best = None
                    for k, v in bdata.items():
                        if k.startswith("top_"):
                            if best is None or v.get("halluc_rate", 1) < best.get("halluc_rate", 1):
                                acc_drop = baseline.get("accuracy", 0) - v.get("accuracy", 0)
                                if acc_drop <= 0.02:
                                    best = v
                    if best is None:
                        # fallback to top_3
                        best = bdata.get("top_3", bdata.get("baseline", {}))
                    summary[model_key][bench_name] = {
                        "baseline_acc": baseline.get("accuracy", 0),
                        "baseline_halluc": baseline.get("halluc_rate", 0),
                        "targeted_acc": best.get("accuracy", 0),
                        "targeted_halluc": best.get("halluc_rate", 0),
                    }

    return summary


def print_console_table(summary: dict):
    """Print a readable console table."""
    print()
    print("=" * 100)
    print("MULTI-BENCHMARK INTERVENTION RESULTS")
    print("=" * 100)

    all_benches = set()
    for model_data in summary.values():
        all_benches.update(model_data.keys())
    benches = sorted(all_benches)

    for bench in benches:
        display = BENCH_DISPLAY.get(bench, bench)
        print(f"\n  {display}:")
        print(f"    {'Model':<20s} {'Base Acc':>10s} {'Base Hall':>10s} "
              f"{'Int Acc':>10s} {'Int Hall':>10s} "
              f"{'ΔAcc':>8s} {'ΔHall':>8s} {'Rel↓':>8s}")
        print("    " + "─" * 86)

        for model_key in MODEL_DISPLAY:
            if model_key not in summary or bench not in summary[model_key]:
                continue
            r = summary[model_key][bench]
            ba = r["baseline_acc"]
            bh = r["baseline_halluc"]
            ta = r["targeted_acc"]
            th = r["targeted_halluc"]
            da = ta - ba
            dh = th - bh
            rel = (dh / bh * 100) if bh > 0 else 0

            name = MODEL_DISPLAY[model_key]
            print(f"    {name:<20s} {ba:>10.1%} {bh:>10.1%} "
                  f"{ta:>10.1%} {th:>10.1%} "
                  f"{da:>+8.1%} {dh:>+8.1%} {rel:>7.0f}%")


def generate_latex_table(summary: dict) -> str:
    """Generate a LaTeX table for the paper."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Multi-benchmark intervention results. "
                 r"Hallucination reduction (\%) and accuracy change "
                 r"across POPE splits and AMBER hallucination types.}")
    lines.append(r"\label{tab:multi-benchmark}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{ll" + "rr" * 3 + "}")
    lines.append(r"\toprule")
    lines.append(r"& & \multicolumn{2}{c}{POPE-adv} "
                 r"& \multicolumn{2}{c}{AMBER-exist} "
                 r"& \multicolumn{2}{c}{AMBER-attr} \\")
    lines.append(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}")
    lines.append(r"Model & Config & Acc. & Hall. & Acc. & Hall. & Acc. & Hall. \\")
    lines.append(r"\midrule")

    bench_keys = ["pope_adversarial", "amber_existence", "amber_attribute"]

    for model_key in MODEL_DISPLAY:
        if model_key not in summary:
            continue
        name = MODEL_DISPLAY[model_key]

        # Baseline row
        cells = [f"{name}", "Baseline"]
        for bk in bench_keys:
            r = summary[model_key].get(bk, {})
            ba = r.get("baseline_acc", 0)
            bh = r.get("baseline_halluc", 0)
            cells.append(f"{ba:.1%}")
            cells.append(f"{bh:.1%}")
        lines.append(" & ".join(cells) + r" \\")

        # Intervention row
        cells = ["", "Ours"]
        for bk in bench_keys:
            r = summary[model_key].get(bk, {})
            ta = r.get("targeted_acc", 0)
            th = r.get("targeted_halluc", 0)
            cells.append(f"{ta:.1%}")
            cells.append(f"{th:.1%}")
        lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    summary = load_results(args.results_dir)

    if not summary:
        print("No results found. Check --results-dir path.")
        return

    # Console output
    print_console_table(summary)

    # LaTeX table
    latex = generate_latex_table(summary)
    print("\n\nLaTeX Table:")
    print(latex)

    # Save
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved JSON summary to {args.output}")

        latex_path = args.output.replace(".json", ".tex")
        with open(latex_path, "w") as f:
            f.write(latex)
        print(f"Saved LaTeX table to {latex_path}")


if __name__ == "__main__":
    main()
