"""
Re-extract circuits from existing batch_results.pkl files.

Usage:
  PYTHONPATH=. python experiments/reextract_circuits.py \
    results/qwen3vl_component_n500/ \
    results/llava_component/ \
    --output results/comparison_fixed/
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
from scipy import stats

from analysis.circuits import extract_circuit, save_circuit, compare_architectures, save_comparison


def load_batch(results_dir):
    pkl_path = os.path.join(results_dir, "batch_results.pkl")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def full_diagnosis(batch, model_name):
    """Print comprehensive diagnosis and return circuit."""
    results = batch.results
    correct = [r for r in results if not r.is_hallucination]
    halluc = [r for r in results if r.is_hallucination]

    print(f"\n{'='*70}")
    print(f"  {model_name}")
    print(f"  {len(correct)} correct, {len(halluc)} halluc")
    print(f"{'='*70}")

    n_layers = results[0].n_layers
    n_heads = results[0].n_heads

    # Use the (now-fixed) normalized_component property
    norm_c = np.array([r.normalized_component for r in correct])
    norm_h = np.array([r.normalized_component for r in halluc])

    mean_c = norm_c.mean(axis=0)
    mean_h = norm_h.mean(axis=0)
    diff = mean_c - mean_h

    # Compute stats for all components
    all_stats = []
    for l in range(n_layers):
        for c_idx, c_name in enumerate(["attn", "mlp"]):
            cv = norm_c[:, l, c_idx]
            hv = norm_h[:, l, c_idx]
            t, p = stats.ttest_ind(cv, hv, equal_var=False)
            pooled = np.sqrt((cv.var(ddof=1) + hv.var(ddof=1)) / 2)
            d = (cv.mean() - hv.mean()) / pooled if pooled > 1e-12 else 0
            all_stats.append({
                "layer": l, "comp": c_name, "c_idx": c_idx,
                "mean_correct": float(cv.mean()),
                "mean_halluc": float(hv.mean()),
                "diff": float(diff[l, c_idx]),
                "cohen_d": float(d),
                "p_value": float(p),
                "rel_depth": l / (n_layers - 1),
            })

    # Sort by |d|
    all_stats.sort(key=lambda x: abs(x["cohen_d"]), reverse=True)

    # Print top 15
    print(f"\n  Top 15 components by |Cohen's d|:")
    print(f"  {'Layer':>6} {'Comp':>5} {'depth':>6} {'correct':>9} {'halluc':>9} "
          f"{'Δ':>9} {'d':>7} {'p':>10}")
    print(f"  {'-'*68}")

    for s in all_stats[:15]:
        sig = "***" if s["p_value"] < 0.001 else "**" if s["p_value"] < 0.01 else "*" if s["p_value"] < 0.05 else ""
        print(f"  L{s['layer']:2d}   {s['comp']:>5} {s['rel_depth']:>6.0%} "
              f"{s['mean_correct']:>+9.4f} {s['mean_halluc']:>+9.4f} "
              f"{s['diff']:>+9.4f} {s['cohen_d']:>+7.3f} {s['p_value']:>9.4f} {sig}")

    # Summarize by layer band
    print(f"\n  Layer band summary (significant components with |d|>0.3):")
    bands = [
        ("Early", 0, n_layers * 0.25),
        ("Mid-early", n_layers * 0.25, n_layers * 0.5),
        ("Mid-late", n_layers * 0.5, n_layers * 0.75),
        ("Late", n_layers * 0.75, n_layers),
    ]
    for name, lo, hi in bands:
        band = [s for s in all_stats if lo <= s["layer"] < hi and abs(s["cohen_d"]) > 0.3]
        pos = [s for s in band if s["cohen_d"] > 0]
        neg = [s for s in band if s["cohen_d"] < 0]
        print(f"    {name:12s} ({int(lo):2d}-{int(hi)-1:2d}): "
              f"{len(band):2d} sig ({len(pos)} pos-d, {len(neg)} neg-d)")

    # Count positive-d vs negative-d among all significant
    sig_all = [s for s in all_stats if s["p_value"] < 0.05]
    sig_pos = [s for s in sig_all if s["cohen_d"] > 0]
    sig_neg = [s for s in sig_all if s["cohen_d"] < 0]
    print(f"\n  Total significant (p<0.05): {len(sig_all)} "
          f"({len(sig_pos)} with d>0, {len(sig_neg)} with d<0)")

    # Extract circuit
    circuit = extract_circuit(batch, model_name)
    return circuit, all_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+")
    parser.add_argument("--output", default="results/comparison_fixed/")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    circuits = []
    for rd in args.result_dirs:
        # Infer model name from summary.json or dir name
        summary_path = os.path.join(rd, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
            model_name = summary.get("model", Path(rd).name).split("/")[-1]
        else:
            model_name = Path(rd).name

        batch = load_batch(rd)
        circuit, all_stats = full_diagnosis(batch, model_name)

        # Save updated circuit
        out_dir = os.path.join(args.output, model_name)
        os.makedirs(out_dir, exist_ok=True)
        save_circuit(circuit, os.path.join(out_dir, "circuit.json"))

        # Save raw stats for plotting
        with open(os.path.join(out_dir, "component_stats.json"), "w") as f:
            json.dump(all_stats, f, indent=2)

        # Save matrices
        if circuit.component_matrix_correct is not None:
            np.save(os.path.join(out_dir, "comp_correct.npy"),
                    circuit.component_matrix_correct)
        if circuit.component_matrix_halluc is not None:
            np.save(os.path.join(out_dir, "comp_halluc.npy"),
                    circuit.component_matrix_halluc)

        circuits.append((model_name, circuit))

    # Cross-architecture comparison
    if len(circuits) >= 2:
        for i in range(len(circuits)):
            for j in range(i + 1, len(circuits)):
                name_a, circ_a = circuits[i]
                name_b, circ_b = circuits[j]
                comp = compare_architectures(circ_a, circ_b)

                print(f"\n{'='*70}")
                print(f"  CROSS-ARCHITECTURE: {name_a} vs {name_b}")
                print(f"{'='*70}")
                print(f"  Layer profile correlation: r={comp.layer_profile_corr:.3f} "
                      f"(p={comp.layer_profile_p:.4f})")
                print(f"  Component overlap (Jaccard): {comp.component_overlap:.3f}")
                print(f"  → {comp.interpretation}")

                save_comparison(comp,
                    os.path.join(args.output,
                                 f"{name_a}_vs_{name_b}.json"))


if __name__ == "__main__":
    main()
