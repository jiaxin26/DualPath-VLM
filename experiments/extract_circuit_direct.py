"""
Direct Circuit Extraction from Pickle
======================================

Bypasses extract_circuit / component_significance entirely.
Computes stats from raw pickle data, applies BH correction,
writes a clean circuit.json.

Usage:
  python experiments/extract_circuit_direct.py \
    results/qwen3vl_component_n500/batch_results.pkl \
    results/llava_component/batch_results.pkl \
    --output results/circuits_clean/
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats


def benjamini_hochberg(p_values, alpha=0.05):
    """BH FDR correction. Returns (adjusted_p, significant_mask)."""
    m = len(p_values)
    if m == 0:
        return np.array([]), np.array([], dtype=bool)

    # Handle NaN: treat as p=1.0
    p_clean = np.where(np.isnan(p_values), 1.0, p_values)

    idx = np.argsort(p_clean)
    sorted_p = p_clean[idx]

    significant = np.zeros(m, dtype=bool)
    max_k = -1
    for k in range(m):
        if sorted_p[k] <= (k + 1) / m * alpha:
            max_k = k
    if max_k >= 0:
        significant[idx[:max_k + 1]] = True

    return significant


def extract_from_pickle(pkl_path, model_name=None, alpha=0.05, min_d=0.3):
    """Load pickle, compute all stats, return circuit dict."""

    with open(pkl_path, "rb") as f:
        batch = pickle.load(f)

    results = batch.results
    correct = [r for r in results if not r.is_hallucination]
    halluc = [r for r in results if r.is_hallucination]

    n_layers = results[0].n_layers
    n_heads = results[0].n_heads

    print(f"\n{'='*70}")
    print(f"  {model_name or pkl_path}")
    print(f"  {len(correct)} correct, {len(halluc)} halluc, {n_layers} layers")
    print(f"{'='*70}")

    # Get normalized component effects
    corr = np.array([r.normalized_component for r in correct])   # (n_c, n_layers, 2)
    hall = np.array([r.normalized_component for r in halluc])     # (n_h, n_layers, 2)

    print(f"  corr shape: {corr.shape}, hall shape: {hall.shape}")

    # Check for NaN/inf
    n_nan_c = np.isnan(corr).sum()
    n_nan_h = np.isnan(hall).sum()
    if n_nan_c > 0 or n_nan_h > 0:
        print(f"  WARNING: {n_nan_c} NaN in correct, {n_nan_h} NaN in halluc")
        corr = np.nan_to_num(corr, 0.0)
        hall = np.nan_to_num(hall, 0.0)

    # Compute stats for all components
    comp_names = ["attn", "mlp"]
    all_p = []
    all_stats = []

    for l in range(n_layers):
        for c_idx, c_name in enumerate(comp_names):
            cv = corr[:, l, c_idx]
            hv = hall[:, l, c_idx]

            t, p = sp_stats.ttest_ind(cv, hv, equal_var=False)
            if np.isnan(p):
                p = 1.0

            pooled = np.sqrt((cv.var(ddof=1) + hv.var(ddof=1)) / 2)
            d = (cv.mean() - hv.mean()) / pooled if pooled > 1e-12 else 0.0

            all_p.append(p)
            all_stats.append({
                "layer": l,
                "type": c_name,
                "head_idx": -1,
                "effect_correct": float(cv.mean()),
                "effect_halluc": float(hv.mean()),
                "differential": float(cv.mean() - hv.mean()),
                "p_value": float(p),
                "cohen_d": float(d),
                "rel_depth": float(l / max(n_layers - 1, 1)),
            })

    # BH correction
    all_p = np.array(all_p)
    sig_mask = benjamini_hochberg(all_p, alpha)

    # Filter: significant AND |d| >= min_d
    components = []
    for i, s in enumerate(all_stats):
        s["significant_bh"] = bool(sig_mask[i])
        if sig_mask[i] and abs(s["cohen_d"]) >= min_d:
            components.append(s)

    # Also compute head-level if available
    head_components = []
    correct_heads = [r.normalized_heads for r in correct if r.normalized_heads is not None]
    halluc_heads = [r.normalized_heads for r in halluc if r.normalized_heads is not None]

    if len(correct_heads) >= 3 and len(halluc_heads) >= 3:
        ch = np.array(correct_heads)
        hh = np.array(halluc_heads)
        print(f"  Head data: correct {ch.shape}, halluc {hh.shape}")

        head_p = []
        head_stats = []
        for l in range(n_layers):
            for h in range(n_heads):
                cv = ch[:, l, h]
                hv = hh[:, l, h]
                t, p = sp_stats.ttest_ind(cv, hv, equal_var=False)
                if np.isnan(p):
                    p = 1.0
                pooled = np.sqrt((cv.var(ddof=1) + hv.var(ddof=1)) / 2)
                d = (cv.mean() - hv.mean()) / pooled if pooled > 1e-12 else 0.0
                head_p.append(p)
                head_stats.append({
                    "layer": l, "type": "head", "head_idx": h,
                    "effect_correct": float(cv.mean()),
                    "effect_halluc": float(hv.mean()),
                    "differential": float(cv.mean() - hv.mean()),
                    "p_value": float(p), "cohen_d": float(d),
                    "rel_depth": float(l / max(n_layers - 1, 1)),
                })

        head_sig = benjamini_hochberg(np.array(head_p), alpha)
        for i, s in enumerate(head_stats):
            if head_sig[i] and abs(s["cohen_d"]) >= min_d:
                head_components.append(s)
        print(f"  Head-level: {len(head_components)} significant")

    all_components = components + head_components

    # Print results
    print(f"\n  Component-level: {len(components)} significant (BH α={alpha}, |d|≥{min_d})")
    print(f"  Pre-BH with p<0.05 and |d|≥{min_d}: {sum(1 for i,s in enumerate(all_stats) if all_p[i] < 0.05 and abs(s['cohen_d']) >= min_d)}")

    # Sort by |d| and print
    components_sorted = sorted(components, key=lambda x: -abs(x["cohen_d"]))
    print(f"\n  {'Name':>12s} {'depth':>6s} {'d':>8s} {'p':>12s} {'correct':>10s} {'halluc':>10s} {'pathway':>10s}")
    print(f"  {'-'*72}")
    for c in components_sorted[:15]:
        name = f"L{c['layer']}.{c['type']}"
        pathway = "Visual" if c["cohen_d"] > 0 else "Halluc"
        print(f"  {name:>12s} {c['rel_depth']:>6.0%} {c['cohen_d']:>+8.3f} "
              f"{c['p_value']:>12.2e} {c['effect_correct']:>+10.4f} "
              f"{c['effect_halluc']:>+10.4f} {pathway:>10s}")

    # Layer band summary
    print(f"\n  Layer band summary:")
    bands = [("Early (0-24%)", 0, 0.25), ("Mid-early (25-49%)", 0.25, 0.50),
             ("Mid-late (50-74%)", 0.50, 0.75), ("Late (75-100%)", 0.75, 1.01)]
    for bname, lo, hi in bands:
        band = [c for c in components if lo <= c["rel_depth"] < hi]
        pos = sum(1 for c in band if c["cohen_d"] > 0)
        neg = sum(1 for c in band if c["cohen_d"] < 0)
        print(f"    {bname:20s}: {len(band):2d} sig (d>0: {pos}, d<0: {neg})")

    # Mean effect matrices for viz
    comp_correct = corr.mean(axis=0)  # (n_layers, 2)
    comp_halluc = hall.mean(axis=0)

    circuit = {
        "model_name": model_name,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_samples": len(results),
        "n_correct": len(correct),
        "n_hallucination": len(halluc),
        "n_components": len(all_components),
        "n_component_level": len(components),
        "n_head_level": len(head_components),
        "components": all_components,
        "component_matrix_correct": comp_correct.tolist(),
        "component_matrix_halluc": comp_halluc.tolist(),
    }

    return circuit, all_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pickles", nargs="+",
                        help="Paths to batch_results.pkl files")
    parser.add_argument("--output", default="results/circuits_clean/")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-d", type=float, default=0.3)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    circuits = []

    for pkl_path in args.pickles:
        # Infer model name
        results_dir = str(Path(pkl_path).parent)
        summary_path = os.path.join(results_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
            model_name = summary.get("model", Path(results_dir).name).split("/")[-1]
        else:
            model_name = Path(results_dir).name

        circuit, all_stats = extract_from_pickle(
            pkl_path, model_name, args.alpha, args.min_d)

        # Save circuit
        out_path = os.path.join(args.output, f"{model_name}_circuit.json")
        with open(out_path, "w") as f:
            json.dump(circuit, f, indent=2)
        print(f"\n  Saved: {out_path}")

        # Save full stats for reference
        stats_path = os.path.join(args.output, f"{model_name}_all_stats.json")
        with open(stats_path, "w") as f:
            json.dump(all_stats, f, indent=2)

        circuits.append((model_name, circuit))

    # Cross-architecture summary if multiple
    if len(circuits) >= 2:
        print(f"\n\n{'='*70}")
        print(f"  CROSS-ARCHITECTURE QUICK COMPARISON")
        print(f"{'='*70}")
        for i, (na, ca) in enumerate(circuits):
            for j, (nb, cb) in enumerate(circuits):
                if j <= i:
                    continue

                # Compare d<0 component depth distributions
                depths_a = [c["rel_depth"] for c in ca["components"]
                           if c["cohen_d"] < 0 and c["type"] != "head"]
                depths_b = [c["rel_depth"] for c in cb["components"]
                           if c["cohen_d"] < 0 and c["type"] != "head"]

                print(f"\n  {na} vs {nb}")
                print(f"    {na}: {len(ca['components'])} components "
                      f"({ca['n_component_level']} comp-level, {ca['n_head_level']} head-level)")
                print(f"    {nb}: {len(cb['components'])} components "
                      f"({cb['n_component_level']} comp-level, {cb['n_head_level']} head-level)")

                if depths_a and depths_b:
                    da = np.array(depths_a)
                    db = np.array(depths_b)
                    print(f"    Halluc pathway (d<0) depth:")
                    print(f"      {na}: median={np.median(da):.2f}, "
                          f"range=[{da.min():.2f}, {da.max():.2f}]")
                    print(f"      {nb}: median={np.median(db):.2f}, "
                          f"range=[{db.min():.2f}, {db.max():.2f}]")
                    u, p = sp_stats.mannwhitneyu(da, db, alternative="two-sided")
                    print(f"      Mann-Whitney U={u:.1f}, p={p:.4f}")


if __name__ == "__main__":
    main()