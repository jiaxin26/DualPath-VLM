"""
Cross-Architecture Divergence Tests (N-way pairwise)
=====================================================

Runs Fisher's exact, Mann-Whitney, permutation, TOST, and bootstrap CI
for every pair of circuits. Supports 2, 3, or more models.

Usage:
  # 2-model (original)
  python experiments/reanalyze_crossarch.py \
    --circuits results/qwen3vl_head/circuit.json \
              results/llava_head/circuit.json \
    --output results/comparison/

  # 3-model
  python experiments/reanalyze_crossarch.py \
    --circuits results/qwen3vl_head/circuit.json \
              results/llava_head/circuit.json \
              results/internvl3_head/circuit.json \
    --output results/comparison_3way/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from itertools import combinations
from typing import List, Tuple

import numpy as np
from scipy import stats as sp_stats

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("reanalyze")


def load_circuit_components(path: str) -> Tuple[str, int, List[dict]]:
    """Load circuit.json, return (model_name, n_layers, components)."""
    with open(path) as f:
        data = json.load(f)
    return data["model_name"], data["n_layers"], data["components"]


def classify_depth_band(layer: int, n_layers: int) -> str:
    rel = layer / max(n_layers - 1, 1)
    if rel < 0.25:
        return "early"
    elif rel < 0.50:
        return "mid-early"
    elif rel < 0.75:
        return "mid-late"
    else:
        return "late"


# ════════════════════════════════════════════════════════════════════
# TEST 1: Fisher's exact
# ════════════════════════════════════════════════════════════════════

def test_fisher(comps_a, nl_a, comps_b, nl_b, name_a, name_b):
    halluc_a = [c for c in comps_a if c["cohen_d"] < 0 and c["type"] != "head"]
    halluc_b = [c for c in comps_b if c["cohen_d"] < 0 and c["type"] != "head"]

    def split(comps, nl):
        early = sum(1 for c in comps if c["layer"] / max(nl - 1, 1) < 0.5)
        return early, len(comps) - early

    ea, la = split(halluc_a, nl_a)
    eb, lb = split(halluc_b, nl_b)
    table = np.array([[ea, la], [eb, lb]])

    print(f"\n  Contingency (d<0 components):")
    print(f"  {'':25s} {'Early':>8s} {'Late':>8s}")
    print(f"  {name_a:25s} {ea:>8d} {la:>8d}")
    print(f"  {name_b:25s} {eb:>8d} {lb:>8d}")

    odds_ratio, p = sp_stats.fisher_exact(table)
    print(f"  Fisher's exact: OR={odds_ratio:.3f}, p={p:.4f}")

    return {"odds_ratio": float(odds_ratio), "p": float(p),
            "table": table.tolist(),
            "n_halluc_a": len(halluc_a), "n_halluc_b": len(halluc_b)}


# ════════════════════════════════════════════════════════════════════
# TEST 2: Depth distribution (Mann-Whitney, KS, permutation)
# ════════════════════════════════════════════════════════════════════

def test_depth(comps_a, nl_a, comps_b, nl_b, name_a, name_b, n_perm=10000):
    depths_a = np.array([
        c["layer"] / max(nl_a - 1, 1)
        for c in comps_a if c["cohen_d"] < 0 and c["type"] != "head"])
    depths_b = np.array([
        c["layer"] / max(nl_b - 1, 1)
        for c in comps_b if c["cohen_d"] < 0 and c["type"] != "head"])

    if len(depths_a) < 2 or len(depths_b) < 2:
        print("  Too few halluc components for depth tests.")
        return {}

    print(f"  {name_a}: n={len(depths_a)}, median={np.median(depths_a):.3f}")
    print(f"  {name_b}: n={len(depths_b)}, median={np.median(depths_b):.3f}")

    u, p_mw = sp_stats.mannwhitneyu(depths_a, depths_b, alternative="two-sided")
    ks, p_ks = sp_stats.ks_2samp(depths_a, depths_b)
    print(f"  Mann-Whitney: U={u:.1f}, p={p_mw:.4f}")
    print(f"  KS test: D={ks:.3f}, p={p_ks:.4f}")

    # Permutation test on |Δmedian|
    obs = abs(np.median(depths_a) - np.median(depths_b))
    combined = np.concatenate([depths_a, depths_b])
    na = len(depths_a)
    rng = np.random.RandomState(42)
    count = sum(1 for _ in range(n_perm)
                if abs(np.median(rng.permutation(combined)[:na])
                       - np.median(rng.permutation(combined)[na:])) >= obs)
    p_perm = (count + 1) / (n_perm + 1)
    print(f"  Permutation (|Δmedian|={obs:.4f}): p={p_perm:.4f}")

    return {"mann_whitney": {"U": float(u), "p": float(p_mw)},
            "ks": {"D": float(ks), "p": float(p_ks)},
            "permutation": {"obs_diff": float(obs), "p": float(p_perm)},
            "median_a": float(np.median(depths_a)),
            "median_b": float(np.median(depths_b))}


# ════════════════════════════════════════════════════════════════════
# TEST 3: TOST equivalence + bootstrap CI
# ════════════════════════════════════════════════════════════════════

def test_equivalence(comps_a, nl_a, comps_b, nl_b, name_a, name_b,
                     r_bound=0.3):
    def layer_profile(comps, nl):
        prof = np.zeros(nl)
        for c in comps:
            if c["type"] != "head":
                prof[c["layer"]] += abs(c["cohen_d"])
        return prof

    prof_a = layer_profile(comps_a, nl_a)
    prof_b = layer_profile(comps_b, nl_b)

    bins = 100
    norm_a = np.interp(np.linspace(0, 1, bins),
                       np.linspace(0, 1, len(prof_a)), prof_a)
    norm_b = np.interp(np.linspace(0, 1, bins),
                       np.linspace(0, 1, len(prof_b)), prof_b)

    r, p_r = sp_stats.pearsonr(norm_a, norm_b)
    print(f"  Pearson r={r:.3f}, p={p_r:.4f}")

    # TOST
    z = np.arctanh(r)
    se = 1 / np.sqrt(bins - 3)
    z_bound = np.arctanh(r_bound)

    z_upper = (z - z_bound) / se
    p_upper = sp_stats.norm.cdf(z_upper)
    z_lower = (z - (-z_bound)) / se
    p_lower = 1 - sp_stats.norm.cdf(z_lower)
    p_tost = max(p_upper, p_lower)
    print(f"  TOST (|r|<{r_bound}): p={p_tost:.4f}"
          f" {'← rejects equivalence' if p_tost < 0.05 else ''}")

    # Bootstrap CI
    rng = np.random.RandomState(42)
    r_boots = []
    for _ in range(5000):
        idx = rng.choice(bins, bins, replace=True)
        rb, _ = sp_stats.pearsonr(norm_a[idx], norm_b[idx])
        r_boots.append(rb)
    ci_lo, ci_hi = np.percentile(r_boots, [2.5, 97.5])
    print(f"  Bootstrap 95% CI for r: [{ci_lo:.3f}, {ci_hi:.3f}]")

    return {"pearson_r": float(r), "pearson_p": float(p_r),
            "tost_p": float(p_tost), "tost_bound": r_bound,
            "bootstrap_ci": [float(ci_lo), float(ci_hi)]}


# ════════════════════════════════════════════════════════════════════
# MAIN — N-way pairwise
# ════════════════════════════════════════════════════════════════════

def run_pair(name_a, nl_a, comps_a, name_b, nl_b, comps_b):
    """Run all tests for one pair."""
    print("\n" + "=" * 70)
    print(f"  {name_a}  vs  {name_b}")
    print("=" * 70)

    result = {
        "model_a": name_a, "n_layers_a": nl_a, "n_comps_a": len(comps_a),
        "model_b": name_b, "n_layers_b": nl_b, "n_comps_b": len(comps_b),
    }

    print("\n--- Fisher's exact (halluc-pathway depth bands) ---")
    result["fisher"] = test_fisher(comps_a, nl_a, comps_b, nl_b, name_a, name_b)

    print("\n--- Depth distribution tests ---")
    result["depth"] = test_depth(comps_a, nl_a, comps_b, nl_b, name_a, name_b)

    print("\n--- Equivalence / profile correlation ---")
    result["equivalence"] = test_equivalence(
        comps_a, nl_a, comps_b, nl_b, name_a, name_b)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Cross-architecture divergence tests (N-way pairwise)")
    parser.add_argument("--circuits", nargs="+", required=True,
                        help="Paths to circuit.json files (2 or more)")
    parser.add_argument("--output", default="results/comparison_3way/")

    # Keep backward compat with old 2-model args
    parser.add_argument("--qwen-circuit", default=None)
    parser.add_argument("--llava-circuit", default=None)

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Handle old-style args
    circuit_paths = args.circuits or []
    if not circuit_paths:
        if args.qwen_circuit and args.llava_circuit:
            circuit_paths = [args.qwen_circuit, args.llava_circuit]
        else:
            parser.error("Provide --circuits (2+ paths)")

    # Load all circuits
    circuits = []
    for path in circuit_paths:
        name, nl, comps = load_circuit_components(path)
        circuits.append((name, nl, comps))
        print(f"Loaded: {name} ({nl} layers, {len(comps)} components)")

    if len(circuits) < 2:
        parser.error("Need at least 2 circuits for comparison")

    # All pairwise comparisons
    all_results = {}
    for (name_a, nl_a, comps_a), (name_b, nl_b, comps_b) in combinations(circuits, 2):
        pair_key = f"{name_a}_vs_{name_b}"
        all_results[pair_key] = run_pair(
            name_a, nl_a, comps_a, name_b, nl_b, comps_b)

    # ── Summary table ────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("SUMMARY TABLE (for paper)")
    print("=" * 70)

    header = (f"  {'Pair':40s} {'Fisher p':>10s} {'MW p':>10s} "
              f"{'KS p':>10s} {'TOST p':>10s} {'r':>8s} {'CI':>20s}")
    print(header)
    print("  " + "─" * 108)

    for pair_key, res in all_results.items():
        fp = res["fisher"]["p"]
        dp = res.get("depth", {})
        mw_p = dp.get("mann_whitney", {}).get("p", float("nan"))
        ks_p = dp.get("ks", {}).get("p", float("nan"))
        eq = res["equivalence"]
        ci = eq["bootstrap_ci"]
        print(f"  {pair_key:40s} {fp:>10.4f} {mw_p:>10.4f} "
              f"{ks_p:>10.4f} {eq['tost_p']:>10.4f} "
              f"{eq['pearson_r']:>8.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")

    # Macro-structure summary: Jaccard overlap (reuse from run_patching)
    print(f"\n  Interpretation:")
    all_fisher_ns = all(r["fisher"]["p"] > 0.05 for r in all_results.values())
    all_tost_sig = all(r["equivalence"]["tost_p"] < 0.05
                       for r in all_results.values())

    if all_fisher_ns and all_tost_sig:
        print("  ✓ ALL pairs: Fisher p>0.05 (same macro depth bands)")
        print("  ✓ ALL pairs: TOST p<0.05 (negligible profile correlation)")
        print("  → Shared macro-structure, distinct micro-implementation")
    else:
        for pk, r in all_results.items():
            fp = r["fisher"]["p"]
            tp = r["equivalence"]["tost_p"]
            status_f = "same bands" if fp > 0.05 else "DIFFERENT bands"
            status_t = "negligible r" if tp < 0.05 else "r not negligible"
            print(f"    {pk}: Fisher→{status_f}, TOST→{status_t}")

    # ── Save ─────────────────────────────────────────────────────
    out_path = os.path.join(args.output, "divergence_tests.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved to {out_path}")

    # LaTeX snippet
    tex_path = os.path.join(args.output, "crossarch_table.tex")
    with open(tex_path, "w") as f:
        f.write("% Cross-architecture comparison (3-way)\n")
        f.write("% Auto-generated by reanalyze_crossarch.py\n\n")
        for pk, res in all_results.items():
            eq = res["equivalence"]
            ci = eq["bootstrap_ci"]
            dp = res.get("depth", {})
            mw = dp.get("mann_whitney", {})
            ks = dp.get("ks", {})
            fi = res["fisher"]

            f.write(f"% {pk}\n")
            f.write(f"% Fisher p={fi['p']:.4f}, "
                    f"MW U={mw.get('U','?'):.1f} p={mw.get('p','?'):.4f}, "
                    f"KS D={ks.get('D','?'):.3f} p={ks.get('p','?'):.4f}\n")
            f.write(f"% Pearson r={eq['pearson_r']:.3f}, "
                    f"TOST p={eq['tost_p']:.4f}, "
                    f"CI=[{ci[0]:.3f}, {ci[1]:.3f}]\n\n")

    logger.info(f"LaTeX saved to {tex_path}")


if __name__ == "__main__":
    main()