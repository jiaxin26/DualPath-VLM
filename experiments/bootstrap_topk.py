"""
Bootstrap Analysis: Top-k Selection Stability
==============================================

This script is PURELY POST-HOC — it consumes per-sample predictions
saved by eval_topk_per_sample.py and bootstraps the selection process.

For B=1000 bootstrap iterations:
  1. Sample 100 samples WITH replacement from the evaluation set
  2. Compute accuracy and halluc_rate for each k on this subset
  3. Apply the paper's selection criterion (min halluc subject to
     acc_drop <= 2pp) to pick the best k
  4. Record the selected k

We also bootstrap the halluc_rate of the paper's chosen k (k=10 for
LLaVA) to produce a 95% CI on the relative reduction.

Outputs:
  • Distribution over selected k (how often each k wins)
  • 95% CI on relative hallucination reduction at the paper's chosen k
  • CI on the accuracy-cost of the chosen k

No GPU needed.

Usage:
  # After running eval_topk_per_sample.py:
  PYTHONPATH=. python experiments/bootstrap_topk.py \
    --results-dir results/robustness/topk_per_sample_llava/ \
    --n-bootstrap 1000 \
    --n-selection 100 \
    --paper-k 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bootstrap_topk")


def compute_metrics(per_sample: List[dict], indices: np.ndarray) -> Dict:
    """Compute accuracy and halluc_rate on a subset of per-sample records.

    Args:
        per_sample: list of {gt, pred, ...} dicts
        indices: array of sample indices (with replacement allowed)
    """
    tp = fp = tn = fn = 0
    for i in indices:
        r = per_sample[i]
        gt, pred = r["gt"], r["pred"]
        if pred == "yes" and gt == "yes":   tp += 1
        elif pred == "yes" and gt == "no":  fp += 1
        elif pred == "no" and gt == "no":   tn += 1
        else:                                fn += 1
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    halluc = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {"accuracy": acc, "halluc_rate": halluc,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def select_best_k(baseline_m: Dict, k_results: Dict[int, Dict],
                  max_acc_drop: float = 0.02) -> Optional[int]:
    """Paper's selection criterion: min halluc_rate subject to
    acc_drop <= 2pp relative to baseline. Returns None if no k satisfies.
    """
    best_k, best_halluc = None, float("inf")
    for k, m in k_results.items():
        acc_drop = baseline_m["accuracy"] - m["accuracy"]
        if acc_drop > max_acc_drop:
            continue
        if m["halluc_rate"] < best_halluc:
            best_k = k
            best_halluc = m["halluc_rate"]
    # Fallback: if none satisfies, pick lowest halluc ignoring constraint
    if best_k is None and k_results:
        best_k = min(k_results, key=lambda k: k_results[k]["halluc_rate"])
    return best_k


def load_per_sample_results(results_dir: str) -> tuple:
    """Load baseline + all per-k per-sample results from a directory."""
    baseline_path = os.path.join(results_dir, "baseline.pkl")
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(
            f"No baseline.pkl in {results_dir}. "
            f"Run eval_topk_per_sample.py first.")
    with open(baseline_path, "rb") as f:
        baseline = pickle.load(f)

    k_results = {}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.startswith("top-") or not fname.endswith(".pkl"):
            continue
        k = int(fname[len("top-"):-len(".pkl")])
        with open(os.path.join(results_dir, fname), "rb") as f:
            k_results[k] = pickle.load(f)

    if not k_results:
        raise ValueError(f"No top-k pickles found in {results_dir}")

    logger.info(f"Loaded baseline ({len(baseline['per_sample'])} samples) "
                f"+ {len(k_results)} k values: {sorted(k_results.keys())}")
    return baseline, k_results


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap analysis of top-k selection stability")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory containing baseline.pkl and "
                             "top-k.pkl files from eval_topk_per_sample.py")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-selection", type=int, default=100,
                        help="Size of each bootstrap selection set "
                             "(matches the paper's Section 4 split)")
    parser.add_argument("--paper-k", type=int, default=10,
                        help="The k value the paper selected (for 95% CI)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-acc-drop", type=float, default=0.02)
    parser.add_argument("--output", type=str, default=None,
                        help="JSON output path (default: results-dir/bootstrap.json)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    baseline, k_results = load_per_sample_results(args.results_dir)
    baseline_ps = baseline["per_sample"]
    n_total = len(baseline_ps)

    if args.n_selection > n_total:
        logger.warning(f"n_selection={args.n_selection} > n_total={n_total}, "
                       f"clamping")
        args.n_selection = n_total

    # Sanity check: all k_results must have the same per_sample ordering
    # (they should, since eval_topk_per_sample.py evaluates the same set)
    ref_len = len(baseline_ps)
    for k, kr in k_results.items():
        if len(kr["per_sample"]) != ref_len:
            raise ValueError(
                f"Length mismatch: baseline has {ref_len} samples but "
                f"k={k} has {len(kr['per_sample'])}. Re-run "
                f"eval_topk_per_sample.py with consistent data.")

    logger.info(f"Running {args.n_bootstrap} bootstrap iterations "
                f"(n_selection={args.n_selection})")

    selected_ks: List[int] = []
    relative_reductions: List[float] = []
    acc_drops: List[float] = []
    per_k_wins = {k: 0 for k in k_results}

    for b in range(args.n_bootstrap):
        # Sample indices with replacement
        idx = rng.choice(n_total, size=args.n_selection, replace=True)

        # Compute metrics for baseline + each k on this bootstrap subset
        base_m = compute_metrics(baseline_ps, idx)
        k_m = {k: compute_metrics(kr["per_sample"], idx)
               for k, kr in k_results.items()}

        # Select best k via paper's criterion
        best_k = select_best_k(base_m, k_m, args.max_acc_drop)
        if best_k is not None:
            selected_ks.append(best_k)
            per_k_wins[best_k] += 1

        # Also track relative reduction at the paper's chosen k
        if args.paper_k in k_m:
            pk_m = k_m[args.paper_k]
            if base_m["halluc_rate"] > 0:
                rel_red = 1 - pk_m["halluc_rate"] / base_m["halluc_rate"]
                relative_reductions.append(rel_red)
            acc_drops.append(base_m["accuracy"] - pk_m["accuracy"])

    # ── Distribution over selected k ────────────────────────────────
    print("\n" + "=" * 60)
    print("BOOTSTRAP RESULTS: how often does each k win?")
    print("=" * 60)
    counter = Counter(selected_ks)
    total_selections = sum(counter.values())
    print(f"  {'k':>4s}  {'wins':>5s}  {'freq':>7s}  bar")
    for k in sorted(k_results.keys()):
        wins = counter.get(k, 0)
        freq = wins / max(total_selections, 1)
        bar = "█" * int(freq * 40)
        print(f"  {k:>4d}  {wins:>5d}  {freq:>6.1%}  {bar}")

    mode_k = counter.most_common(1)[0][0] if counter else None
    print(f"\n  Mode:        k = {mode_k} "
          f"({counter.get(mode_k, 0)/total_selections:.1%} of runs)")

    # Check stability vs paper's chosen k
    paper_wins = counter.get(args.paper_k, 0)
    print(f"  Paper's k={args.paper_k}: selected in "
          f"{paper_wins/max(total_selections,1):.1%} of bootstraps")

    if mode_k == args.paper_k:
        print(f"  → STABLE: the paper's choice IS the bootstrap mode.")
    else:
        print(f"  → UNSTABLE: mode is k={mode_k}, not k={args.paper_k}. "
              f"Paper's choice may be an artifact of the specific selection set.")

    # ── 95% CI on relative reduction at paper's chosen k ────────────
    print("\n" + "=" * 60)
    print(f"95% CI ON METRICS AT PAPER'S k={args.paper_k}")
    print("=" * 60)
    if relative_reductions:
        rr = np.array(relative_reductions)
        rr_mean = rr.mean()
        rr_lo, rr_hi = np.percentile(rr, [2.5, 97.5])
        print(f"  Relative hallucination reduction:")
        print(f"    Mean:   {rr_mean:>+7.1%}")
        print(f"    95% CI: [{rr_lo:>+7.1%}, {rr_hi:>+7.1%}]")

    if acc_drops:
        ad = np.array(acc_drops)
        print(f"\n  Accuracy drop (pp):")
        print(f"    Mean:   {ad.mean()*100:>+6.2f}")
        print(f"    95% CI: [{np.percentile(ad,2.5)*100:>+6.2f}, "
              f"{np.percentile(ad,97.5)*100:>+6.2f}]")

    # ── Save ────────────────────────────────────────────────────────
    out = args.output or os.path.join(args.results_dir, "bootstrap.json")
    with open(out, "w") as f:
        json.dump({
            "n_bootstrap": args.n_bootstrap,
            "n_selection": args.n_selection,
            "paper_k": args.paper_k,
            "max_acc_drop": args.max_acc_drop,
            "selection_counts": {str(k): int(counter.get(k, 0))
                                 for k in k_results},
            "mode_k": int(mode_k) if mode_k is not None else None,
            "paper_k_selection_freq": paper_wins / max(total_selections, 1),
            "relative_reduction_mean": (float(np.mean(relative_reductions))
                                        if relative_reductions else None),
            "relative_reduction_ci95": (
                [float(np.percentile(relative_reductions, 2.5)),
                 float(np.percentile(relative_reductions, 97.5))]
                if relative_reductions else None),
            "acc_drop_mean_pp": (float(np.mean(acc_drops) * 100)
                                 if acc_drops else None),
            "acc_drop_ci95_pp": (
                [float(np.percentile(acc_drops, 2.5) * 100),
                 float(np.percentile(acc_drops, 97.5) * 100)]
                if acc_drops else None),
        }, f, indent=2)
    logger.info(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
