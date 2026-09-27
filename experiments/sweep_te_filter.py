"""
Robustness Sweep: Circuit Stability Across |TE| Filter Thresholds
==================================================================

This script:
  • Loads batch_results.pkl
  • Applies |TE| filters at multiple thresholds {0.0, 0.05, 0.1, 0.15, 0.2}
  • Re-extracts the circuit under each filter
  • Reports how many samples are dropped and whether pathway composition
    changes

Also reports the fraction of samples that would be dropped — if it's <5%,
the filter is effectively a no-op and the paper's claim "we exclude <2%"
is consistent with any reasonable threshold in this range.

Purely post-hoc. No GPU needed.

Usage:
  PYTHONPATH=. python experiments/sweep_te_filter.py \
    --results-dir results/ \
    --te-thresholds 0.0 0.05 0.1 0.15 0.2 \
    --output results/robustness/te_filter/
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sweep_te")


DEFAULT_MODELS = {
    "qwen":        "qwen_component",
    "llava":       "llava_component",
    "llama":       "llama_component",
    "internvl8b":  "internvl8b_component",
    "internvl14b": "internvl14b_component",
}


def filter_batch_by_te(batch, te_threshold: float):
    """Return a new BatchPatchingResult with |TE| >= threshold samples only."""
    from patching.core import BatchPatchingResult

    kept = [r for r in batch.results if abs(r.total_effect) >= te_threshold]
    filtered = BatchPatchingResult(results=kept)
    return filtered, len(batch.results) - len(kept)


def pathway_keys(components):
    g = {(c.layer, c.component_type) for c in components if c.cohen_d > 0}
    h = {(c.layer, c.component_type) for c in components if c.cohen_d < 0}
    return g, h


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def sweep_one_model(mkey: str, result_dir: str,
                    te_thresholds: List[float],
                    d_threshold: float,
                    output_dir: str) -> Dict:
    from analysis.circuits import extract_circuit, save_circuit

    logger.info(f"\n{'='*70}\nMODEL: {mkey}   ({result_dir})\n{'='*70}")

    pkl = os.path.join(result_dir, "batch_results.pkl")
    if not os.path.exists(pkl):
        raise FileNotFoundError(f"No batch_results.pkl in {result_dir}")
    with open(pkl, "rb") as f:
        batch = pickle.load(f)
    n_orig = len(batch.results)

    model_out = os.path.join(output_dir, mkey)
    os.makedirs(model_out, exist_ok=True)

    # Reference: TE=0 (no filter, which matches current code behavior)
    ref_batch, _ = filter_batch_by_te(batch, 0.0)
    ref_circuit = extract_circuit(ref_batch, mkey, min_effect_size=d_threshold)
    ref_grnd, ref_hall = pathway_keys(ref_circuit.components)

    rows = []
    for te in te_thresholds:
        filtered, n_dropped = filter_batch_by_te(batch, te)
        n_kept = n_orig - n_dropped

        if n_kept < 10 or filtered.n_correct < 3 or filtered.n_hallucination < 3:
            logger.warning(
                f"  TE>={te}: only {n_kept} samples kept "
                f"({filtered.n_correct} correct, "
                f"{filtered.n_hallucination} halluc) — skipping")
            continue

        circuit = extract_circuit(filtered, mkey, min_effect_size=d_threshold)
        g_keys, h_keys = pathway_keys(circuit.components)

        summary = {
            "te_threshold": te,
            "n_samples_kept": n_kept,
            "frac_dropped": n_dropped / n_orig,
            "n_correct": filtered.n_correct,
            "n_halluc": filtered.n_hallucination,
            "halluc_rate": filtered.n_hallucination / n_kept,
            "n_total_components": len(circuit.components),
            "n_grounding": len(g_keys),
            "n_hallucination": len(h_keys),
            "jaccard_grounding": jaccard(ref_grnd, g_keys),
            "jaccard_hallucination": jaccard(ref_hall, h_keys),
        }
        rows.append(summary)

        save_circuit(
            circuit, os.path.join(model_out, f"circuit_te{te:.2f}.json"))

    # Print table
    print(f"\n  TE-filter sweep for {mkey} (original n={n_orig})")
    print(f"  {'|TE|>=':>7s}  {'kept':>5s}  {'drop%':>6s}  "
          f"{'Ntot':>4s}  {'Ngrnd':>5s}  {'Nhall':>5s}  "
          f"{'J_grnd':>7s}  {'J_hall':>7s}")
    print("  " + "-" * 60)
    for r in rows:
        print(f"  {r['te_threshold']:>7.2f}  {r['n_samples_kept']:>5d}  "
              f"{r['frac_dropped']*100:>5.1f}%  "
              f"{r['n_total_components']:>4d}  "
              f"{r['n_grounding']:>5d}  {r['n_hallucination']:>5d}  "
              f"{r['jaccard_grounding']:>7.3f}  "
              f"{r['jaccard_hallucination']:>7.3f}")

    # ── Sanity check against paper's "< 2%" claim ───────────────────
    te01 = [r for r in rows if abs(r["te_threshold"] - 0.1) < 1e-9]
    if te01:
        pct = te01[0]["frac_dropped"] * 100
        verdict = ("✓ consistent with paper" if pct < 5
                   else f"⚠ ACTUAL {pct:.1f}% dropped, paper says <2% — investigate")
        print(f"\n  |TE|<0.1 drops {pct:.2f}% of samples   [{verdict}]")

    return {"model": mkey, "n_original": n_orig, "rows": rows}


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc sweep of |TE| filter for circuit stability")
    parser.add_argument("--results-dir", type=str, default="results/")
    parser.add_argument("--single-model", type=str, default=None)
    parser.add_argument("--te-thresholds", nargs="+", type=float,
                        default=[0.0, 0.05, 0.1, 0.15, 0.2])
    parser.add_argument("--d-threshold", type=float, default=0.3,
                        help="Cohen's d threshold for extract_circuit")
    parser.add_argument("--output", type=str,
                        default="results/robustness/te_filter/")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.single_model:
        models = {args.single_model: args.results_dir}
    else:
        models = {k: os.path.join(args.results_dir, v)
                  for k, v in DEFAULT_MODELS.items()
                  if os.path.isdir(os.path.join(args.results_dir, v))}

    all_results = {}
    for mkey, rdir in models.items():
        try:
            all_results[mkey] = sweep_one_model(
                mkey, rdir, args.te_thresholds, args.d_threshold, args.output)
        except FileNotFoundError as e:
            logger.warning(f"Skipping {mkey}: {e}")

    # ── Cross-model summary ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CROSS-MODEL: fraction of samples dropped at each |TE| threshold")
    print("=" * 70)
    print(f"  {'model':<14s}" + "".join(
        f" {t:>7.2f}" for t in args.te_thresholds))
    for mkey, res in all_results.items():
        row = f"  {mkey:<14s}"
        for t in args.te_thresholds:
            m = [r for r in res["rows"] if abs(r["te_threshold"] - t) < 1e-9]
            row += (f" {m[0]['frac_dropped']*100:>6.1f}%" if m
                    else f" {'--':>7s}")
        print(row)

    print("\n" + "=" * 70)
    print("CROSS-MODEL: Jaccard(hallucination pathway vs TE=0 reference)")
    print("=" * 70)
    print(f"  {'model':<14s}" + "".join(
        f" {t:>7.2f}" for t in args.te_thresholds))
    for mkey, res in all_results.items():
        row = f"  {mkey:<14s}"
        for t in args.te_thresholds:
            m = [r for r in res["rows"] if abs(r["te_threshold"] - t) < 1e-9]
            row += (f" {m[0]['jaccard_hallucination']:>7.3f}" if m
                    else f" {'--':>7s}")
        print(row)

    summary_path = os.path.join(args.output, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Summary saved to {summary_path}")

    # ── Advisory messages ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
  If J_hall stays > 0.7 across all thresholds:
    → Pathway is highly stable. Report this in the robustness appendix.

  If the |TE|<0.1 drop rate is < 2% in all models:
    → The paper's claim holds. Add a sentence confirming that this was
      verified post-hoc and cite this experiment.

  If the |TE|<0.1 drop rate is noticeably > 2% in some models:
    → The paper's claim does NOT hold as stated. Fix the text: either
      report the actual drop rate, or apply the filter explicitly in
      extract_circuit() and re-run the main Table 1 numbers.

  If Jaccard drops sharply at some threshold:
    → That threshold changes the circuit. Discuss which components
      drop in/out and why.
""")


if __name__ == "__main__":
    main()
