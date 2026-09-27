"""
Robustness Sweep: Circuit Stability Across Cohen's d Thresholds
================================================================

This script is PURELY POST-HOC — it loads the cached batch_results.pkl
from each model's Phase-1 patching run and re-extracts circuits at
multiple thresholds, WITHOUT re-running patching. No GPU needed.

For each threshold t ∈ {0.2, 0.25, 0.3, 0.35, 0.4}, we:
  1. Re-extract the circuit using extract_circuit(min_effect_size=t)
  2. Record n_grounding, n_hallucination, pathway membership
  3. Compute Jaccard similarity with the t=0.3 baseline circuit
  4. Save a new circuit_d{t}.json

After this runs, the user gets a ready-to-paste command block to re-run
CPA on each new circuit (this DOES need GPU, ~2-3h per model per threshold).

Usage:
  # Sweep thresholds for all 5 models (post-hoc, no GPU)
  PYTHONPATH=. python experiments/sweep_d_threshold.py \
    --results-dir results/ \
    --thresholds 0.2 0.25 0.3 0.35 0.4 \
    --output results/robustness/d_threshold/

  # Single model
  PYTHONPATH=. python experiments/sweep_d_threshold.py \
    --results-dir results/qwen_component/ \
    --single-model qwen \
    --output results/robustness/d_threshold_qwen/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sweep_d")


# Default model directory names under results/ (matches reproduce.sh)
DEFAULT_MODELS = {
    "qwen":         "qwen_component",
    "llava":        "llava_component",
    "llama":        "llama_component",
    "internvl8b":   "internvl8b_component",
    "internvl14b":  "internvl14b_component",
}


def load_batch(result_dir: str):
    """Load batch_results.pkl from a Phase-1 patching run."""
    path = os.path.join(result_dir, "batch_results.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No batch_results.pkl in {result_dir}. "
            f"Run Phase 1 (run_patching.py) first.")
    with open(path, "rb") as f:
        batch = pickle.load(f)
    logger.info(f"  Loaded {len(batch.results)} patching results from {path}")
    return batch


def pathway_summary(components) -> Dict[str, int]:
    """Count grounding vs halluc components."""
    n_grnd = sum(1 for c in components if c.cohen_d > 0)
    n_hall = sum(1 for c in components if c.cohen_d < 0)
    n_attn_grnd = sum(1 for c in components
                      if c.cohen_d > 0 and c.component_type == "attn")
    n_mlp_grnd = sum(1 for c in components
                     if c.cohen_d > 0 and c.component_type == "mlp")
    n_attn_hall = sum(1 for c in components
                      if c.cohen_d < 0 and c.component_type == "attn")
    n_mlp_hall = sum(1 for c in components
                     if c.cohen_d < 0 and c.component_type == "mlp")
    return {
        "n_total": len(components),
        "n_grounding": n_grnd,
        "n_hallucination": n_hall,
        "n_attn_grounding": n_attn_grnd,
        "n_mlp_grounding": n_mlp_grnd,
        "n_attn_halluc": n_attn_hall,
        "n_mlp_halluc": n_mlp_hall,
    }


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def component_keys(components) -> Dict[str, set]:
    """Convert components to hashable sets for Jaccard."""
    grnd = {(c.layer, c.component_type) for c in components if c.cohen_d > 0}
    hall = {(c.layer, c.component_type) for c in components if c.cohen_d < 0}
    all_ = grnd | hall
    return {"grounding": grnd, "hallucination": hall, "all": all_}


def sweep_one_model(model_key: str, result_dir: str,
                    thresholds: List[float],
                    baseline_d: float,
                    output_dir: str) -> Dict:
    """Run d-threshold sweep for a single model."""
    from analysis.circuits import extract_circuit, save_circuit

    logger.info(f"\n{'='*70}\nMODEL: {model_key}   ({result_dir})\n{'='*70}")

    batch = load_batch(result_dir)
    model_out = os.path.join(output_dir, model_key)
    os.makedirs(model_out, exist_ok=True)

    # Extract circuit at baseline threshold first (for Jaccard reference)
    baseline_circuit = extract_circuit(batch, model_key,
                                       min_effect_size=baseline_d)
    baseline_keys = component_keys(baseline_circuit.components)

    rows = []
    per_threshold_circuits = {}

    for t in thresholds:
        circuit = extract_circuit(batch, model_key, min_effect_size=t)
        summary = pathway_summary(circuit.components)
        keys = component_keys(circuit.components)

        # Jaccard vs baseline
        j_grnd = jaccard(baseline_keys["grounding"], keys["grounding"])
        j_hall = jaccard(baseline_keys["hallucination"], keys["hallucination"])
        j_all  = jaccard(baseline_keys["all"], keys["all"])

        row = {
            "threshold": t,
            "is_baseline": abs(t - baseline_d) < 1e-9,
            **summary,
            "jaccard_grounding_vs_baseline": j_grnd,
            "jaccard_halluc_vs_baseline": j_hall,
            "jaccard_all_vs_baseline": j_all,
        }
        rows.append(row)

        # Save circuit JSON at each threshold
        circuit_path = os.path.join(model_out, f"circuit_d{t:.2f}.json")
        save_circuit(circuit, circuit_path)
        per_threshold_circuits[t] = circuit_path

    # Print table
    print(f"\n  d-threshold sweep for {model_key}")
    print(f"  {'thr':>5s}  {'N':>4s}  {'Ngrnd':>5s}  {'Nhall':>5s}  "
          f"{'J_grnd':>7s}  {'J_hall':>7s}  {'J_all':>7s}  note")
    print("  " + "-" * 65)
    for r in rows:
        note = "← baseline" if r["is_baseline"] else ""
        print(f"  {r['threshold']:>5.2f}  {r['n_total']:>4d}  "
              f"{r['n_grounding']:>5d}  {r['n_hallucination']:>5d}  "
              f"{r['jaccard_grounding_vs_baseline']:>7.3f}  "
              f"{r['jaccard_halluc_vs_baseline']:>7.3f}  "
              f"{r['jaccard_all_vs_baseline']:>7.3f}  {note}")

    # Dual-pathway preservation check
    dual_pathway_preserved = all(
        r["n_grounding"] >= 2 and r["n_hallucination"] >= 2 for r in rows)
    print(f"\n  Dual-pathway preserved at all thresholds: "
          f"{'YES' if dual_pathway_preserved else 'NO'}")

    return {
        "model": model_key,
        "result_dir": result_dir,
        "baseline_threshold": baseline_d,
        "rows": rows,
        "dual_pathway_preserved": dual_pathway_preserved,
        "circuit_paths": per_threshold_circuits,
    }


def generate_cpa_commands(all_results: Dict, output_dir: str,
                          n_cpa: int = 400) -> str:
    """Generate bash commands for re-running CPA at each threshold."""
    # Map model key → HF model id (same as reproduce.sh)
    model_ids = {
        "qwen":         "Qwen/Qwen3-VL-8B-Instruct",
        "llava":        "llava-hf/llava-v1.6-mistral-7b-hf",
        "internvl8b":   "OpenGVLab/InternVL3-8B-hf",
        "llama":        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "internvl14b":  "OpenGVLab/InternVL3-14B-hf",
    }

    lines = []
    lines.append("# " + "=" * 66)
    lines.append("# Commands to re-run CPA (Phase 3) at each d-threshold")
    lines.append("# Each command needs GPU, ~2-3h per model.")
    lines.append("# " + "=" * 66)
    lines.append("")

    for mkey, res in all_results.items():
        if mkey not in model_ids:
            continue
        model_id = model_ids[mkey]
        lines.append(f"# ── {mkey} ──────────────────────────────")
        for t, circuit_path in res["circuit_paths"].items():
            if abs(t - res["baseline_threshold"]) < 1e-9:
                continue  # skip baseline, already done
            cpa_out = os.path.join(output_dir, mkey, f"cpa_d{t:.2f}")
            lines.append(
                f"PYTHONPATH=. python experiments/run_pathway_patching.py \\\n"
                f"  --model {model_id} \\\n"
                f"  --circuit {circuit_path} \\\n"
                f"  --pope-path data/pope/ --image-dir data/coco/val2014/ \\\n"
                f"  --n-samples {n_cpa} --min-d {t} \\\n"
                f"  --output {cpa_out}/")
            lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc sweep of d-threshold for circuit stability")
    parser.add_argument("--results-dir", type=str, default="results/",
                        help="Base directory containing per-model Phase-1 dirs "
                             "(e.g. results/qwen_component/)")
    parser.add_argument("--single-model", type=str, default=None,
                        help="If set, sweep only this model key. "
                             "Use --results-dir as the model's Phase-1 dir.")
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0.2, 0.25, 0.3, 0.35, 0.4])
    parser.add_argument("--baseline-d", type=float, default=0.3,
                        help="The threshold used in the paper (for Jaccard "
                             "reference)")
    parser.add_argument("--output", type=str,
                        default="results/robustness/d_threshold/")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.single_model:
        models = {args.single_model: args.results_dir}
    else:
        models = {k: os.path.join(args.results_dir, v)
                  for k, v in DEFAULT_MODELS.items()
                  if os.path.isdir(os.path.join(args.results_dir, v))}

    if not models:
        logger.error(f"No valid model result dirs found under {args.results_dir}")
        logger.error(f"Expected subdirs: {list(DEFAULT_MODELS.values())}")
        return

    logger.info(f"Sweeping {len(models)} models × {len(args.thresholds)} "
                f"thresholds = {len(models) * len(args.thresholds)} circuits")

    all_results = {}
    for mkey, rdir in models.items():
        try:
            all_results[mkey] = sweep_one_model(
                mkey, rdir, args.thresholds, args.baseline_d, args.output)
        except FileNotFoundError as e:
            logger.warning(f"Skipping {mkey}: {e}")

    # ── Aggregate summary across models ─────────────────────────────
    print("\n" + "=" * 70)
    print("CROSS-MODEL SUMMARY: n_components at each threshold")
    print("=" * 70)
    header = f"  {'model':<14s}" + "".join(
        f" {t:>6.2f}" for t in args.thresholds)
    print(header)
    for mkey, res in all_results.items():
        row = f"  {mkey:<14s}"
        for t in args.thresholds:
            matching = [r for r in res["rows"] if abs(r["threshold"] - t) < 1e-9]
            n = matching[0]["n_total"] if matching else -1
            row += f" {n:>6d}"
        print(row)

    print("\n" + "=" * 70)
    print("CROSS-MODEL SUMMARY: (n_grounding / n_hallucination)")
    print("=" * 70)
    print(header)
    for mkey, res in all_results.items():
        row = f"  {mkey:<14s}"
        for t in args.thresholds:
            matching = [r for r in res["rows"] if abs(r["threshold"] - t) < 1e-9]
            if matching:
                r = matching[0]
                row += f" {r['n_grounding']:>2d}/{r['n_hallucination']:<2d} "
            else:
                row += f" {'--':>6s}"
        print(row)

    # ── Save summary JSON ───────────────────────────────────────────
    summary_path = os.path.join(args.output, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump({k: {**v, "circuit_paths": {str(t): p
                                              for t, p in v["circuit_paths"].items()}}
                   for k, v in all_results.items()},
                  f, indent=2, default=str)
    logger.info(f"\nSummary saved to {summary_path}")

    # ── Generate CPA follow-up commands ─────────────────────────────
    cmd_path = os.path.join(args.output, "run_cpa_sweep.sh")
    cmd_text = generate_cpa_commands(all_results, args.output)
    with open(cmd_path, "w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("# Auto-generated by sweep_d_threshold.py\n")
        f.write("set -euo pipefail\n\n")
        f.write(cmd_text)
    os.chmod(cmd_path, 0o755)
    logger.info(f"CPA follow-up commands written to {cmd_path}")
    logger.info(f"Run: bash {cmd_path}")


if __name__ == "__main__":
    main()
