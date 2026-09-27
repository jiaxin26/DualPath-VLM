"""
Top-k Evaluation with Per-Sample Predictions (for bootstrap analysis)
======================================================================

Standalone script that evaluates a set of k values on POPE and saves
the per-sample (gt, pred) for each k. These predictions are then
consumed by bootstrap_topk.py to estimate the variance of k-selection
across different random 100-sample selection sets.

This is separate from run_intervention.py to avoid patching the main
intervention pipeline. Produces exactly the same evaluation as
run_intervention.py's top-k ablation, but also dumps per-sample preds.

Usage:
  # LLaVA top-k evaluation with per-sample tracking (~6h on H200)
  PYTHONPATH=. python experiments/eval_topk_per_sample.py \
    --model llava-hf/llava-v1.6-mistral-7b-hf \
    --circuit results/llava_component/circuit.json \
    --pope-path data/pope/ --image-dir data/coco/val2014/ \
    --n-samples 500 \
    --k-values 3 5 8 10 15 20 30 \
    --scale 0.5 \
    --output results/robustness/topk_per_sample_llava/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_topk_per_sample")


def evaluate_with_per_sample(model, processor, model_id: str,
                             samples: List[dict], image_dir: str,
                             max_new_tokens: int = 5) -> Dict:
    """Evaluate on samples and return both aggregate metrics AND per-sample
    predictions.

    This is a drop-in replacement for run_intervention.evaluate_samples()
    with the addition of a 'per_sample' list in the returned dict.
    """
    import torch
    from PIL import Image
    # Reuse helpers from run_intervention to keep identical semantics
    from experiments.run_intervention import (_resolve_image_path,
                                               _prepare_inputs)

    tp = fp = tn = fn = errors = 0
    per_sample = []

    for idx, sample in enumerate(samples):
        q = sample.get("text") or sample.get("question", "")
        gt = (sample.get("label") or sample.get("answer", "")).strip().lower()
        image_file = sample.get("image", "")
        image_path = _resolve_image_path(image_dir, image_file)

        if image_path is None:
            errors += 1
            continue
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            errors += 1
            continue

        prompt = q + " Answer with yes or no."
        try:
            inputs = _prepare_inputs(processor, model, model_id, image, prompt)
        except Exception as e:
            errors += 1
            continue

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None, top_p=None)
        input_len = inputs["input_ids"].shape[1]
        generated = processor.decode(
            output_ids[0][input_len:], skip_special_tokens=True).strip().lower()
        pred = "yes" if "yes" in generated[:10] else "no"

        if pred == "yes" and gt == "yes":   tp += 1
        elif pred == "yes" and gt == "no":  fp += 1
        elif pred == "no" and gt == "no":   tn += 1
        else:                                fn += 1

        per_sample.append({
            "idx": idx, "gt": gt, "pred": pred,
            "question": q, "image": image_file,
        })

        if (idx + 1) % 100 == 0:
            total_so_far = tp + fp + tn + fn
            acc = (tp + tn) / total_so_far if total_so_far > 0 else 0
            logger.info(f"    [{idx+1}/{len(samples)}] acc={acc:.3f}")

    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    halluc_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": acc, "halluc_rate": halluc_rate,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total": total, "errors": errors,
        "per_sample": per_sample,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate top-k values with per-sample predictions")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--circuit", type=str, required=True)
    parser.add_argument("--pope-path", type=str, default="data/pope/")
    parser.add_argument("--image-dir", type=str, default="data/coco/val2014/")
    parser.add_argument("--split", type=str, default="adversarial")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--k-values", nargs="+", type=int,
                        default=[3, 5, 8, 10, 15, 20, 30],
                        help="Top-k values to evaluate")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="Scale factor for pathway suppression")
    parser.add_argument("--min-d", type=float, default=0.3)
    parser.add_argument("--pathway", type=str, default="halluc",
                        choices=["halluc", "visual"])
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    from experiments.run_patching import load_model, load_pope
    from experiments.run_intervention import (ComponentIntervention,
                                               load_halluc_targets)

    model, processor = load_model(args.model)
    pope_data = load_pope(args.pope_path, split=args.split,
                          n_samples=args.n_samples)
    logger.info(f"Loaded {len(pope_data)} POPE-{args.split} samples")

    # ── Baseline (no intervention) ──────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("BASELINE (no intervention)")
    logger.info("=" * 60)
    t0 = time.time()
    baseline = evaluate_with_per_sample(
        model, processor, args.model, pope_data, args.image_dir)
    logger.info(f"  acc={baseline['accuracy']:.3f}  "
                f"halluc={baseline['halluc_rate']:.3f}  "
                f"time={(time.time()-t0)/60:.1f}min")
    with open(os.path.join(args.output, "baseline.pkl"), "wb") as f:
        pickle.dump(baseline, f)

    # ── Each k value ────────────────────────────────────────────────
    all_results = {"baseline": {k: baseline[k]
                                for k in ["accuracy", "halluc_rate",
                                          "tp", "fp", "tn", "fn", "total"]}}

    for k in args.k_values:
        logger.info("\n" + "=" * 60)
        logger.info(f"TOP-K = {k}   (scale={args.scale})")
        logger.info("=" * 60)

        targets = load_halluc_targets(
            args.circuit, pathway=args.pathway,
            min_abs_d=args.min_d, top_k=k)
        if not targets:
            logger.warning(f"No targets for k={k}, skipping")
            continue

        intervention = ComponentIntervention(
            model, args.model, targets, scale=args.scale)
        intervention.attach()
        try:
            t0 = time.time()
            res = evaluate_with_per_sample(
                model, processor, args.model, pope_data, args.image_dir)
            logger.info(f"  acc={res['accuracy']:.3f}  "
                        f"halluc={res['halluc_rate']:.3f}  "
                        f"time={(time.time()-t0)/60:.1f}min")
        finally:
            intervention.detach()

        # Save per-sample predictions for bootstrap
        out_pkl = os.path.join(args.output, f"top-{k}.pkl")
        with open(out_pkl, "wb") as f:
            pickle.dump({
                "k": k, "scale": args.scale,
                "n_targets": len(targets),
                "targets": targets,
                **res,
            }, f)

        all_results[f"k={k}"] = {kk: res[kk]
                                  for kk in ["accuracy", "halluc_rate",
                                             "tp", "fp", "tn", "fn", "total"]}

    # ── Aggregate summary ───────────────────────────────────────────
    summary_path = os.path.join(args.output, "aggregate_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "model": args.model,
            "circuit": args.circuit,
            "scale": args.scale,
            "n_samples": args.n_samples,
            "results": all_results,
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("AGGREGATE SUMMARY")
    print("=" * 60)
    print(f"  {'config':<12s}  {'acc':>6s}  {'halluc':>7s}")
    for cfg, r in all_results.items():
        print(f"  {cfg:<12s}  {r['accuracy']:>6.3f}  {r['halluc_rate']:>7.3f}")

    print(f"\nPer-sample pickles saved to: {args.output}")
    print(f"Run bootstrap analysis with:\n"
          f"  python experiments/bootstrap_topk.py "
          f"--results-dir {args.output}")


if __name__ == "__main__":
    main()
