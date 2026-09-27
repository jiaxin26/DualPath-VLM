"""
Pathway Geometry: Raw Activation Collection (GPU)
==================================================

This script reuses the existing DirectionExtractor in
run_intervention_v2.py to capture per-component last-token activations
on correct vs hallucinating samples, but DUMPS THE RAW ACTIVATION
TENSORS (not just the mean-difference vector). Both the hallucination
pathway (d<0) and the grounding pathway (d>0) are collected, because
the contrast between the two is the central diagnostic.

Output: a single .pt file containing
  {
    "halluc": {(layer, comp_type): {"correct": Tensor[n_c, hidden],
                                    "halluc":  Tensor[n_h, hidden]}},
    "ground": { ... same shape ... },
    "meta":   {"model": str, "n_samples": int, ...}
  }

Downstream analysis is done on CPU by analyze_pathway_geometry.py.

Usage
-----
  PYTHONPATH=. python experiments/collect_pathway_geometry.py \\
    --model meta-llama/Llama-3.2-11B-Vision-Instruct \\
    --circuit results/llama_component/circuit.json \\
    --pope-path data/pope/ --image-dir data/coco/val2014/ \\
    --n-samples 1000 \\
    --output results/geometry_llama/

Runtime per model on H200 (1000 samples):
  Llama-3.2-V-11B   ~30 min
  Qwen3-VL-8B       ~50 min
  LLaVA-v1.6-7B     ~40 min
  InternVL3-8B      ~3 h
  InternVL3-14B     ~5 h
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, List, Tuple

import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("collect_geometry")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--circuit", type=str, required=True)
    parser.add_argument("--pope-path", type=str, default="data/pope/")
    parser.add_argument("--image-dir", type=str, default="data/coco/val2014/")
    parser.add_argument("--split", type=str, default="adversarial")
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Total POPE samples to process. Need enough "
                             "of BOTH classes for a stable SVD; 1000 yields "
                             "~130-180 hallucinating samples per model.")
    parser.add_argument("--min-d", type=float, default=0.3)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Reuse existing infrastructure.
    from experiments.run_patching import load_model, load_pope
    from experiments.run_intervention import load_halluc_targets
    from experiments.run_intervention_v2 import DirectionExtractor

    logger.info(f"Loading model: {args.model}")
    model, processor = load_model(args.model)

    pope_data = load_pope(args.pope_path, split=args.split,
                          n_samples=args.n_samples)
    logger.info(f"Loaded {len(pope_data)} POPE-{args.split} samples")

    # Both pathways. "visual" → grounding pathway (d>0).
    targets_halluc = load_halluc_targets(args.circuit, pathway="halluc",
                                         min_abs_d=args.min_d)
    targets_ground = load_halluc_targets(args.circuit, pathway="visual",
                                         min_abs_d=args.min_d)
    logger.info(f"Hallucination pathway: {len(targets_halluc)} components")
    logger.info(f"Grounding pathway:     {len(targets_ground)} components")

    # Single forward pass captures both pathways simultaneously to avoid
    # paying the model-forward cost twice. We register hooks on the union.
    all_targets = list(set(targets_halluc) | set(targets_ground))
    logger.info(f"Union of target components: {len(all_targets)}")

    extractor = DirectionExtractor(model, processor, args.model, all_targets)

    n_correct = 0
    n_halluc = 0
    n_skipped = 0

    for idx, sample in enumerate(pope_data):
        question = sample.get("text") or sample.get("question", "")
        gt_answer = (sample.get("label") or sample.get("answer", "")).lower()
        image_file = sample.get("image", "")
        image_path = os.path.join(args.image_dir, image_file)

        if not os.path.exists(image_path):
            n_skipped += 1
            continue

        img = Image.open(image_path).convert("RGB")
        prompt = question + " Answer with yes or no."

        is_correct = extractor.collect_sample(
            img, prompt, gt_answer, args.model)

        if is_correct:
            n_correct += 1
        else:
            n_halluc += 1

        if (idx + 1) % 50 == 0:
            logger.info(f"  [{idx+1}/{len(pope_data)}] "
                        f"correct={n_correct} halluc={n_halluc} "
                        f"skipped={n_skipped}")

    logger.info(f"Final tallies: correct={n_correct} halluc={n_halluc} "
                f"skipped={n_skipped}")

    # Re-package by pathway. We materialize the lists into stacked tensors
    # here (rather than later) because the ndarray is what analysis wants.
    def _pack(targets: List[Tuple[int, str]]) -> Dict:
        out = {}
        for key in targets:
            acts = extractor.activations.get(key, None)
            if acts is None:
                continue
            n_c = len(acts["correct"])
            n_h = len(acts["halluc"])
            if n_c < 5 or n_h < 5:
                logger.warning(f"  {key}: too few samples (c={n_c} h={n_h})")
                continue
            X_c = torch.cat(acts["correct"], dim=0).float()  # (n_c, hidden)
            X_h = torch.cat(acts["halluc"], dim=0).float()
            out[f"L{key[0]}.{key[1]}"] = {
                "correct": X_c.cpu(),
                "halluc": X_h.cpu(),
                "layer": int(key[0]),
                "comp_type": key[1],
            }
        return out

    payload = {
        "halluc": _pack(targets_halluc),
        "ground": _pack(targets_ground),
        "meta": {
            "model": args.model,
            "split": args.split,
            "n_samples": args.n_samples,
            "n_correct": n_correct,
            "n_halluc_samples": n_halluc,
            "n_skipped": n_skipped,
            "min_d": args.min_d,
        },
    }

    out_path = os.path.join(args.output, "geometry_acts.pt")
    torch.save(payload, out_path)
    logger.info(f"Saved raw activations to {out_path}")
    logger.info(f"  Hallucination components dumped: {len(payload['halluc'])}")
    logger.info(f"  Grounding components dumped:    {len(payload['ground'])}")


if __name__ == "__main__":
    main()
