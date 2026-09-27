"""
Directional Suppression Baseline (Section 3.6 / 5.5)
======================================================

Alternative intervention strategies tested as baselines against
uniform scaling. Reported as negative results in the paper:
directional suppression does not yield meaningful hallucination
reduction, consistent with the hallucination signal not being
well captured by a single linear direction.

Implements:

  1. Directional Suppression:
     For each hallucination-pathway component, compute a "hallucination
     direction" vector: mean(activation | halluc) - mean(activation | correct).
     During inference, project out this direction:
       h' = h - α * (h · d̂) * d̂
     where d̂ is the unit hallucination direction and α is a strength parameter.

  2. Optimal Per-Component Scaling (grid search):
     For each component, find optimal scale factor s ∈ {0.0, 0.1, ..., 1.0}
     on a held-out validation set.

Evaluation includes:
  - Baseline (no intervention)
  - Uniform scaling (main method from run_intervention.py, for comparison)
  - Directional suppression
  - Optimal per-component scaling
  - Random control (matched count, 5 seeds)

Usage:
  # Step 1: Extract hallucination directions from patching data
  PYTHONPATH=. python experiments/run_intervention_v2.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --circuit results/qwen3vl_component/circuit.json \
    --mode extract-directions \
    --pope-path data/pope/ --image-dir data/coco/val2014/ \
    --n-samples 250 \
    --output results/intervention_v2_qwen/

  # Step 2: Evaluate all intervention modes
  PYTHONPATH=. python experiments/run_intervention_v2.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --circuit results/qwen3vl_component/circuit.json \
    --directions results/intervention_v2_qwen/directions.pt \
    --mode evaluate \
    --pope-path data/pope/ --image-dir data/coco/val2014/ \
    --n-samples 500 \
    --output results/intervention_v2_qwen/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("intervention_v2")


# ════════════════════════════════════════════════════════════════════
# DIRECTION EXTRACTION
# ════════════════════════════════════════════════════════════════════

class DirectionExtractor:
    """Collect per-component activations from correct vs hallucinating samples,
    then compute hallucination direction vectors."""

    def __init__(self, model, processor, model_id: str,
                 targets: List[Tuple[int, str]]):
        self.model = model
        self.processor = processor
        self.model_id = model_id
        self.targets = targets
        self.layers = self._get_decoder_layers(model_id)

        # Accumulators: {(layer, comp): {"correct": [tensors], "halluc": [tensors]}}
        self.activations: Dict[Tuple[int, str], Dict[str, List[torch.Tensor]]] = {
            (l, c): {"correct": [], "halluc": []}
            for l, c in targets
        }

    def _get_decoder_layers(self, model_id: str) -> nn.ModuleList:
        model_lower = model_id.lower()
        if "qwen" in model_lower:
            from patching.qwen_patcher import _get_decoder_layers
            return _get_decoder_layers(self.model)
        elif "llava" in model_lower:
            from patching.llava_patcher import _get_decoder_layers
            return _get_decoder_layers(self.model)
        elif "internvl" in model_lower:
            from patching.internvl_patcher import _get_decoder_layers
            return _get_decoder_layers(self.model)
        elif "llama" in model_lower or "vicuna" in model_lower:
            from patching.llama_patcher import _get_decoder_layers
            return _get_decoder_layers(self.model)
        else:
            from patching.qwen_patcher import _get_decoder_layers
            return _get_decoder_layers(self.model)

    @torch.no_grad()
    def collect_sample(self, image, prompt: str,
                        gt_answer: str, model_id: str):
        """Run one sample, capture activations at target components,
        classify as correct/halluc based on model output."""
        from experiments.run_intervention import _prepare_inputs

        device = next(self.model.parameters()).device
        inputs = _prepare_inputs(self.processor, self.model,
                                  model_id, image, prompt)

        # Register capture hooks
        handles = []
        captured = {}

        for layer_idx, comp_type in self.targets:
            layer = self.layers[layer_idx]
            attn_module = getattr(layer, "self_attn", None) or \
                          getattr(layer, "cross_attn", None)

            if comp_type == "attn":
                if attn_module is None:
                    continue
                def make_hook(key=(layer_idx, "attn")):
                    def hook(module, inp, output):
                        out = output[0] if isinstance(output, tuple) else output
                        # Mean-pool over sequence, take last token
                        captured[key] = out[:, -1, :].detach().cpu()
                    return hook
                handles.append(
                    attn_module.register_forward_hook(make_hook()))
            elif comp_type == "mlp":
                if not hasattr(layer, "mlp"):
                    continue
                def make_hook(key=(layer_idx, "mlp")):
                    def hook(module, inp, output):
                        captured[key] = output[:, -1, :].detach().cpu()
                    return hook
                handles.append(
                    layer.mlp.register_forward_hook(make_hook()))

        try:
            output_ids = self.model.generate(
                **inputs, max_new_tokens=5,
                do_sample=False, temperature=None, top_p=None)
        finally:
            for h in handles:
                h.remove()

        # Parse model answer
        input_len = inputs["input_ids"].shape[1]
        generated = self.processor.decode(
            output_ids[0][input_len:],
            skip_special_tokens=True).strip().lower()
        pred = "yes" if "yes" in generated[:10] else "no"
        is_correct = (pred == gt_answer.lower())

        category = "correct" if is_correct else "halluc"

        for key, tensor in captured.items():
            self.activations[key][category].append(tensor)

        return is_correct

    def compute_directions(self) -> Dict[Tuple[int, str], dict]:
        """Compute hallucination direction for each component.

        Direction = mean(halluc activations) - mean(correct activations)
        Normalized to unit vector.

        Returns:
            Dict mapping (layer, comp) → {
                "direction": unit vector (1, hidden),
                "magnitude": |mean_halluc - mean_correct|,
                "n_correct": int,
                "n_halluc": int,
            }
        """
        directions = {}

        for key, acts in self.activations.items():
            n_c = len(acts["correct"])
            n_h = len(acts["halluc"])

            if n_c < 3 or n_h < 3:
                logger.warning(f"  {key}: too few samples "
                               f"(correct={n_c}, halluc={n_h}), skipping")
                continue

            mean_correct = torch.stack(acts["correct"]).mean(dim=0)  # (1, hidden)
            mean_halluc = torch.stack(acts["halluc"]).mean(dim=0)

            direction = mean_halluc - mean_correct  # (1, hidden)
            magnitude = direction.norm().item()

            if magnitude < 1e-8:
                logger.warning(f"  {key}: near-zero direction magnitude, skipping")
                continue

            unit_dir = direction / direction.norm()

            directions[key] = {
                "direction": unit_dir,
                "magnitude": magnitude,
                "n_correct": n_c,
                "n_halluc": n_h,
                "mean_correct_norm": mean_correct.norm().item(),
                "mean_halluc_norm": mean_halluc.norm().item(),
            }

            logger.info(f"  {key}: |dir|={magnitude:.4f}, "
                        f"n_correct={n_c}, n_halluc={n_h}")

        return directions


# ════════════════════════════════════════════════════════════════════
# INTERVENTION HOOKS
# ════════════════════════════════════════════════════════════════════

class DirectionalIntervention:
    """Project out the hallucination direction from component outputs."""

    def __init__(self, model, model_id: str,
                 directions: Dict[Tuple[int, str], dict],
                 alpha: float = 1.0):
        """
        Args:
            directions: {(layer, comp_type): {"direction": tensor, ...}}
            alpha: suppression strength (1.0 = full projection, 0.5 = half)
        """
        self.model = model
        self.alpha = alpha
        self.handles = []

        model_lower = model_id.lower()
        if "qwen" in model_lower:
            from patching.qwen_patcher import _get_decoder_layers
        elif "llava" in model_lower:
            from patching.llava_patcher import _get_decoder_layers
        elif "internvl" in model_lower:
            from patching.internvl_patcher import _get_decoder_layers
        elif "llama" in model_lower or "vicuna" in model_lower:
            from patching.llama_patcher import _get_decoder_layers
        else:
            from patching.qwen_patcher import _get_decoder_layers

        self.layers = _get_decoder_layers(model)
        self.directions = directions
        self.device = next(model.parameters()).device

    def attach(self):
        """Register directional suppression hooks."""
        self.detach()

        for (layer_idx, comp_type), dir_info in self.directions.items():
            layer = self.layers[layer_idx]
            direction = dir_info["direction"].to(self.device)
            alpha = self.alpha
            attn_module = getattr(layer, "self_attn", None) or \
                          getattr(layer, "cross_attn", None)

            if comp_type == "attn":
                if attn_module is None:
                    continue
                def make_hook(d=direction, a=alpha):
                    def hook(module, inp, output):
                        out = output[0] if isinstance(output, tuple) else output
                        # Lazy device cast: layer may live on a different
                        # GPU than where `d` was originally stored.
                        d_local = d.to(device=out.device, dtype=out.dtype)
                        # Project out hallucination direction
                        # proj = (out · d̂) * d̂
                        proj = (out * d_local).sum(dim=-1, keepdim=True) * d_local
                        out_new = out - a * proj
                        if isinstance(output, tuple):
                            return (out_new,) + output[1:]
                        return out_new
                    return hook
                self.handles.append(
                    attn_module.register_forward_hook(make_hook()))

            elif comp_type == "mlp":
                if not hasattr(layer, "mlp"):
                    continue
                def make_hook(d=direction, a=alpha):
                    def hook(module, inp, output):
                        d_local = d.to(device=output.device, dtype=output.dtype)
                        proj = (output * d_local).sum(dim=-1, keepdim=True) * d_local
                        return output - a * proj
                    return hook
                self.handles.append(
                    layer.mlp.register_forward_hook(make_hook()))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


class OptimalScalingIntervention:
    """Per-component optimal scale factors (from grid search)."""

    def __init__(self, model, model_id: str,
                 scale_map: Dict[Tuple[int, str], float]):
        """
        Args:
            scale_map: {(layer, comp_type): optimal_scale_factor}
        """
        self.handles = []

        model_lower = model_id.lower()
        if "qwen" in model_lower:
            from patching.qwen_patcher import _get_decoder_layers
        elif "llava" in model_lower:
            from patching.llava_patcher import _get_decoder_layers
        elif "internvl" in model_lower:
            from patching.internvl_patcher import _get_decoder_layers
        elif "llama" in model_lower or "vicuna" in model_lower:
            from patching.llama_patcher import _get_decoder_layers
        else:
            from patching.qwen_patcher import _get_decoder_layers

        self.layers = _get_decoder_layers(model)
        self.scale_map = scale_map

    def attach(self):
        self.detach()
        for (layer_idx, comp_type), scale in self.scale_map.items():
            layer = self.layers[layer_idx]
            attn_module = getattr(layer, "self_attn", None) or \
                          getattr(layer, "cross_attn", None)

            if comp_type == "attn":
                if attn_module is None:
                    continue
                def make_hook(s=scale):
                    def hook(module, inp, output):
                        if isinstance(output, tuple):
                            return (output[0] * s,) + output[1:]
                        return output * s
                    return hook
                self.handles.append(
                    attn_module.register_forward_hook(make_hook()))

            elif comp_type == "mlp":
                if not hasattr(layer, "mlp"):
                    continue
                def make_hook(s=scale):
                    def hook(module, inp, output):
                        return output * s
                    return hook
                self.handles.append(
                    layer.mlp.register_forward_hook(make_hook()))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ════════════════════════════════════════════════════════════════════
# GRID SEARCH FOR OPTIMAL SCALES
# ════════════════════════════════════════════════════════════════════

def grid_search_scales(model, processor, model_id: str,
                        targets: List[Tuple[int, str]],
                        val_data: list, image_dir: str,
                        scales: List[float] = None,
                        ) -> Dict[Tuple[int, str], float]:
    """Find optimal per-component scale factor on validation set.

    For each target component, evaluate all scale values and pick the
    one that minimizes hallucination rate while maintaining accuracy.
    """
    from experiments.run_intervention import (
        evaluate_pope, ComponentIntervention)

    if scales is None:
        scales = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    logger.info(f"Grid search: {len(targets)} components × {len(scales)} scales "
                f"on {len(val_data)} validation samples")

    best_scales = {}

    for target in targets:
        layer_idx, comp_type = target
        best_score = -1e9
        best_s = 1.0

        for s in scales:
            intervention = ComponentIntervention(
                model, model_id, [target], scale=s)
            intervention.attach()
            try:
                metrics = evaluate_pope(
                    model, processor, model_id,
                    val_data, image_dir, max_new_tokens=5)
            finally:
                intervention.detach()

            # Score: maximize accuracy - halluc_rate
            score = metrics["accuracy"] - metrics["halluc_rate"]
            if score > best_score:
                best_score = score
                best_s = s

        best_scales[target] = best_s
        logger.info(f"  L{layer_idx}.{comp_type}: best scale = {best_s}")

    return best_scales


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Intervention V2: directional suppression + optimal scaling")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--circuit", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True,
                        choices=["extract-directions", "evaluate", "full"],
                        help="'extract-directions': collect activations and "
                             "compute halluc directions. "
                             "'evaluate': run all intervention modes. "
                             "'full': both.")
    parser.add_argument("--directions", type=str, default=None,
                        help="Path to directions.pt (for evaluate mode)")
    parser.add_argument("--pope-path", type=str, default="data/pope/")
    parser.add_argument("--image-dir", type=str, default="data/coco/val2014/")
    parser.add_argument("--split", type=str, default="adversarial")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--min-d", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Directional suppression strength")
    parser.add_argument("--n-val", type=int, default=100,
                        help="Validation samples for grid search")
    parser.add_argument("--skip-optimal-scaling", action="store_true",
                        help="Skip Condition 4 (per-component grid search). "
                             "This condition is not used in the paper and "
                             "dominates runtime; skipping is recommended.")
    parser.add_argument("--output", type=str, default="results/intervention_v2/")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    from experiments.run_patching import load_model, load_pope
    from experiments.run_intervention import (
        evaluate_pope, ComponentIntervention, load_halluc_targets,
        make_random_targets, _prepare_inputs)

    model, processor = load_model(args.model)
    pope_data = load_pope(args.pope_path, split=args.split,
                          n_samples=args.n_samples)

    targets = load_halluc_targets(args.circuit, pathway="halluc",
                                   min_abs_d=args.min_d)

    # Determine n_layers
    model_lower = args.model.lower()
    if "qwen" in model_lower:
        from patching.qwen_patcher import _get_decoder_layers
    elif "llava" in model_lower:
        from patching.llava_patcher import _get_decoder_layers
    elif "internvl" in model_lower:
        from patching.internvl_patcher import _get_decoder_layers
    elif "llama" in model_lower or "vicuna" in model_lower:
        from patching.llama_patcher import _get_decoder_layers
    else:
        from patching.qwen_patcher import _get_decoder_layers
    n_layers = len(_get_decoder_layers(model))

    directions = None

    # ── Phase 1: Extract directions ──────────────────────────────
    if args.mode in ["extract-directions", "full"]:
        logger.info("=" * 60)
        logger.info("PHASE 1: Extracting hallucination directions")
        logger.info("=" * 60)

        extractor = DirectionExtractor(model, processor, args.model, targets)
        n_extract = min(args.n_samples, 250)

        for idx, sample in enumerate(pope_data[:n_extract]):
            question = sample.get("text") or sample.get("question", "")
            gt_answer = (sample.get("label") or sample.get("answer", "")).lower()
            image_file = sample.get("image", "")
            image_path = os.path.join(args.image_dir, image_file)

            if not os.path.exists(image_path):
                continue

            img = Image.open(image_path).convert("RGB")
            prompt = question + " Answer with yes or no."

            is_correct = extractor.collect_sample(
                img, prompt, gt_answer, args.model)

            if (idx + 1) % 50 == 0:
                status = "OK" if is_correct else "HALLUC"
                logger.info(f"  [{idx+1}/{n_extract}] {status}")

        directions = extractor.compute_directions()

        # Save directions
        dir_path = os.path.join(args.output, "directions.pt")
        save_directions = {}
        for key, info in directions.items():
            save_directions[f"L{key[0]}.{key[1]}"] = {
                "direction": info["direction"],
                "magnitude": info["magnitude"],
                "n_correct": info["n_correct"],
                "n_halluc": info["n_halluc"],
                "layer": key[0],
                "comp_type": key[1],
            }
        torch.save(save_directions, dir_path)
        logger.info(f"Saved directions to {dir_path}")

    # ── Load directions if not extracted ─────────────────────────
    if args.mode == "evaluate" and directions is None:
        if args.directions is None:
            raise ValueError("--directions required in evaluate mode")
        raw = torch.load(args.directions, map_location="cpu",
                         weights_only=False)
        directions = {}
        for name, info in raw.items():
            key = (info["layer"], info["comp_type"])
            directions[key] = info
        logger.info(f"Loaded {len(directions)} directions from {args.directions}")

    # ── Phase 2: Evaluate ────────────────────────────────────────
    if args.mode in ["evaluate", "full"]:
        logger.info("=" * 60)
        logger.info("PHASE 2: Evaluation")
        logger.info("=" * 60)

        # Split protocol (matching paper Section 4):
        #   first n_val samples = selection set (hyperparameter tuning only)
        #   remaining samples    = held-out test set (reported numbers)
        val_data = pope_data[:args.n_val]
        test_data = pope_data[args.n_val:]
        if len(test_data) < 50:
            logger.warning(
                f"Test set only {len(test_data)} samples — falling back "
                f"to full pope_data (not held-out).")
            test_data = pope_data
        logger.info(f"Selection set: {len(val_data)} | "
                    f"Test set: {len(test_data)}")

        results_all = {}

        # Condition 1: Baseline (test set)
        logger.info("CONDITION 1: Baseline (test)")
        baseline = evaluate_pope(model, processor, args.model,
                                  test_data, args.image_dir)
        results_all["baseline"] = baseline
        logger.info(f"  acc={baseline['accuracy']:.4f}  "
                    f"halluc={baseline['halluc_rate']:.4f}")

        # Baseline on val (for α selection below)
        logger.info("  (also: baseline on val set for α selection)")
        baseline_val = evaluate_pope(model, processor, args.model,
                                     val_data, args.image_dir)
        results_all["baseline_val"] = baseline_val

        # Condition 2: V1 — uniform scaling (original method, test set)
        logger.info("\nCONDITION 2: V1 uniform scaling (s=0.5, test)")
        v1 = ComponentIntervention(model, args.model, targets, scale=0.5)
        v1.attach()
        try:
            v1_metrics = evaluate_pope(model, processor, args.model,
                                        test_data, args.image_dir)
        finally:
            v1.detach()
        results_all["v1_uniform_0.5"] = v1_metrics
        logger.info(f"  acc={v1_metrics['accuracy']:.4f}  "
                    f"halluc={v1_metrics['halluc_rate']:.4f}")

        # Condition 3: V2 — directional suppression
        # α is now selected on val set, reported on test set.
        if directions:
            logger.info("\nCONDITION 3: V2 directional — α grid on val")
            dir_val_results = {}
            for alpha in [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]:
                v2_dir = DirectionalIntervention(
                    model, args.model, directions, alpha=alpha)
                v2_dir.attach()
                try:
                    m_val = evaluate_pope(model, processor, args.model,
                                          val_data, args.image_dir)
                finally:
                    v2_dir.detach()
                dir_val_results[alpha] = m_val
                results_all[f"v2_directional_a{alpha}_val"] = m_val
                logger.info(f"  α={alpha}: val acc={m_val['accuracy']:.3f}, "
                            f"hall={m_val['halluc_rate']:.3f}")

            # Select α: minimize val halluc rate s.t. val accuracy drop ≤ 2pp
            acc_budget = 0.02
            feasible = {a: r for a, r in dir_val_results.items()
                        if (baseline_val["accuracy"] - r["accuracy"])
                            <= acc_budget}
            if feasible:
                alpha_sel = min(feasible.keys(),
                                key=lambda a: dir_val_results[a]["halluc_rate"])
                sel_note = f"feasible ({len(feasible)}/3 met ≤2pp budget)"
            else:
                alpha_sel = min(dir_val_results.keys(),
                                key=lambda a: dir_val_results[a]["halluc_rate"])
                sel_note = "NO config met ≤2pp budget; picking lowest halluc"
            logger.info(f"  → Selected α={alpha_sel} ({sel_note})")

            # Apply selected to test set
            v2_dir = DirectionalIntervention(
                model, args.model, directions, alpha=alpha_sel)
            v2_dir.attach()
            try:
                m_test = evaluate_pope(model, processor, args.model,
                                       test_data, args.image_dir)
            finally:
                v2_dir.detach()
            results_all["v2_directional_selected"] = {
                "alpha": alpha_sel,
                "selection_note": sel_note,
                **m_test,
            }
            logger.info(f"  Test (α={alpha_sel}): "
                        f"acc={m_test['accuracy']:.4f}, "
                        f"hall={m_test['halluc_rate']:.4f}")

        # Condition 4: V2 — optimal per-component scaling
        # This condition is not used in the paper and is expensive
        # (grid search over scale values per component).  Skip by default.
        if not args.skip_optimal_scaling:
            logger.info("\nCONDITION 4: V2 optimal scaling (grid search on val)")
            optimal_scales = grid_search_scales(
                model, processor, args.model, targets,
                val_data, args.image_dir)

            logger.info("  Optimal scales:")
            for (l, c), s in sorted(optimal_scales.items()):
                logger.info(f"    L{l}.{c}: {s:.1f}")

            v2_opt = OptimalScalingIntervention(model, args.model, optimal_scales)
            v2_opt.attach()
            try:
                v2_opt_metrics = evaluate_pope(model, processor, args.model,
                                                test_data, args.image_dir)
            finally:
                v2_opt.detach()
            results_all["v2_optimal_scaling"] = v2_opt_metrics
            logger.info(f"  acc={v2_opt_metrics['accuracy']:.4f}  "
                        f"halluc={v2_opt_metrics['halluc_rate']:.4f}")
        else:
            logger.info("\nCONDITION 4: SKIPPED (--skip-optimal-scaling)")
            optimal_scales = {}

        # Condition 5: Random control (5 seeds, evaluated on test set)
        logger.info("\nCONDITION 5: Random control (test set, 5 seeds)")
        random_metrics_list = []
        for seed in range(5):
            rand_targets = make_random_targets(n_layers, len(targets), seed=seed)
            rand_int = ComponentIntervention(
                model, args.model, rand_targets, scale=0.5)
            rand_int.attach()
            try:
                rm = evaluate_pope(model, processor, args.model,
                                    test_data, args.image_dir)
                random_metrics_list.append(rm)
            finally:
                rand_int.detach()

        # Average random results
        avg_random = {}
        for key in random_metrics_list[0]:
            vals = [m[key] for m in random_metrics_list
                    if isinstance(m[key], (int, float))]
            if vals:
                avg_random[key] = float(np.mean(vals))
        results_all["random_control_avg"] = avg_random
        logger.info(f"  acc={avg_random.get('accuracy',0):.4f}  "
                    f"halluc={avg_random.get('halluc_rate',0):.4f}")

        # ── Summary table ────────────────────────────────────────
        print(f"\n{'='*70}")
        print(f"INTERVENTION V2 RESULTS — {args.model.split('/')[-1]}")
        print(f"{'='*70}")
        print(f"{'Condition':<30s} {'Acc':>8s} {'Halluc':>8s} {'ΔAcc':>8s} "
              f"{'ΔHalluc':>8s}")
        print("─" * 70)

        base_acc = baseline["accuracy"]
        base_hr = baseline["halluc_rate"]

        for name, metrics in results_all.items():
            acc = metrics.get("accuracy", 0)
            hr = metrics.get("halluc_rate", 0)
            d_acc = acc - base_acc
            d_hr = hr - base_hr
            print(f"{name:<30s} {acc:>8.4f} {hr:>8.4f} "
                  f"{d_acc:>+8.4f} {d_hr:>+8.4f}")

        # Save
        out_path = os.path.join(args.output, "intervention_v2_results.json")
        with open(out_path, "w") as f:
            json.dump(results_all, f, indent=2)
        logger.info(f"\nSaved to {out_path}")

        # Save optimal scales
        scales_path = os.path.join(args.output, "optimal_scales.json")
        with open(scales_path, "w") as f:
            json.dump({f"L{l}.{c}": s
                       for (l, c), s in optimal_scales.items()}, f, indent=2)


if __name__ == "__main__":
    main()