"""
Targeted Intervention: Suppress Hallucination Pathways
=======================================================

Suppresses hallucination-pathway components by scaling their outputs
during inference (Section 3.6 of the paper).

Supports:
  - Multi-benchmark evaluation (POPE + MME + AMBER)
  - Top-k ablation: suppress only the top-k strongest components
    (addresses accuracy degradation from suppressing dispersed circuits)
  - Multiple scale factors with automatic selection
  - Selection/evaluation split for unbiased reporting

Three conditions per model per benchmark:
  1. Baseline:     no intervention
  2. Targeted:     suppress halluc-pathway components (negative-d)
  3. Random ctrl:  suppress same number of random components (avg 5 seeds)

Usage:
  # Single benchmark (backward-compatible)
  PYTHONPATH=. python experiments/run_intervention.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --circuit results/qwen3vl_component/circuit.json \
    --pope-path data/pope/ --image-dir data/coco/val2014/ \
    --split adversarial --n-samples 500 \
    --output results/intervention_qwen/

  # Multi-benchmark evaluation
  PYTHONPATH=. python experiments/run_intervention.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --circuit results/qwen3vl_component/circuit.json \
    --benchmarks pope mme \
    --pope-path data/pope/ --mme-path data/mme/ \
    --image-dir data/coco/val2014/ \
    --mme-image-dir data/mme/images/ \
    --n-samples 500 \
    --output results/intervention_qwen_multi/

  # Top-k ablation (for models with dispersed circuits like LLaVA)
  PYTHONPATH=. python experiments/run_intervention.py \
    --model llava-hf/llava-v1.6-mistral-7b-hf \
    --circuit results/llava_component/circuit.json \
    --topk-ablation 3 5 10 15 20 \
    --pope-path data/pope/ --image-dir data/coco/val2014/ \
    --n-samples 500 \
    --output results/intervention_llava_topk/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("intervention")


# ════════════════════════════════════════════════════════════════════
# INTERVENTION HOOKS
# ════════════════════════════════════════════════════════════════════

class ComponentIntervention:
    """Register hooks that scale specific (layer, component) outputs."""

    def __init__(self, model, model_id: str,
                 targets: List[Tuple[int, str]],
                 scale: float = 0.0):
        self.model = model
        self.targets = targets
        self.scale = scale
        self.handles: List[torch.utils.hooks.RemovableHook] = []
        self.layers = self._get_decoder_layers(model_id)
        logger.info(f"Intervention: {len(targets)} components, scale={scale}")

    def _get_decoder_layers(self, model_id: str) -> nn.ModuleList:
        model_lower = model_id.lower()
        if "qwen" in model_lower:
            from patching.qwen_patcher import _get_decoder_layers
            return _get_decoder_layers(self.model)
        elif "llava" in model_lower and "llama" not in model_lower:
            from patching.llava_patcher import _get_decoder_layers
            return _get_decoder_layers(self.model)
        elif "internvl" in model_lower:
            from patching.internvl_patcher import _get_decoder_layers
            return _get_decoder_layers(self.model)
        elif "llama" in model_lower or "mllama" in model_lower:
            from patching.llama_patcher import _get_decoder_layers
            return _get_decoder_layers(self.model)
        else:
            from patching.qwen_patcher import _get_decoder_layers
            return _get_decoder_layers(self.model)

    def attach(self):
        """Register forward hooks for all target components."""
        self.detach()
        for layer_idx, comp_type in self.targets:
            layer = self.layers[layer_idx]
            scale = self.scale
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
                h = attn_module.register_forward_hook(make_hook())

            elif comp_type == "mlp":
                if not hasattr(layer, "mlp"):
                    continue
                def make_hook(s=scale):
                    def hook(module, inp, output):
                        return output * s
                    return hook
                h = layer.mlp.register_forward_hook(make_hook())
            else:
                continue

            self.handles.append(h)

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ════════════════════════════════════════════════════════════════════
# EVALUATION
# ════════════════════════════════════════════════════════════════════

def _resolve_image_path(image_dir: str, image_file: str) -> Optional[str]:
    """Try multiple strategies to find the image file.

    Handles:
      - Normal filename: "COCO_val2014_000000123.jpg"
      - Numeric ID (AMBER): "1" -> try "1.jpg", "1.png"
      - URL: "https://..." -> skip (needs download)
      - Nested path: "image/1.jpg" -> try as-is, then basename
    """
    if not image_file:
        return None

    # Skip URLs
    if image_file.startswith("http://") or image_file.startswith("https://"):
        return None

    # Strategy 1: Direct join
    path = os.path.join(image_dir, image_file)
    if os.path.exists(path):
        return path

    # Strategy 2: Try adding common extensions (for numeric IDs like AMBER)
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        path = os.path.join(image_dir, image_file + ext)
        if os.path.exists(path):
            return path

    # Strategy 3: Try basename only (strips subdirectories)
    basename = os.path.basename(image_file)
    if basename != image_file:
        path = os.path.join(image_dir, basename)
        if os.path.exists(path):
            return path
        for ext in [".jpg", ".jpeg", ".png"]:
            path = os.path.join(image_dir, basename + ext)
            if os.path.exists(path):
                return path

    # Strategy 4: Try zero-padded numeric (e.g., "1" -> "00001.jpg")
    if image_file.isdigit():
        for pad in [4, 5, 6, 8, 12]:
            padded = image_file.zfill(pad)
            for ext in [".jpg", ".png"]:
                path = os.path.join(image_dir, padded + ext)
                if os.path.exists(path):
                    return path

    return None


def evaluate_samples(model, processor, model_id: str,
                     samples: List[dict], image_dir: str,
                     max_new_tokens: int = 5) -> Dict[str, float]:
    """Run yes/no evaluation on samples in the unified schema.

    Works with any benchmark that uses the unified format from benchmarks.py.
    """
    tp, fp, tn, fn = 0, 0, 0, 0
    errors = 0
    _logged_missing = 0  # only log first few missing images

    for idx, sample in enumerate(samples):
        question = sample.get("text") or sample.get("question", "")
        gt = (sample.get("label") or sample.get("answer", "")).strip().lower()
        image_file = sample.get("image", "")

        image_path = _resolve_image_path(image_dir, image_file)

        if image_path is None:
            errors += 1
            if _logged_missing < 3:
                logger.warning(
                    f"Image not found: dir='{image_dir}', "
                    f"file='{image_file[:80]}' "
                    f"(benchmark={sample.get('benchmark', '?')})")
                _logged_missing += 1
            elif _logged_missing == 3:
                logger.warning("... suppressing further missing-image warnings")
                _logged_missing += 1
            continue

        img = Image.open(image_path).convert("RGB")
        prompt = question + " Answer with yes or no."

        try:
            inputs = _prepare_inputs(processor, model, model_id, img, prompt)
        except Exception as e:
            logger.error(f"Input prep error at {idx}: {e}")
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

        if pred == "yes" and gt == "yes":
            tp += 1
        elif pred == "yes" and gt == "no":
            fp += 1
        elif pred == "no" and gt == "no":
            tn += 1
        else:
            fn += 1

        if (idx + 1) % 100 == 0:
            acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
            logger.info(f"[{idx+1}/{len(samples)}] running acc={acc:.3f}")

    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    halluc_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": acc, "precision": precision,
        "recall": recall, "f1": f1,
        "halluc_rate": halluc_rate,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total": total, "errors": errors,
    }


# Backward-compatible alias
def evaluate_pope(model, processor, model_id, pope_data, image_dir,
                  max_new_tokens=5):
    return evaluate_samples(model, processor, model_id, pope_data, image_dir,
                            max_new_tokens)


def _prepare_inputs(processor, model, model_id, image, prompt):
    model_lower = model_id.lower()
    device = next(model.parameters()).device

    if "qwen" in model_lower:
        from patching.qwen_patcher import _prepare_inputs as prep
        return prep(processor, image, prompt, device)
    elif "llava" in model_lower and "llama" not in model_lower:
        from patching.llava_patcher import _prepare_inputs as prep
        return prep(processor, image, prompt, device)
    elif "internvl" in model_lower:
        from patching.internvl_patcher import _prepare_inputs as prep
        return prep(processor, image, prompt, device)
    elif "llama" in model_lower or "mllama" in model_lower:
        from patching.llama_patcher import _prepare_inputs as prep
        return prep(processor, image, prompt, device)
    else:
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        return inputs.to(device)


# ════════════════════════════════════════════════════════════════════
# CIRCUIT LOADING
# ════════════════════════════════════════════════════════════════════

def load_halluc_targets(circuit_path: str,
                        pathway: str = "halluc",
                        min_abs_d: float = 0.3,
                        top_k: Optional[int] = None,
                        ) -> List[Tuple[int, str]]:
    """Load circuit JSON and extract hallucination-pathway targets.

    Args:
        pathway: "halluc" (d<0) or "visual" (d>0) or "all"
        min_abs_d: minimum |Cohen's d| to include
        top_k: if set, only keep the top-k components by |d|
    """
    with open(circuit_path) as f:
        data = json.load(f)

    candidates = []
    for c in data["components"]:
        d = c["cohen_d"]
        ctype = c["type"]

        if ctype == "head":
            continue
        if pathway == "halluc" and d >= 0:
            continue
        if pathway == "visual" and d <= 0:
            continue
        if abs(d) < min_abs_d:
            continue

        candidates.append((c["layer"], ctype, abs(d)))

    # Sort by |d| descending for top-k selection
    candidates.sort(key=lambda x: x[2], reverse=True)

    if top_k is not None and top_k < len(candidates):
        candidates = candidates[:top_k]

    targets = [(layer, ctype) for layer, ctype, _ in candidates]

    logger.info(f"Loaded {len(targets)} {pathway}-pathway targets "
                f"(top_k={top_k}) from {circuit_path}")
    for layer, ctype in targets:
        logger.info(f"  L{layer}.{ctype}")
    return targets


def make_random_targets(n_layers: int, n_targets: int,
                        seed: int = 42,
                        circuit_targets: Optional[List[Tuple[int, str]]] = None,
                        mode: str = "global") -> List[Tuple[int, str]]:
    """Sample random component targets for the random control.

    mode='global': sample n_targets components uniformly from all (L, type)
        slots. Backward-compatible default.
    mode='type-matched': for each (L, ctype) in circuit_targets, sample one
        replacement (L', ctype) from layers L' != L that are not themselves
        in circuit_targets. Yields the same per-type counts as the circuit
        but disjoint layers, controlling for "any attn/mlp at any layer".
    """
    rng = random.Random(seed)
    if mode == "global":
        all_components = [(l, c) for l in range(n_layers) for c in ["attn", "mlp"]]
        return rng.sample(all_components, min(n_targets, len(all_components)))
    elif mode == "type-matched":
        if circuit_targets is None:
            raise ValueError("type-matched mode requires circuit_targets")
        circuit_set = set(circuit_targets)
        matched: List[Tuple[int, str]] = []
        used: set = set()
        for (_, ctype) in circuit_targets:
            pool = [(l, ctype) for l in range(n_layers)
                    if (l, ctype) not in circuit_set
                    and (l, ctype) not in used]
            if not pool:
                # Pool exhausted (rare): fall back to allowing reuse across
                # circuit components but never picking a circuit member.
                pool = [(l, ctype) for l in range(n_layers)
                        if (l, ctype) not in circuit_set]
            pick = rng.choice(pool)
            matched.append(pick)
            used.add(pick)
        return matched
    else:
        raise ValueError(f"unknown random mode: {mode}")


def get_n_layers(model, model_id: str) -> int:
    model_lower = model_id.lower()
    if "qwen" in model_lower:
        from patching.qwen_patcher import _get_decoder_layers
    elif "llava" in model_lower and "llama" not in model_lower:
        from patching.llava_patcher import _get_decoder_layers
    elif "internvl" in model_lower:
        from patching.internvl_patcher import _get_decoder_layers
    elif "llama" in model_lower or "mllama" in model_lower:
        from patching.llama_patcher import _get_decoder_layers
    else:
        from patching.qwen_patcher import _get_decoder_layers
    return len(_get_decoder_layers(model))


# ════════════════════════════════════════════════════════════════════
# MULTI-BENCHMARK RUNNER
# ════════════════════════════════════════════════════════════════════

def run_condition(model, processor, model_id, data, image_dir,
                  label: str) -> dict:
    """Run one evaluation condition and return results with timing."""
    t0 = time.time()
    res = evaluate_samples(model, processor, model_id, data, image_dir)
    res["time_min"] = (time.time() - t0) / 60
    logger.info(f"  {label}: acc={res['accuracy']:.4f}  "
                f"halluc_rate={res['halluc_rate']:.4f}")
    return res


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Targeted intervention on hallucination pathways (v2)")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--circuit", type=str, required=True)

    # Data — single benchmark (backward-compatible)
    parser.add_argument("--pope-path", type=str, default="data/pope/")
    parser.add_argument("--image-dir", type=str, default="data/coco/val2014/")
    parser.add_argument("--split", type=str, default="adversarial")
    parser.add_argument("--n-samples", type=int, default=500)

    # Data — multi-benchmark
    parser.add_argument("--benchmarks", nargs="+", default=None,
                        help="Benchmarks to evaluate on (e.g., pope mme amber)")
    parser.add_argument("--mme-path", type=str, default="data/mme/")
    parser.add_argument("--mme-image-dir", type=str, default="data/mme/images/")
    parser.add_argument("--amber-path", type=str, default="data/amber/")
    parser.add_argument("--amber-image-dir", type=str,
                        default="data/amber/image/")
    parser.add_argument("--amber-types", nargs="+", default=None,
                        help="AMBER hallucination types to include "
                             "(e.g., existence attribute relation). "
                             "Default: all types.")

    # Intervention params
    parser.add_argument("--scale", type=float, default=0.5,
                        help="Scale factor for suppressed components")
    parser.add_argument("--scales", nargs="+", type=float, default=None,
                        help="Multiple scale factors to test (e.g., 0.0 0.25 0.5 0.75)")
    parser.add_argument("--min-d", type=float, default=0.3)
    parser.add_argument("--n-random-seeds", type=int, default=5)
    parser.add_argument("--random-mode", type=str, default="global",
                        choices=["global", "type-matched"],
                        help="Random control sampling: 'global' samples "
                             "uniformly across all (L, type) slots; "
                             "'type-matched' samples per-circuit-component "
                             "from same type but different layers, "
                             "excluding circuit members.")
    parser.add_argument("--pathway", type=str, default="halluc",
                        choices=["halluc", "visual", "all"])

    # Top-k ablation
    parser.add_argument("--topk-ablation", nargs="+", type=int, default=None,
                        help="Run top-k ablation with these k values "
                             "(e.g., 3 5 10 15 20)")

    # Selection/evaluation split (paper Section 4)
    parser.add_argument("--n-selection", type=int, default=100,
                        help="Number of samples for hyperparameter selection "
                             "(remaining used for evaluation). Paper uses 100/400.")
    parser.add_argument("--no-split", action="store_true",
                        help="Disable selection/eval split (legacy behaviour)")

    parser.add_argument("--output", type=str, default="results/intervention/")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Load model
    from experiments.run_patching import load_model
    model, processor = load_model(args.model)
    n_layers = get_n_layers(model, args.model)

    # Resolve benchmarks + data
    benchmark_configs = []
    if args.benchmarks:
        from experiments.benchmarks import load_benchmark
        for bench in args.benchmarks:
            if bench == "pope":
                data = load_benchmark("pope", args.pope_path,
                                      split=args.split, n_samples=args.n_samples)
                # Add backward-compat keys
                for s in data:
                    s["text"] = s.get("text", s["question"])
                    s["label"] = s.get("label", s["answer"])
                benchmark_configs.append({
                    "name": f"pope_{args.split}",
                    "data": data,
                    "image_dir": args.image_dir,
                })
            elif bench == "mme":
                data = load_benchmark("mme", args.mme_path,
                                      n_samples=args.n_samples)
                for s in data:
                    s["text"] = s.get("text", s["question"])
                    s["label"] = s.get("label", s["answer"])
                benchmark_configs.append({
                    "name": "mme_existence",
                    "data": data,
                    "image_dir": args.mme_image_dir,
                })
            elif bench == "amber":
                data = load_benchmark("amber", args.amber_path,
                                      n_samples=args.n_samples,
                                      halluc_types=args.amber_types)
                for s in data:
                    s["text"] = s.get("text", s["question"])
                    s["label"] = s.get("label", s["answer"])
                amber_name = "amber_disc"
                if args.amber_types:
                    amber_name = f"amber_{'_'.join(args.amber_types)}"
                benchmark_configs.append({
                    "name": amber_name,
                    "data": data,
                    "image_dir": args.amber_image_dir,
                })
    else:
        # Single-benchmark mode
        from experiments.run_patching import load_pope
        pope_data = load_pope(args.pope_path, split=args.split,
                              n_samples=args.n_samples)
        benchmark_configs.append({
            "name": f"pope_{args.split}",
            "data": pope_data,
            "image_dir": args.image_dir,
        })

    # Resolve scale factors
    scale_factors = args.scales if args.scales else [args.scale]

    # ── SPLIT DATA INTO SELECTION / EVALUATION SETS ──────────────
    # Paper Section 4: first 100 samples = selection set (choose s, k),
    # remaining 400 = held-out evaluation set (report final numbers).
    use_split = (not args.no_split) and args.n_samples > args.n_selection
    if use_split:
        n_sel = args.n_selection
        for cfg in benchmark_configs:
            full_data = cfg["data"]
            cfg["selection_data"] = full_data[:n_sel]
            cfg["eval_data"] = full_data[n_sel:]
            logger.info(f"[{cfg['name']}] Split: {len(cfg['selection_data'])} "
                        f"selection + {len(cfg['eval_data'])} eval")
    else:
        for cfg in benchmark_configs:
            cfg["selection_data"] = cfg["data"]
            cfg["eval_data"] = cfg["data"]
        if not args.no_split:
            logger.warning("n_samples <= n_selection, using all data for both")

    # ── Helper: pick best config on selection set ────────────────
    def _is_better(res, baseline, best_res,
                   max_acc_drop: float = 0.02) -> bool:
        """Select config with lowest halluc_rate subject to acc drop <= 2pp."""
        acc_drop = baseline["accuracy"] - res["accuracy"]
        if acc_drop > max_acc_drop:
            return False
        if best_res is None:
            return True
        return res["halluc_rate"] < best_res["halluc_rate"]

    # ── TOP-K ABLATION MODE ──────────────────────────────────────
    if args.topk_ablation:
        logger.info("=" * 60)
        logger.info("TOP-K ABLATION MODE (with selection/eval split)"
                    if use_split else "TOP-K ABLATION MODE")
        logger.info("=" * 60)

        topk_results = {}
        for bench_cfg in benchmark_configs:
            bench_name = bench_cfg["name"]
            sel_data = bench_cfg["selection_data"]
            eval_data = bench_cfg["eval_data"]
            img_dir = bench_cfg["image_dir"]
            topk_results[bench_name] = {}

            # ── Phase 1: Select best k on selection set ──────────
            logger.info(f"\n[{bench_name}] === SELECTION PHASE "
                        f"({len(sel_data)} samples) ===")
            baseline_sel = run_condition(model, processor, args.model,
                                         sel_data, img_dir, "baseline_sel")

            best_k = None
            best_sel_res = None
            selection_details = {}

            for k in args.topk_ablation:
                targets = load_halluc_targets(
                    args.circuit, pathway=args.pathway,
                    min_abs_d=args.min_d, top_k=k)
                if not targets:
                    logger.warning(f"No targets for top_k={k}, skipping")
                    continue

                intervention = ComponentIntervention(
                    model, args.model, targets, scale=args.scale)
                intervention.attach()
                label = f"select_top-{k} (scale={args.scale})"
                res = run_condition(model, processor, args.model,
                                    sel_data, img_dir, label)
                intervention.detach()
                selection_details[k] = res

                if _is_better(res, baseline_sel, best_sel_res):
                    best_k = k
                    best_sel_res = res

            if best_k is None:
                # Fallback: pick k with lowest halluc_rate ignoring acc
                logger.warning("No k satisfies acc constraint; "
                               "falling back to lowest halluc_rate")
                best_k = min(selection_details,
                             key=lambda k: selection_details[k]["halluc_rate"])

            logger.info(f"[{bench_name}] Selected best k={best_k} "
                        f"on selection set")
            topk_results[bench_name]["selected_k"] = best_k
            topk_results[bench_name]["selection_phase"] = {
                "baseline": baseline_sel,
                "per_k": {str(k): v for k, v in selection_details.items()},
            }

            # ── Phase 2: Evaluate ALL k values on eval set ───────
            logger.info(f"\n[{bench_name}] === EVALUATION PHASE "
                        f"({len(eval_data)} samples) ===")
            baseline_eval = run_condition(model, processor, args.model,
                                          eval_data, img_dir, "baseline_eval")
            topk_results[bench_name]["baseline"] = baseline_eval

            for k in args.topk_ablation:
                targets = load_halluc_targets(
                    args.circuit, pathway=args.pathway,
                    min_abs_d=args.min_d, top_k=k)
                if not targets:
                    continue
                intervention = ComponentIntervention(
                    model, args.model, targets, scale=args.scale)
                intervention.attach()
                label = f"eval_top-{k} (scale={args.scale})"
                res = run_condition(model, processor, args.model,
                                    eval_data, img_dir, label)
                intervention.detach()
                res["targets"] = [(l, c) for l, c in targets]
                res["top_k"] = k
                res["is_selected"] = (k == best_k)
                topk_results[bench_name][f"top_{k}"] = res

        # Summary table
        print("\n" + "=" * 80)
        print(f"TOP-K ABLATION RESULTS — {args.model}")
        print("=" * 80)
        for bench_name, results in topk_results.items():
            selected_k = results.get("selected_k", "?")
            print(f"\n  Benchmark: {bench_name}  "
                  f"(selected k={selected_k} on selection set)")
            bl = results["baseline"]
            print(f"    {'Condition':<20s} {'Accuracy':>10s} {'Halluc%':>10s} "
                  f"{'dAcc':>8s} {'dHalluc':>8s} {'':>4s}")
            print(f"    {'baseline':<20s} {bl['accuracy']:>10.4f} "
                  f"{bl['halluc_rate']:>10.4f} {'---':>8s} {'---':>8s}")
            for k in args.topk_ablation:
                key = f"top_{k}"
                if key in results:
                    r = results[key]
                    da = r["accuracy"] - bl["accuracy"]
                    dh = r["halluc_rate"] - bl["halluc_rate"]
                    sel_mark = " <<<" if r.get("is_selected") else ""
                    print(f"    {f'top-{k}':<20s} {r['accuracy']:>10.4f} "
                          f"{r['halluc_rate']:>10.4f} {da:>+8.4f} "
                          f"{dh:>+8.4f}{sel_mark}")

        out_path = os.path.join(args.output, "topk_ablation_results.json")
        with open(out_path, "w") as f:
            json.dump(topk_results, f, indent=2, default=str)
        logger.info(f"Top-k ablation results saved to {out_path}")
        return

    # ── STANDARD INTERVENTION MODE ───────────────────────────────
    halluc_targets = load_halluc_targets(
        args.circuit, pathway=args.pathway, min_abs_d=args.min_d)

    if not halluc_targets:
        logger.error("No targets found! Check circuit.json and --min-d")
        return

    all_results = {}

    for bench_cfg in benchmark_configs:
        bench_name = bench_cfg["name"]
        sel_data = bench_cfg["selection_data"]
        eval_data = bench_cfg["eval_data"]
        img_dir = bench_cfg["image_dir"]

        logger.info("\n" + "=" * 60)
        logger.info(f"EVALUATING: {bench_name}")
        logger.info("=" * 60)

        bench_results = {}

        # ── Phase 1: Select best scale on selection set ──────────
        logger.info(f"\n--- SELECTION PHASE ({len(sel_data)} samples) ---")

        baseline_sel = run_condition(model, processor, args.model,
                                     sel_data, img_dir, "baseline_sel")

        best_s = scale_factors[0]
        best_sel_res = None
        selection_details = {}

        for s in scale_factors:
            logger.info(f"  [select] scale={s}, "
                        f"{len(halluc_targets)} components")
            intervention = ComponentIntervention(
                model, args.model, halluc_targets, scale=s)
            intervention.attach()
            res = run_condition(model, processor, args.model,
                                sel_data, img_dir, f"select_s{s}")
            intervention.detach()
            selection_details[s] = res

            if _is_better(res, baseline_sel, best_sel_res):
                best_s = s
                best_sel_res = res

        logger.info(f"[{bench_name}] Selected best scale={best_s} "
                    f"on selection set")
        bench_results["selected_scale"] = best_s
        bench_results["selection_phase"] = {
            "baseline": baseline_sel,
            "per_scale": {str(s): v for s, v in selection_details.items()},
        }

        # ── Phase 2: Evaluate on held-out eval set ───────────────
        logger.info(f"\n--- EVALUATION PHASE ({len(eval_data)} samples) ---")

        # Baseline on eval set
        logger.info("Condition 1: BASELINE (eval set)")
        baseline_eval = run_condition(model, processor, args.model,
                                      eval_data, img_dir, "baseline")
        bench_results["baseline"] = baseline_eval

        # Targeted with selected scale
        logger.info(f"Condition 2: TARGETED (scale={best_s}, eval set)")
        intervention = ComponentIntervention(
            model, args.model, halluc_targets, scale=best_s)
        intervention.attach()
        targeted = run_condition(model, processor, args.model,
                                 eval_data, img_dir, f"targeted_s{best_s}")
        intervention.detach()
        targeted["targets"] = [(l, c) for l, c in halluc_targets]
        targeted["scale"] = best_s
        bench_results["targeted"] = targeted

        # Also run ALL scale factors on eval set for completeness (Table 10)
        if len(scale_factors) > 1:
            bench_results["all_scales_eval"] = {}
            for s in scale_factors:
                intervention = ComponentIntervention(
                    model, args.model, halluc_targets, scale=s)
                intervention.attach()
                res = run_condition(model, processor, args.model,
                                    eval_data, img_dir, f"eval_s{s}")
                intervention.detach()
                res["scale"] = s
                bench_results["all_scales_eval"][str(s)] = res

        # Condition 3: Random control (on eval set)
        logger.info(f"Condition 3: RANDOM CONTROL "
                    f"({args.n_random_seeds} seeds, mode={args.random_mode})")
        random_results = []
        for seed in range(args.n_random_seeds):
            rand_targets = make_random_targets(
                n_layers, len(halluc_targets), seed=seed + 100,
                circuit_targets=halluc_targets, mode=args.random_mode)
            intervention = ComponentIntervention(
                model, args.model, rand_targets, scale=best_s)
            intervention.attach()
            rand_res = run_condition(model, processor, args.model,
                                     eval_data, img_dir, f"random_s{seed+100}")
            intervention.detach()
            rand_res["seed"] = seed + 100
            rand_res["mode"] = args.random_mode
            rand_res["sampled_targets"] = [list(t) for t in rand_targets]
            random_results.append(rand_res)

        rand_accs = [r["accuracy"] for r in random_results]
        rand_halls = [r["halluc_rate"] for r in random_results]
        bench_results["random_control"] = {
            "mode": args.random_mode,
            "mean_accuracy": float(np.mean(rand_accs)),
            "std_accuracy": float(np.std(rand_accs)),
            "mean_halluc_rate": float(np.mean(rand_halls)),
            "std_halluc_rate": float(np.std(rand_halls)),
            "per_seed": random_results,
        }

        all_results[bench_name] = bench_results

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"INTERVENTION RESULTS — {args.model}")
    if use_split:
        print(f"  (selection: {args.n_selection} samples, "
              f"eval: {args.n_samples - args.n_selection} samples)")
    print("=" * 80)

    for bench_name, results in all_results.items():
        bl = results["baseline"]
        sel_s = results.get("selected_scale", "?")
        print(f"\n  Benchmark: {bench_name}  (selected scale={sel_s})")
        print(f"    Baseline:    acc={bl['accuracy']:.4f}  "
              f"halluc={bl['halluc_rate']:.4f}")

        tgt = results.get("targeted")
        if tgt:
            da = tgt["accuracy"] - bl["accuracy"]
            dh = tgt["halluc_rate"] - bl["halluc_rate"]
            rel = (-dh / bl["halluc_rate"] * 100) if bl["halluc_rate"] > 0 else 0
            print(f"    Targeted(s={sel_s}): acc={tgt['accuracy']:.4f}  "
                  f"halluc={tgt['halluc_rate']:.4f}  "
                  f"dAcc={da:+.4f}  dHalluc={dh:+.4f}  "
                  f"Rel.↓={rel:.0f}%")

        rc = results.get("random_control")
        if rc:
            rd = rc["mean_accuracy"] - bl["accuracy"]
            print(f"    Random ctrl: acc={rc['mean_accuracy']:.4f} "
                  f"+-{rc['std_accuracy']:.4f}  dAcc={rd:+.4f}")

    # Save
    out_path = os.path.join(args.output, "intervention_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()