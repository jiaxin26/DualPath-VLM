"""
Literal ITI (Inference-Time Intervention) Baseline
=====================================================

Implements Li et al. (NeurIPS 2024) "Inference-Time Intervention:
Eliciting Truthful Answers from a Language Model", adapted to VLMs.

This is a stronger baseline than the simpler directional suppression
in run_intervention_v2.py.  Whereas v2 uses a mean-difference direction
and projects it out, literal ITI:

  1. Trains a binary logistic-regression probe per (layer, head) to
     classify correct vs hallucinating samples from head activations.
  2. Ranks heads by held-out probe accuracy.  Selects top-K heads.
  3. Uses the probe's normalized weight vector as the "truthful
     direction" for each selected head.
  4. At inference time, additively shifts each selected head's output
     by  alpha * sigma * w_unit   (sigma = std(w . h) on training).
  5. Operates at head granularity (not component), hooking o_proj
     pre-hook so each head's contribution is modified before W_O.

Usage
-----

  # Option A: full pipeline (extract probes + evaluate), Llama only
  PYTHONPATH=. python experiments/run_iti_baseline.py \\
    --model meta-llama/Llama-3.2-11B-Vision-Instruct \\
    --mode full \\
    --pope-path data/pope/ --image-dir data/coco/val2014/ \\
    --n-extract 1000 --n-samples 500 --n-val 100 \\
    --output results/iti_llama/

  # Option B: two steps (useful if you want to re-evaluate without
  # re-extracting activations)
  PYTHONPATH=. python experiments/run_iti_baseline.py \\
    --model meta-llama/Llama-3.2-11B-Vision-Instruct \\
    --mode extract --n-extract 1000 \\
    --pope-path data/pope/ --image-dir data/coco/val2014/ \\
    --output results/iti_llama/

  PYTHONPATH=. python experiments/run_iti_baseline.py \\
    --model meta-llama/Llama-3.2-11B-Vision-Instruct \\
    --mode evaluate --probes results/iti_llama/probes.pt \\
    --pope-path data/pope/ --image-dir data/coco/val2014/ \\
    --n-samples 500 --n-val 100 \\
    --output results/iti_llama/

Runtime estimate (H200 144GB, Llama-3.2-11B-Vision)
---------------------------------------------------
  Extraction (1000 samples, all layers × heads captured):  ~30 min
  Probe training (one LR per head):                         ~5 min (CPU)
  Grid evaluation (16 configs × 100 val + 1 × 400 test):    ~60 min
  TOTAL:                                                    ~95 min
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("iti")


# ════════════════════════════════════════════════════════════════════
# MODEL-AGNOSTIC HELPERS
# ════════════════════════════════════════════════════════════════════

def _get_decoder_layers(model, model_id: str) -> nn.ModuleList:
    m = model_id.lower()
    if "qwen" in m:
        from patching.qwen_patcher import _get_decoder_layers as f
    elif "llava" in m and "llama" not in m:
        from patching.llava_patcher import _get_decoder_layers as f
    elif "internvl" in m:
        from patching.internvl_patcher import _get_decoder_layers as f
    elif "llama" in m or "mllama" in m:
        from patching.llama_patcher import _get_decoder_layers as f
    else:
        from patching.qwen_patcher import _get_decoder_layers as f
    return f(model)


def _get_head_config(model, model_id: str) -> Tuple[int, int, int]:
    m = model_id.lower()
    if "qwen" in m:
        from patching.qwen_patcher import _get_head_config as f
    elif "llava" in m and "llama" not in m:
        from patching.llava_patcher import _get_head_config as f
    elif "internvl" in m:
        from patching.internvl_patcher import _get_head_config as f
    elif "llama" in m or "mllama" in m:
        from patching.llama_patcher import _get_head_config as f
    else:
        from patching.qwen_patcher import _get_head_config as f
    return f(model)


def _iter_attn_layers(layers) -> List[Tuple[int, nn.Module]]:
    """Return list of (layer_idx, attn_module) for layers whose
    attention module has an o_proj.  Skips cross-attn layers that
    don't follow the standard MHA pattern."""
    out = []
    for l_idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None) or \
               getattr(layer, "cross_attn", None)
        if attn is None or not hasattr(attn, "o_proj"):
            continue
        out.append((l_idx, attn))
    return out


# ════════════════════════════════════════════════════════════════════
# HEAD-LEVEL ACTIVATION EXTRACTION
# ════════════════════════════════════════════════════════════════════

class HeadActivationExtractor:
    """For each sample, capture the pre-o_proj activation of every
    head at the last input token position, and label the sample as
    correct / hallucinating based on the model's generated answer."""

    def __init__(self, model, processor, model_id: str):
        self.model = model
        self.processor = processor
        self.model_id = model_id
        self.layers = _get_decoder_layers(model, model_id)
        self.n_heads, _, self.head_dim = _get_head_config(model, model_id)
        self.n_layers = len(self.layers)
        self.attn_layers = _iter_attn_layers(self.layers)

        # activations[l][h] is a list of (head_dim,) float tensors
        self.activations: List[List[List[torch.Tensor]]] = [
            [[] for _ in range(self.n_heads)] for _ in range(self.n_layers)
        ]
        self.labels: List[int] = []  # 1 = correct, 0 = hallucinating

        logger.info(f"HeadActivationExtractor: {self.n_layers} layers, "
                    f"{self.n_heads} heads/layer, head_dim={self.head_dim}, "
                    f"{len(self.attn_layers)} attn-with-o_proj layers")

    @torch.no_grad()
    def collect_sample(self, image, prompt: str, gt_answer: str) -> bool:
        """Run one sample; returns True if the model answered correctly."""
        from experiments.run_intervention import _prepare_inputs

        inputs = _prepare_inputs(
            self.processor, self.model, self.model_id, image, prompt)

        captured: Dict[int, torch.Tensor] = {}
        handles = []

        for l_idx, attn in self.attn_layers:
            def make_hook(li=l_idx):
                def hook(module, args):
                    # args[0]: (B, S, n_heads * head_dim)
                    captured[li] = args[0].detach().to("cpu")
                    return None
                return hook
            handles.append(attn.o_proj.register_forward_pre_hook(make_hook()))

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
        is_correct = (pred == gt_answer.strip().lower())
        label = 1 if is_correct else 0

        # Store head activations at the last input-token position
        for l_idx, pre_o in captured.items():
            B, S, D = pre_o.shape
            pre_o_r = pre_o.view(B, S, self.n_heads, self.head_dim)
            last_tok = pre_o_r[0, -1, :, :].float()  # (n_heads, head_dim)
            for h_idx in range(self.n_heads):
                self.activations[l_idx][h_idx].append(last_tok[h_idx].clone())

        self.labels.append(label)
        return is_correct

    def to_arrays(self) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
        """Stack activations into per-head numpy arrays."""
        y = np.asarray(self.labels, dtype=np.int32)
        out: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
        for l in range(self.n_layers):
            for h in range(self.n_heads):
                acts = self.activations[l][h]
                if not acts or len(acts) != len(y):
                    continue
                X = torch.stack(acts).numpy()
                out[(l, h)] = {"X": X, "y": y}
        return out


# ════════════════════════════════════════════════════════════════════
# PROBE TRAINING
# ════════════════════════════════════════════════════════════════════

def train_head_probes(
    head_data: Dict[Tuple[int, int], Dict[str, np.ndarray]],
    val_frac: float = 0.2,
    C: float = 1.0,
    seed: int = 0,
) -> Dict[Tuple[int, int], dict]:
    """Fit a logistic-regression probe per head.  Returns dict mapping
    (layer, head) -> {val_acc, w_unit, sigma, n_train, n_val}.

    Direction convention: w_unit points toward class 1 (= correct), so
    shifting activations in the +w_unit direction should increase the
    probe's correct-answer confidence.
    """
    from sklearn.linear_model import LogisticRegression

    rng = np.random.RandomState(seed)
    probes: Dict[Tuple[int, int], dict] = {}

    for (l, h), data in head_data.items():
        X, y = data["X"], data["y"]
        n = len(y)
        n_correct = int(y.sum())
        n_halluc = n - n_correct

        # Need enough of each class
        if n_correct < 5 or n_halluc < 5:
            continue

        perm = rng.permutation(n)
        n_val = max(int(n * val_frac), 1)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            continue

        try:
            clf = LogisticRegression(
                C=C, max_iter=1000, solver="liblinear",
                class_weight="balanced",
            ).fit(X_tr, y_tr)
        except Exception as e:
            logger.warning(f"  probe L{l}.H{h} failed: {e}")
            continue

        val_acc = float(clf.score(X_va, y_va))

        w = clf.coef_[0]
        w_norm = float(np.linalg.norm(w))
        if w_norm < 1e-8:
            continue
        w_unit = (w / w_norm).astype(np.float32)
        sigma = float((X_tr @ w_unit).std())

        probes[(l, h)] = {
            "val_acc": val_acc,
            "w_unit": w_unit,
            "sigma": sigma,
            "n_train": int(len(tr_idx)),
            "n_val": int(n_val),
        }

    return probes


def rank_heads_by_acc(
    probes: Dict[Tuple[int, int], dict]
) -> List[Tuple[int, int]]:
    items = [(k, v["val_acc"]) for k, v in probes.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    return [k for k, _ in items]


# ════════════════════════════════════════════════════════════════════
# ITI INTERVENTION HOOK
# ════════════════════════════════════════════════════════════════════

class ITIIntervention:
    """Pre-hook on o_proj.  For every selected (layer, head) pair,
    add  alpha * sigma * w_unit  to that head's slice of the pre-o_proj
    activation.  Fires at every forward pass (every generated token).
    """

    def __init__(
        self,
        model,
        model_id: str,
        probes: Dict[Tuple[int, int], dict],
        selected: List[Tuple[int, int]],
        alpha: float,
    ):
        self.model = model
        self.alpha = alpha
        self.layers = _get_decoder_layers(model, model_id)
        self.n_heads, _, self.head_dim = _get_head_config(model, model_id)
        self.handles: List = []

        # Group selected heads by layer
        by_layer: Dict[int, List[Tuple[int, torch.Tensor, float]]] = defaultdict(list)
        for (l, h) in selected:
            p = probes.get((l, h))
            if p is None:
                continue
            by_layer[l].append(
                (h, torch.from_numpy(p["w_unit"]), p["sigma"]))
        self.by_layer = by_layer

        logger.info(f"ITIIntervention: {len(selected)} heads "
                    f"across {len(self.by_layer)} layers, alpha={alpha}")

    def attach(self):
        self.detach()
        n_heads = self.n_heads
        head_dim = self.head_dim
        alpha = self.alpha

        for l_idx, shifts in self.by_layer.items():
            layer = self.layers[l_idx]
            attn = getattr(layer, "self_attn", None) or \
                   getattr(layer, "cross_attn", None)
            if attn is None or not hasattr(attn, "o_proj"):
                continue

            # Precompute the layer's shift matrix: (n_heads, head_dim)
            shift_mat = torch.zeros(n_heads, head_dim)
            for h_idx, w_unit, sigma in shifts:
                shift_mat[h_idx] = alpha * sigma * w_unit

            def make_hook(s=shift_mat):
                def hook(module, args):
                    pre_o = args[0]  # (B, S, n_heads * head_dim)
                    B, S, D = pre_o.shape
                    s_local = s.to(device=pre_o.device, dtype=pre_o.dtype)
                    pre_o_r = pre_o.view(B, S, n_heads, head_dim).clone()
                    pre_o_r = pre_o_r + s_local.view(1, 1, n_heads, head_dim)
                    return (pre_o_r.reshape(B, S, D),)
                return hook
            self.handles.append(
                attn.o_proj.register_forward_pre_hook(make_hook()))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ════════════════════════════════════════════════════════════════════
# I/O HELPERS
# ════════════════════════════════════════════════════════════════════

def save_probes(probes: Dict[Tuple[int, int], dict],
                path: str, meta: Optional[dict] = None):
    serializable = {f"L{l}.H{h}": {
        "val_acc": p["val_acc"],
        "w_unit": torch.from_numpy(p["w_unit"]),
        "sigma": p["sigma"],
        "n_train": p["n_train"],
        "n_val": p["n_val"],
        "layer": l,
        "head": h,
    } for (l, h), p in probes.items()}
    payload = {"probes": serializable}
    if meta:
        payload["meta"] = meta
    torch.save(payload, path)


def load_probes(path: str) -> Tuple[Dict[Tuple[int, int], dict], Optional[dict]]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    probes = {}
    for _, info in raw["probes"].items():
        l, h = int(info["layer"]), int(info["head"])
        probes[(l, h)] = {
            "val_acc": info["val_acc"],
            "w_unit": info["w_unit"].numpy().astype(np.float32),
            "sigma": info["sigma"],
            "n_train": info["n_train"],
            "n_val": info["n_val"],
        }
    return probes, raw.get("meta")


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Literal ITI baseline (Li et al. 2024)")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True,
                        choices=["extract", "evaluate", "full"])
    parser.add_argument("--probes", type=str, default=None,
                        help="Path to probes.pt (for evaluate mode)")
    parser.add_argument("--pope-path", type=str, default="data/pope/")
    parser.add_argument("--image-dir", type=str, default="data/coco/val2014/")
    parser.add_argument("--split", type=str, default="adversarial")
    parser.add_argument("--n-extract", type=int, default=1000,
                        help="Samples for probe training")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Samples for intervention evaluation")
    parser.add_argument("--n-val", type=int, default=100,
                        help="First n_val of n_samples = selection set")
    parser.add_argument("--k-values", type=int, nargs="+",
                        default=[12, 24, 48, 96])
    parser.add_argument("--alpha-values", type=float, nargs="+",
                        default=[-20.0, -15.0, -10.0, -5.0, 5.0, 10.0, 15.0, 20.0])
    parser.add_argument("--probe-C", type=float, default=1.0,
                        help="Inverse L2 regularization for probes")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--disjoint-splits", action="store_true",
                        help="If set, draw probe-training samples from "
                             "indices [n_samples : n_samples+n_extract] "
                             "instead of [:n_extract], so they are "
                             "disjoint from the evaluation set.")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    from experiments.run_patching import load_model, load_pope

    logger.info(f"Loading model: {args.model}")
    model, processor = load_model(args.model)

    # Figure out how many POPE samples we need total
    if args.disjoint_splits:
        n_total_needed = args.n_samples + args.n_extract
    else:
        n_total_needed = max(args.n_samples, args.n_extract)
    pope_data = load_pope(args.pope_path, split=args.split,
                          n_samples=n_total_needed)

    probes: Optional[Dict[Tuple[int, int], dict]] = None

    # ── Phase 1: Extraction + probes ───────────────────────────────
    if args.mode in ("extract", "full"):
        logger.info("=" * 60)
        logger.info("PHASE 1: Collect head activations + train probes")
        logger.info("=" * 60)

        if args.disjoint_splits:
            extract_data = pope_data[args.n_samples:
                                      args.n_samples + args.n_extract]
            logger.info(f"Using samples [{args.n_samples}:"
                        f"{args.n_samples+args.n_extract}] for extraction "
                        f"(disjoint from evaluation set)")
        else:
            extract_data = pope_data[:args.n_extract]
            logger.info(f"Using samples [0:{args.n_extract}] for extraction "
                        f"(overlaps evaluation set — set --disjoint-splits "
                        f"to avoid)")

        extractor = HeadActivationExtractor(model, processor, args.model)
        skipped = 0
        for idx, sample in enumerate(extract_data):
            question = sample.get("text") or sample.get("question", "")
            gt = (sample.get("label") or sample.get("answer", "")
                  ).strip().lower()
            image_file = sample.get("image", "")
            image_path = os.path.join(args.image_dir, image_file)
            if not os.path.exists(image_path):
                skipped += 1
                continue
            try:
                img = Image.open(image_path).convert("RGB")
                prompt = question + " Answer with yes or no."
                extractor.collect_sample(img, prompt, gt)
            except Exception as e:
                logger.warning(f"  sample {idx} failed: {e}")
                skipped += 1
                continue
            if (idx + 1) % 100 == 0:
                n_c = sum(extractor.labels)
                n_h = len(extractor.labels) - n_c
                logger.info(f"  [{idx+1}/{len(extract_data)}] "
                            f"correct={n_c}, halluc={n_h}, skipped={skipped}")

        n_c = sum(extractor.labels)
        n_h = len(extractor.labels) - n_c
        logger.info(f"Extraction done: {n_c} correct, {n_h} hallucinating, "
                    f"{skipped} skipped")
        if n_h < 20:
            logger.warning(
                f"Only {n_h} hallucinating samples — probes may be noisy. "
                f"Consider --n-extract 2000.")

        logger.info("Training probes...")
        head_data = extractor.to_arrays()
        del extractor  # free activation memory
        probes = train_head_probes(head_data, val_frac=0.2, C=args.probe_C)
        logger.info(f"Trained {len(probes)} probes.")

        probe_path = os.path.join(args.output, "probes.pt")
        save_probes(probes, probe_path, meta={
            "n_correct": n_c,
            "n_halluc": n_h,
            "n_extract_requested": args.n_extract,
            "model": args.model,
            "split": args.split,
            "disjoint_splits": args.disjoint_splits,
        })
        logger.info(f"Saved probes → {probe_path}")

        ranked = rank_heads_by_acc(probes)
        logger.info("Top-20 heads by probe val accuracy:")
        for key in ranked[:20]:
            p = probes[key]
            logger.info(f"  L{key[0]}.H{key[1]}: "
                        f"val_acc={p['val_acc']:.3f}, sigma={p['sigma']:.3f}")

    # ── Load probes if needed ──────────────────────────────────────
    if args.mode == "evaluate":
        if args.probes is None:
            raise SystemExit("--probes required in evaluate mode")
        probes, meta = load_probes(args.probes)
        logger.info(f"Loaded {len(probes)} probes from {args.probes}")

    # ── Phase 2: Evaluation ────────────────────────────────────────
    if args.mode in ("evaluate", "full"):
        logger.info("=" * 60)
        logger.info("PHASE 2: Grid evaluation + selection")
        logger.info("=" * 60)

        from experiments.run_intervention import evaluate_pope

        if probes is None:
            raise RuntimeError("No probes available for evaluation.")

        eval_pool = pope_data[:args.n_samples]
        val_data = eval_pool[:args.n_val]
        test_data = eval_pool[args.n_val:]
        logger.info(f"Selection set: {len(val_data)} | "
                    f"test set: {len(test_data)}")

        ranked = rank_heads_by_acc(probes)
        results: Dict[str, object] = {}

        # Baseline on val (for feasibility check) and on test (for report)
        logger.info("Baseline on val...")
        base_val = evaluate_pope(model, processor, args.model,
                                 val_data, args.image_dir)
        logger.info(f"  val acc={base_val['accuracy']:.4f}, "
                    f"hall={base_val['halluc_rate']:.4f}")
        results["baseline_val"] = base_val

        logger.info("Baseline on test...")
        base_test = evaluate_pope(model, processor, args.model,
                                  test_data, args.image_dir)
        logger.info(f"  test acc={base_test['accuracy']:.4f}, "
                    f"hall={base_test['halluc_rate']:.4f}")
        results["baseline_test"] = base_test

        # Grid search on val set
        logger.info("Grid (K × alpha) on val set...")
        grid: Dict[Tuple[int, float], dict] = {}
        for K in args.k_values:
            if K > len(ranked):
                logger.warning(f"  K={K} > #probes ({len(ranked)}), skip")
                continue
            selected = ranked[:K]
            for alpha in args.alpha_values:
                iti = ITIIntervention(
                    model, args.model, probes, selected, alpha)
                iti.attach()
                try:
                    m = evaluate_pope(model, processor, args.model,
                                      val_data, args.image_dir)
                finally:
                    iti.detach()
                grid[(K, alpha)] = m
                results[f"val_K{K}_a{alpha}"] = m
                d_acc = m["accuracy"] - base_val["accuracy"]
                d_hall = m["halluc_rate"] - base_val["halluc_rate"]
                logger.info(f"  K={K}, α={alpha}: "
                            f"acc={m['accuracy']:.3f} (Δ{d_acc:+.3f}), "
                            f"hall={m['halluc_rate']:.3f} (Δ{d_hall:+.3f})")

        if not grid:
            logger.error("Grid is empty — check --k-values / probe count.")
            return

        # Selection rule: minimise val halluc rate subject to
        # val accuracy drop ≤ 2 pp (matching the paper's criterion).
        acc_budget = 0.02
        feasible = {k: v for k, v in grid.items()
                    if (base_val["accuracy"] - v["accuracy"]) <= acc_budget}
        if feasible:
            sel = min(feasible.keys(),
                      key=lambda ka: grid[ka]["halluc_rate"])
            sel_status = f"feasible ({len(feasible)}/{len(grid)})"
        else:
            sel = min(grid.keys(), key=lambda ka: grid[ka]["halluc_rate"])
            sel_status = "NO config met the ≤2pp accuracy constraint; " \
                         "selecting lowest halluc rate anyway"
        K_sel, alpha_sel = sel
        logger.info(f"\nSelected config: K={K_sel}, α={alpha_sel}  [{sel_status}]")

        # Apply selected to test set
        logger.info("Applying selected config to test set...")
        sel_iti = ITIIntervention(model, args.model, probes,
                                  ranked[:K_sel], alpha_sel)
        sel_iti.attach()
        try:
            test_metrics = evaluate_pope(model, processor, args.model,
                                         test_data, args.image_dir)
        finally:
            sel_iti.detach()
        results["selected_test"] = test_metrics
        results["selected_config"] = {
            "K": K_sel, "alpha": alpha_sel, "status": sel_status}

        d_acc_test = test_metrics["accuracy"] - base_test["accuracy"]
        d_hall_test = test_metrics["halluc_rate"] - base_test["halluc_rate"]
        rel_reduction = (-d_hall_test / base_test["halluc_rate"] * 100
                         if base_test["halluc_rate"] > 0 else 0.0)

        # Summary print
        print("\n" + "=" * 70)
        print(f"ITI BASELINE — {args.model.split('/')[-1]}")
        print("=" * 70)
        print(f"Selected:  K={K_sel} heads, alpha={alpha_sel}")
        print(f"{'Set':<12s} {'Acc':>8s} {'Halluc':>8s} "
              f"{'ΔAcc':>8s} {'ΔHalluc':>10s} {'Rel.↓':>8s}")
        print("-" * 60)
        print(f"{'Baseline':<12s} {base_test['accuracy']:>8.4f} "
              f"{base_test['halluc_rate']:>8.4f}")
        print(f"{'ITI (test)':<12s} {test_metrics['accuracy']:>8.4f} "
              f"{test_metrics['halluc_rate']:>8.4f} "
              f"{d_acc_test:>+8.4f} {d_hall_test:>+10.4f} "
              f"{rel_reduction:>7.1f}%")
        print("=" * 70)

        out_path = os.path.join(args.output, "iti_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
