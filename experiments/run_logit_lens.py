"""
Logit Lens Analysis: Validating Dual-Pathway Functional Roles
===============================================================

Applies the "logit lens" (nostalgebraist, 2020) at every transformer layer
to trace how each layer contributes to building the model's answer.

For each POPE sample, we:
  1. Run a forward pass, capturing hidden states at every layer
  2. Project each layer's hidden state through lm_head (with final norm)
  3. Record P(correct), P(incorrect), rank, and logit_diff at each layer
  4. Partition layers by pathway membership (from circuit.json)

Key hypotheses validated:
  - Visual grounding pathway layers progressively INCREASE P(correct)
  - Hallucination pathway layers INCREASE P(incorrect) or DECREASE P(correct)
  - The transition between pathways should be visible in the logit_diff
    trajectory

This provides evidence beyond statistical grouping (Cohen's d sign) that
the two pathways have genuinely distinct functional roles.

Usage:
  PYTHONPATH=. python experiments/run_logit_lens.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --circuit results/qwen3vl_component/circuit.json \
    --pope-path data/pope/ --image-dir data/coco/val2014/ \
    --n-samples 100 --output results/logit_lens_qwen/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("logit_lens")


# ════════════════════════════════════════════════════════════════════
# PATHWAY MEMBERSHIP
# ════════════════════════════════════════════════════════════════════

def load_layer_pathway_map(circuit_path: str,
                            min_abs_d: float = 0.3
                            ) -> Dict[int, str]:
    """Map each layer to its dominant pathway based on circuit.json.

    A layer is "halluc" if it has more d<0 components than d>0,
    "visual" if vice versa, and "mixed" if equal.
    """
    with open(circuit_path) as f:
        data = json.load(f)

    layer_scores = {}  # layer → net score (+ = visual, - = halluc)

    for c in data["components"]:
        if c["type"] == "head":
            continue
        if abs(c["cohen_d"]) < min_abs_d:
            continue
        layer = c["layer"]
        layer_scores[layer] = layer_scores.get(layer, 0) + np.sign(c["cohen_d"])

    pathway_map = {}
    for layer, score in layer_scores.items():
        if score > 0:
            pathway_map[layer] = "visual"
        elif score < 0:
            pathway_map[layer] = "halluc"
        else:
            pathway_map[layer] = "mixed"

    return pathway_map


# ════════════════════════════════════════════════════════════════════
# ANALYSIS
# ════════════════════════════════════════════════════════════════════

def analyze_logit_lens_results(
    all_profiles: List[dict],
    pathway_map: Dict[int, str],
) -> dict:
    """Aggregate logit lens profiles and compute pathway-level statistics.

    For each pathway type (visual / halluc / other), computes:
      - Mean Δlogit_diff at that layer (how much the layer shifts logit_diff)
      - Correlation between pathway membership and logit_diff change
    """
    if not all_profiles:
        return {}

    n_layers = all_profiles[0]["n_layers"]

    # Separate by hallucination status
    correct_profiles = [p for p in all_profiles if not p["is_hallucination"]]
    halluc_profiles = [p for p in all_profiles if p["is_hallucination"]]

    results = {}

    for category, profiles in [("all", all_profiles),
                                ("correct", correct_profiles),
                                ("hallucination", halluc_profiles)]:
        if not profiles:
            continue

        # Build per-layer arrays
        logit_diffs = np.zeros((len(profiles), n_layers))
        p_correct_arr = np.zeros((len(profiles), n_layers))
        p_incorrect_arr = np.zeros((len(profiles), n_layers))

        for i, prof in enumerate(profiles):
            for ld in prof["layers"]:
                l = ld["layer"]
                logit_diffs[i, l] = ld["logit_diff"]
                p_correct_arr[i, l] = ld["p_correct"]
                p_incorrect_arr[i, l] = ld["p_incorrect"]

        # Per-layer delta (layer L contribution = logit_diff[L] - logit_diff[L-1])
        deltas = np.diff(logit_diffs, axis=1, prepend=logit_diffs[:, :1])

        # Mean trajectory
        mean_ld = logit_diffs.mean(axis=0)
        mean_delta = deltas.mean(axis=0)
        mean_pc = p_correct_arr.mean(axis=0)
        mean_pi = p_incorrect_arr.mean(axis=0)

        # Pathway-level aggregation
        pathway_stats = {}
        for ptype in ["visual", "halluc", "mixed"]:
            layers_in = [l for l, p in pathway_map.items() if p == ptype]
            if not layers_in:
                continue
            mean_delta_pw = mean_delta[layers_in].mean()
            # Do individual layers boost correct or incorrect?
            delta_correct_pw = np.diff(mean_pc)[np.array(layers_in[:-1])
                                                 if layers_in[-1] == n_layers - 1
                                                 else np.array(layers_in)].mean() \
                if len(layers_in) > 0 else 0.0

            pathway_stats[ptype] = {
                "layers": layers_in,
                "mean_delta_logit_diff": float(mean_delta_pw),
                "n_layers": len(layers_in),
            }

        # Functional role test:
        # For visual pathway layers: do they increase logit_diff (positive delta)?
        # For halluc pathway layers: do they decrease logit_diff (negative delta)?
        visual_layers = [l for l, p in pathway_map.items() if p == "visual"]
        halluc_layers = [l for l, p in pathway_map.items() if p == "halluc"]

        if visual_layers and halluc_layers:
            visual_deltas = deltas[:, visual_layers].mean(axis=1)
            halluc_deltas = deltas[:, halluc_layers].mean(axis=1)

            # Test: visual delta > halluc delta
            t, p = sp_stats.ttest_ind(visual_deltas, halluc_deltas)
            d_vals = np.concatenate([visual_deltas, halluc_deltas])
            pooled = np.sqrt((visual_deltas.var(ddof=1) +
                              halluc_deltas.var(ddof=1)) / 2)
            cohen_d = ((visual_deltas.mean() - halluc_deltas.mean()) /
                       (pooled + 1e-12))

            pathway_stats["differential_test"] = {
                "visual_mean_delta": float(visual_deltas.mean()),
                "halluc_mean_delta": float(halluc_deltas.mean()),
                "t": float(t),
                "p": float(p),
                "cohen_d": float(cohen_d),
            }

        results[category] = {
            "n_samples": len(profiles),
            "mean_logit_diff_trajectory": mean_ld.tolist(),
            "mean_delta_trajectory": mean_delta.tolist(),
            "mean_p_correct_trajectory": mean_pc.tolist(),
            "mean_p_incorrect_trajectory": mean_pi.tolist(),
            "pathway_stats": pathway_stats,
        }

    return results


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Logit lens analysis for dual-pathway validation")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--circuit", type=str, required=True)
    parser.add_argument("--pope-path", type=str, default="data/pope/")
    parser.add_argument("--image-dir", type=str, default="data/coco/val2014/")
    parser.add_argument("--split", type=str, default="adversarial")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--min-d", type=float, default=0.3)
    parser.add_argument("--output", type=str, default="results/logit_lens/")

    args = parser.parse_args()

    import torch
    from PIL import Image
    from patching.core import PatchingConfig
    from experiments.run_patching import load_model, load_pope, create_patcher

    os.makedirs(args.output, exist_ok=True)

    model, processor = load_model(args.model)
    pope_data = load_pope(args.pope_path, split=args.split,
                          n_samples=args.n_samples)
    pathway_map = load_layer_pathway_map(args.circuit, min_abs_d=args.min_d)

    logger.info(f"Pathway map: {pathway_map}")

    config = PatchingConfig(granularity="component")
    patcher = create_patcher(model, processor, args.model, config)

    all_profiles = []
    errors = 0
    t_start = time.time()

    for idx, sample in enumerate(pope_data[:args.n_samples]):
        question = sample.get("text") or sample.get("question", "")
        gt_answer = (sample.get("label") or sample.get("answer", "")).lower()
        image_file = sample.get("image", "")
        image_path = os.path.join(args.image_dir, image_file)

        if not os.path.exists(image_path):
            errors += 1
            continue

        img = Image.open(image_path).convert("RGB")
        prompt = question + " Answer with yes or no."

        if gt_answer == "yes":
            correct_tok, incorrect_tok = "Yes", "No"
        else:
            correct_tok, incorrect_tok = "No", "Yes"

        try:
            profile = patcher.logit_lens_profile(
                img, prompt, correct_tok, incorrect_tok)
            all_profiles.append(profile)

            if (idx + 1) % 20 == 0:
                elapsed = time.time() - t_start
                rate = (idx + 1) / elapsed
                eta = (args.n_samples - idx - 1) / rate if rate > 0 else 0
                status = "HALLUC" if profile["is_hallucination"] else "OK"
                logger.info(
                    f"[{idx+1}/{args.n_samples}] {status} | "
                    f"final_ld={profile['final_logit_diff']:+.2f} | "
                    f"ETA {eta/60:.0f}min")

        except Exception as e:
            logger.error(f"Error on sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            errors += 1

    dt = time.time() - t_start
    logger.info(f"Completed: {len(all_profiles)} samples in {dt/60:.1f}min "
                f"({errors} errors)")

    # Analyze
    analysis = analyze_logit_lens_results(all_profiles, pathway_map)

    # Print results
    model_short = args.model.split("/")[-1]
    print(f"\n{'='*70}")
    print(f"LOGIT LENS RESULTS — {model_short}")
    print(f"{'='*70}")

    for cat, data in analysis.items():
        print(f"\n[{cat.upper()}] (n={data['n_samples']})")

        pstats = data.get("pathway_stats", {})
        for ptype in ["visual", "halluc", "mixed"]:
            ps = pstats.get(ptype)
            if ps:
                print(f"  {ptype:>8s} pathway (layers {ps['layers']}): "
                      f"mean Δlogit_diff = {ps['mean_delta_logit_diff']:+.4f}")

        dt = pstats.get("differential_test")
        if dt:
            print(f"\n  Functional role test:")
            print(f"    Visual layers mean Δ:  {dt['visual_mean_delta']:+.4f}")
            print(f"    Halluc layers mean Δ:  {dt['halluc_mean_delta']:+.4f}")
            print(f"    t = {dt['t']:.3f}, p = {dt['p']:.4f}, "
                  f"d = {dt['cohen_d']:.3f}")

            if dt["p"] < 0.05 and dt["visual_mean_delta"] > dt["halluc_mean_delta"]:
                print(f"    → Visual pathway INCREASES correct answer probability")
                print(f"    → Halluc pathway DECREASES it (or increases incorrect)")
                print(f"    → DUAL-PATHWAY FUNCTIONAL ROLES CONFIRMED")
            elif dt["p"] < 0.05:
                print(f"    → Significant difference but unexpected direction")
            else:
                print(f"    → No significant difference in layer contributions")

    # Save
    save_data = {
        "model": model_short,
        "n_samples": len(all_profiles),
        "pathway_map": {str(k): v for k, v in pathway_map.items()},
        "analysis": analysis,
        "profiles": all_profiles,
    }

    out_path = os.path.join(args.output, "logit_lens.json")
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
