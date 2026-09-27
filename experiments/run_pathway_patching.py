"""
Conditional Pathway Analysis (CPA): Pathway-Level Patching
==========================================================

Extends single-component activation patching to pathway-level analysis.
For each identified pathway (visual grounding, hallucination), measures:

  1. Joint effect:    IE(pathway) — restore ALL pathway components at once
  2. Sum individual:  Σ IE(ci)   — sum of restoring each component alone
  3. Interaction:     IE(pathway) - Σ IE(ci)
     > 0 → synergy   (pathway is more than sum of parts)
     < 0 → redundancy (components share information)
     ≈ 0 → independence (components act in isolation)

A significant interaction term proves the pathway is a genuine functional
unit—not just a statistical grouping of independently-acting components.
This is the key methodological novelty over vanilla activation patching.

Usage:
  # Requires a circuit.json from run_patching.py
  PYTHONPATH=. python experiments/run_pathway_patching.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --circuit results/qwen3vl_component/circuit.json \
    --pope-path data/pope/ --image-dir data/coco/val2014/ \
    --split adversarial --n-samples 200 \
    --output results/pathway_qwen/

  # All 3 models
  for m in "Qwen/Qwen3-VL-8B-Instruct" "llava-hf/llava-v1.6-mistral-7b-hf" \
           "OpenGVLab/InternVL3-8B-hf"; do
    short=$(echo $m | tr '/' '_')
    PYTHONPATH=. python experiments/run_pathway_patching.py \
      --model "$m" \
      --circuit "results/${short}_component/circuit.json" \
      --pope-path data/pope/ --image-dir data/coco/val2014/ \
      --n-samples 200 --output "results/pathway_${short}/"
  done
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("pathway_patching")


# ════════════════════════════════════════════════════════════════════
# CIRCUIT → PATHWAY TARGETS
# ════════════════════════════════════════════════════════════════════

def load_pathway_targets(circuit_path: str,
                          min_abs_d: float = 0.3,
                          ) -> Dict[str, List[Tuple[int, str, int]]]:
    """Load circuit.json and partition into hallucination / visual pathways.

    Returns dict with keys "halluc" and "visual", each mapping to a list
    of (layer_idx, component_type, head_idx) tuples.
    """
    with open(circuit_path) as f:
        data = json.load(f)

    pathways = {"halluc": [], "visual": []}

    for c in data["components"]:
        d = c["cohen_d"]
        ctype = c["type"]

        # Skip head-level components (we do pathway at component level)
        if ctype == "head":
            continue
        if abs(d) < min_abs_d:
            continue

        # head_idx = -1 for attn/mlp components
        target = (c["layer"], ctype, -1)

        if d < 0:
            pathways["halluc"].append(target)
        else:
            pathways["visual"].append(target)

    logger.info(f"Loaded pathways from {circuit_path}:")
    logger.info(f"  Hallucination: {len(pathways['halluc'])} components")
    logger.info(f"  Visual ground: {len(pathways['visual'])} components")

    return pathways


# ════════════════════════════════════════════════════════════════════
# PROGRESSIVE PATCHING (subsets of increasing size)
# ════════════════════════════════════════════════════════════════════

def build_progressive_targets(
    targets: List[Tuple[int, str, int]],
    circuit_data: dict,
) -> List[List[Tuple[int, str, int]]]:
    """Order targets by |Cohen's d| and build cumulative subsets.

    Returns list of target lists: [{top-1}, {top-2}, ..., {all}]
    This allows measuring how the joint effect grows as components
    are added, revealing diminishing returns or synergy.
    """
    # Get d values for ordering
    comp_d = {}
    for c in circuit_data["components"]:
        if c["type"] == "head":
            continue
        key = (c["layer"], c["type"], -1)
        comp_d[key] = abs(c["cohen_d"])

    # Sort by |d| descending
    ordered = sorted(targets, key=lambda t: comp_d.get(t, 0), reverse=True)

    # Build cumulative subsets
    subsets = []
    for i in range(1, len(ordered) + 1):
        subsets.append(ordered[:i])

    return subsets


# ════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ════════════════════════════════════════════════════════════════════

def run_pathway_experiment(
    model, processor, model_id: str,
    pope_data: list, image_dir: str,
    pathways: Dict[str, List[Tuple[int, str, int]]],
    circuit_data: dict,
    output_dir: str,
    n_samples: int = 200,
):
    """Run pathway interaction analysis on POPE samples."""
    import torch
    from PIL import Image
    from patching.core import PatchingConfig, PathwayInteractionResult
    from experiments.run_patching import create_patcher

    os.makedirs(output_dir, exist_ok=True)

    config = PatchingConfig(granularity="component")
    patcher = create_patcher(model, processor, model_id, config)

    all_results = {}

    for pathway_name, targets in pathways.items():
        if len(targets) < 2:
            logger.warning(f"Pathway '{pathway_name}' has <2 components, "
                           "skipping interaction analysis")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"PATHWAY: {pathway_name} ({len(targets)} components)")
        logger.info(f"{'='*60}")
        for l, c, h in targets:
            logger.info(f"  L{l}.{c}")

        result = PathwayInteractionResult(
            pathway_name=pathway_name,
            n_components=len(targets),
            targets=targets,
        )

        # Also build progressive subsets for cumulative analysis
        progressive = build_progressive_targets(targets, circuit_data)

        t_start = time.time()
        errors = 0

        for idx, sample in enumerate(pope_data[:n_samples]):
            question = sample.get("text") or sample.get("question", "")
            gt_answer = (sample.get("label") or sample.get("answer", "")
                         ).lower()
            image_file = sample.get("image", "")
            image_path = os.path.join(image_dir, image_file)

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
                r = patcher.patch_pathway_interaction(
                    img, prompt,
                    correct_token=correct_tok,
                    incorrect_token=incorrect_tok,
                    pathway_targets=targets,
                    gt_answer=gt_answer,
                )
                result.sample_results.append(r)

                if (idx + 1) % 20 == 0:
                    elapsed = time.time() - t_start
                    rate = (idx + 1) / elapsed
                    eta = (n_samples - idx - 1) / rate if rate > 0 else 0
                    status = "HALLUC" if r["is_hallucination"] else "OK"
                    logger.info(
                        f"  [{idx+1}/{n_samples}] {status} | "
                        f"joint={r['joint_ie']:+.3f} "
                        f"Σind={r['sum_individual_ie']:+.3f} "
                        f"inter={r['interaction']:+.3f} | "
                        f"ETA {eta/60:.0f}min")

            except Exception as e:
                logger.error(f"Error on sample {idx}: {e}")
                import traceback
                traceback.print_exc()
                errors += 1

        # ── Summary ──────────────────────────────────────────────
        dt = time.time() - t_start

        print(f"\n{'─'*60}")
        print(f"PATHWAY: {pathway_name}")
        print(f"{'─'*60}")
        print(f"  Samples: {result.n_samples} "
              f"({result.n_correct} correct, {result.n_halluc} halluc)")
        print(f"  Time: {dt/60:.1f} min")

        for cat in ["all", "correct", "hallucination"]:
            stats = result.interaction_stats(cat)
            print(f"\n  [{cat.upper()}] (n={stats['n']})")
            print(f"    Mean interaction:  {stats['mean']:+.4f} "
                  f"± {stats['std']:.4f}")
            print(f"    t-test vs 0:       t={stats['t']:.3f}, "
                  f"p={stats['p']:.4f}")
            print(f"    Cohen's d:         {stats['cohen_d']:.3f}")

            if abs(stats["cohen_d"]) > 0.2 and stats["p"] < 0.05:
                direction = "SYNERGY" if stats["mean"] > 0 else "REDUNDANCY"
                print(f"    → Significant {direction}")
            else:
                print(f"    → No significant interaction")

        # Differential interaction
        diff = result.differential_interaction()
        print(f"\n  DIFFERENTIAL (correct vs halluc):")
        print(f"    Mean correct: {diff['mean_correct']:+.4f}")
        print(f"    Mean halluc:  {diff['mean_halluc']:+.4f}")
        print(f"    t={diff['t']:.3f}, p={diff['p']:.4f}, d={diff['d']:.3f}")

        all_results[pathway_name] = result

        # ── Progressive analysis ─────────────────────────────────
        if len(progressive) > 1:
            print(f"\n  PROGRESSIVE ANALYSIS (cumulative components):")
            print(f"  {'N':>4s}  {'Joint IE':>10s}  {'Σ Indiv':>10s}  "
                  f"{'Interact':>10s}  {'Norm':>8s}")

            # Run progressive on a subset of samples for efficiency
            prog_samples = min(50, len(pope_data))
            for subset_idx, subset_targets in enumerate(progressive):
                joint_ies, sum_inds, interactions = [], [], []

                for s_idx, sample in enumerate(pope_data[:prog_samples]):
                    question = sample.get("text") or sample.get("question", "")
                    gt_answer = (sample.get("label") or sample.get("answer", "")
                                 ).lower()
                    image_file = sample.get("image", "")
                    image_path = os.path.join(image_dir, image_file)
                    if not os.path.exists(image_path):
                        continue

                    img = Image.open(image_path).convert("RGB")
                    prompt = question + " Answer with yes or no."
                    ct = "Yes" if gt_answer == "yes" else "No"
                    it = "No" if gt_answer == "yes" else "Yes"

                    try:
                        r = patcher.patch_pathway_interaction(
                            img, prompt, ct, it, subset_targets, gt_answer)
                        joint_ies.append(r["joint_ie"])
                        sum_inds.append(r["sum_individual_ie"])
                        interactions.append(r["interaction"])
                    except Exception:
                        pass

                if joint_ies:
                    mj = np.mean(joint_ies)
                    ms = np.mean(sum_inds)
                    mi = np.mean(interactions)
                    mn = mi / abs(mj) if abs(mj) > 1e-8 else 0.0
                    print(f"  {len(subset_targets):>4d}  {mj:>+10.4f}  "
                          f"{ms:>+10.4f}  {mi:>+10.4f}  {mn:>+8.3f}")

    # ── Save all results ─────────────────────────────────────────
    save_data = {}
    for pname, res in all_results.items():
        save_data[pname] = {
            "pathway_name": pname,
            "n_components": res.n_components,
            "targets": [(l, c, h) for l, c, h in res.targets],
            "n_samples": res.n_samples,
            "n_correct": res.n_correct,
            "n_halluc": res.n_halluc,
            "stats_all": res.interaction_stats("all"),
            "stats_correct": res.interaction_stats("correct"),
            "stats_halluc": res.interaction_stats("hallucination"),
            "differential": res.differential_interaction(),
            "sample_results": res.sample_results,
        }

    out_path = os.path.join(output_dir, "pathway_interaction.json")
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    logger.info(f"Saved to {out_path}")

    # ── LaTeX table snippet ──────────────────────────────────────
    tex_path = os.path.join(output_dir, "pathway_table.tex")
    with open(tex_path, "w") as f:
        f.write("% Pathway Interaction Results\n")
        f.write("% Auto-generated by run_pathway_patching.py\n\n")
        f.write("\\begin{tabular}{llcccc}\n\\toprule\n")
        f.write("\\textbf{Model} & \\textbf{Pathway} & "
                "\\textbf{IE(joint)} & $\\sum$ \\textbf{IE(ind)} & "
                "\\textbf{Interaction} & \\textbf{$p$} \\\\\n\\midrule\n")

        model_short = model_id.split("/")[-1]
        for pname, res in all_results.items():
            stats = res.interaction_stats("all")
            f.write(f"{model_short} & {pname} & "
                    f"{np.mean([r['joint_ie'] for r in res.sample_results]):.3f} & "
                    f"{np.mean([r['sum_individual_ie'] for r in res.sample_results]):.3f} & "
                    f"{stats['mean']:+.3f} & "
                    f"{stats['p']:.4f} \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n")

    logger.info(f"LaTeX saved to {tex_path}")

    return all_results


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Conditional Pathway Analysis: pathway-level patching")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--circuit", type=str, required=True,
                        help="Path to circuit.json from run_patching.py")
    parser.add_argument("--pope-path", type=str, default="data/pope/")
    parser.add_argument("--image-dir", type=str, default="data/coco/val2014/")
    parser.add_argument("--split", type=str, default="adversarial")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--min-d", type=float, default=0.3,
                        help="Minimum |Cohen's d| for pathway membership")
    parser.add_argument("--output", type=str,
                        default="results/pathway/")

    args = parser.parse_args()

    from experiments.run_patching import load_model, load_pope
    model, processor = load_model(args.model)
    pope_data = load_pope(args.pope_path, split=args.split,
                          n_samples=args.n_samples)

    pathways = load_pathway_targets(args.circuit, min_abs_d=args.min_d)

    with open(args.circuit) as f:
        circuit_data = json.load(f)

    run_pathway_experiment(
        model, processor, args.model,
        pope_data, args.image_dir,
        pathways, circuit_data,
        args.output,
        n_samples=args.n_samples,
    )


if __name__ == "__main__":
    main()
