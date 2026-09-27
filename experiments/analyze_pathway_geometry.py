"""
Pathway Geometry: SVD / Participation-Ratio Analysis (CPU)
============================================================

Consumes geometry_acts.pt from collect_pathway_geometry.py and
produces direct geometric evidence for the central claim of
Section 5.5: that the hallucination pathway acts through MULTIPLE
distinct activation directions, while the grounding pathway carries
overlapping (low-rank) information.

Methodology
-----------
For each pathway-component (separately for halluc and grounding),
we form ΔX = X_halluc - mean(X_correct), the per-sample deviation
from the correct-class centroid. A single-direction signal would put
all the variance in σ_1²; a multi-direction signal spreads it out.

Two summary numbers per component:
  - PR (Participation Ratio):  (Σ σᵢ²)² / Σ σᵢ⁴
        ≈ effective number of variance-carrying directions.
        PR ≈ 1     ⇔  signal is essentially 1-D
        PR ≈ K     ⇔  signal occupies a K-dim subspace of similar weight
  - σ₁²/Σσᵢ² (top-1 variance fraction):
        complementary to PR; gives the share captured by the single
        best direction (the only one ITI/mean-diff can target).

Plus an effective-rank-50 / -90 (number of singular values needed
to capture 50% / 90% of the variance), and a per-pathway probe
accuracy upper-bound (a logistic regression on the FULL activation
vs. one restricted to the top-1 PC).

The PR contrast between hallucination-pathway (high) and grounding-
pathway (low) is the headline finding: it converts the negative
ITI/mean-diff result into a positive geometric statement.

Usage
-----
  python experiments/analyze_pathway_geometry.py \\
    --inputs results/geometry_qwen/geometry_acts.pt \\
             results/geometry_llava/geometry_acts.pt \\
             results/geometry_llama/geometry_acts.pt \\
             results/geometry_internvl8b/geometry_acts.pt \\
             results/geometry_internvl14b/geometry_acts.pt \\
    --output results/geometry_summary/

Outputs
-------
  per_component.csv        full per-component metrics
  per_pathway.csv          per-(model, pathway) aggregates
  spectrum_<model>.json    raw σ²-fractions for plotting
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("analyze_geometry")


def model_short_name(model_id: str) -> str:
    m = model_id.lower()
    if "qwen" in m: return "Qwen3-VL-8B"
    if "llava" in m: return "LLaVA-v1.6-7B"
    if "internvl3-8b" in m: return "InternVL3-8B"
    if "internvl3-14b" in m: return "InternVL3-14B"
    if "llama" in m: return "Llama-3.2-V-11B"
    return model_id.split("/")[-1]


def analyze_component(X_correct: np.ndarray,
                      X_halluc: np.ndarray) -> Dict:
    """SVD-based geometry of the hallucination signal at one component.

    Forms ΔX = X_halluc - μ_correct, computes its singular spectrum
    in the (n_halluc × hidden) matrix sense, and returns participation
    ratio plus related summaries.

    Returns NaNs in the degenerate case where one class has < 3 samples
    or the spectrum collapses to zero.
    """
    n_c = X_correct.shape[0]
    n_h = X_halluc.shape[0]

    if n_c < 3 or n_h < 3:
        return {"pr": np.nan, "top1_var": np.nan,
                "rank50": np.nan, "rank90": np.nan,
                "n_correct": n_c, "n_halluc": n_h,
                "spectrum": []}

    mu_c = X_correct.mean(axis=0, keepdims=True)        # (1, d)
    delta = X_halluc - mu_c                             # (n_h, d)

    # SVD; we only need singular values.
    # delta has shape (n_h, d); rank ≤ min(n_h, d).
    s = np.linalg.svd(delta, compute_uv=False)          # descending
    s2 = s ** 2

    total = float(s2.sum())
    if total < 1e-12:
        return {"pr": np.nan, "top1_var": np.nan,
                "rank50": np.nan, "rank90": np.nan,
                "n_correct": n_c, "n_halluc": n_h,
                "spectrum": []}

    pr = float((s2.sum() ** 2) / (s2 ** 2).sum())       # participation ratio
    top1_var = float(s2[0] / total)

    cum = np.cumsum(s2) / total
    rank50 = int(np.searchsorted(cum, 0.50) + 1)
    rank90 = int(np.searchsorted(cum, 0.90) + 1)

    return {
        "pr": pr,
        "top1_var": top1_var,
        "rank50": rank50,
        "rank90": rank90,
        "n_correct": n_c,
        "n_halluc": n_h,
        # First 30 normalized variances; enough for a nice plot,
        # avoids bloating per_component.csv.
        "spectrum": (s2[:30] / total).tolist(),
    }


def aggregate_pathway(rows: List[Dict]) -> Dict:
    """Average geometry metrics over all components in a pathway."""
    valid = [r for r in rows if not np.isnan(r["pr"])]
    if not valid:
        return {"n_components": 0,
                "pr_mean": np.nan, "pr_median": np.nan,
                "top1_var_mean": np.nan,
                "rank50_mean": np.nan, "rank90_mean": np.nan}
    prs = [r["pr"] for r in valid]
    t1s = [r["top1_var"] for r in valid]
    r50 = [r["rank50"] for r in valid]
    r90 = [r["rank90"] for r in valid]
    return {
        "n_components": len(valid),
        "pr_mean": float(np.mean(prs)),
        "pr_median": float(np.median(prs)),
        "pr_std": float(np.std(prs)),
        "top1_var_mean": float(np.mean(t1s)),
        "top1_var_median": float(np.median(t1s)),
        "rank50_mean": float(np.mean(r50)),
        "rank90_mean": float(np.mean(r90)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="One or more geometry_acts.pt files")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    per_component_rows: List[Dict] = []
    per_pathway_rows: List[Dict] = []
    spectra_by_model: Dict[str, Dict] = {}

    for path in args.inputs:
        logger.info(f"Loading {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model_id = payload["meta"]["model"]
        model_short = model_short_name(model_id)
        logger.info(f"  Model: {model_short}")
        logger.info(f"  N halluc components: {len(payload['halluc'])}")
        logger.info(f"  N ground components: {len(payload['ground'])}")

        spectra_by_model[model_short] = {"halluc": {}, "ground": {}}

        for pathway_name in ["halluc", "ground"]:
            pathway_rows: List[Dict] = []
            for comp_id, blob in payload[pathway_name].items():
                X_c = blob["correct"].numpy()
                X_h = blob["halluc"].numpy()

                geom = analyze_component(X_c, X_h)

                row = {
                    "model": model_short,
                    "pathway": pathway_name,
                    "component": comp_id,
                    "layer": blob["layer"],
                    "comp_type": blob["comp_type"],
                    **{k: v for k, v in geom.items() if k != "spectrum"},
                }
                per_component_rows.append(row)
                pathway_rows.append(geom)
                spectra_by_model[model_short][pathway_name][comp_id] = geom["spectrum"]

            agg = aggregate_pathway(pathway_rows)
            per_pathway_rows.append({
                "model": model_short,
                "pathway": pathway_name,
                **agg,
            })
            logger.info(f"  {pathway_name:7s}: "
                        f"PR mean={agg['pr_mean']:.2f} "
                        f"median={agg['pr_median']:.2f}  "
                        f"top1_var mean={agg['top1_var_mean']:.3f}  "
                        f"rank90 mean={agg['rank90_mean']:.1f}")

        # Save spectra per model for plotting.
        spec_path = os.path.join(args.output,
                                 f"spectrum_{model_short}.json")
        with open(spec_path, "w") as f:
            json.dump(spectra_by_model[model_short], f, indent=2)

    # Write per-component CSV.
    pc_path = os.path.join(args.output, "per_component.csv")
    if per_component_rows:
        keys = ["model", "pathway", "component", "layer", "comp_type",
                "pr", "top1_var", "rank50", "rank90",
                "n_correct", "n_halluc"]
        with open(pc_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in per_component_rows:
                w.writerow({k: r.get(k, "") for k in keys})
    logger.info(f"Wrote {pc_path}")

    # Write per-pathway CSV.
    pp_path = os.path.join(args.output, "per_pathway.csv")
    if per_pathway_rows:
        keys = ["model", "pathway", "n_components",
                "pr_mean", "pr_median", "pr_std",
                "top1_var_mean", "top1_var_median",
                "rank50_mean", "rank90_mean"]
        with open(pp_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in per_pathway_rows:
                w.writerow({k: r.get(k, "") for k in keys})
    logger.info(f"Wrote {pp_path}")

    # Print headline contrast table.
    print("\n" + "=" * 72)
    print("HEADLINE: per-pathway PR mean (effective # of directions)")
    print("=" * 72)
    print(f"{'Model':<22} {'pathway':<10} "
          f"{'PR mean':>9} {'PR med':>9} {'top1 var':>10} {'rank90':>8}")
    print("-" * 72)
    by_model: Dict[str, Dict[str, Dict]] = {}
    for r in per_pathway_rows:
        by_model.setdefault(r["model"], {})[r["pathway"]] = r
    for model_short, paths in by_model.items():
        for pname in ["ground", "halluc"]:
            if pname not in paths: continue
            r = paths[pname]
            print(f"{model_short:<22} {pname:<10} "
                  f"{r['pr_mean']:>9.2f} {r['pr_median']:>9.2f} "
                  f"{r['top1_var_mean']:>10.3f} {r['rank90_mean']:>8.1f}")
        print()


if __name__ == "__main__":
    main()
