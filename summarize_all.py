#!/usr/bin/env python3
"""
Comprehensive Results Summarizer
=================================
Auto-discovers and summarizes ALL experimental results for the paper.

Covers:
  - Phase 1: Circuit extraction (patching) → Table 1, Table 2
  - Phase 2: CPA (pathway interaction) → Table 3
  - Phase 3: Logit lens validation → Table 4
  - Phase 4: Cross-architecture comparison → Table 5
  - Phase 5: Intervention (POPE-adv) → Table 6
  - Phase 6: LLaVA top-k ablation → Table 7
  - Phase 7: Cross-benchmark (POPE splits + AMBER) → Table 8

Usage:
  PYTHONPATH=. python summarize_all.py
  PYTHONPATH=. python summarize_all.py --results results/ --output results/full_summary.json
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

SEP = "=" * 80
THIN = "─" * 80


def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, Exception):
        return None


def find_files(base, pattern):
    """Find files matching glob pattern under base directory."""
    return sorted(glob.glob(os.path.join(base, pattern), recursive=True))


def auto_discover(results_dir):
    """Discover all result files and categorize them."""
    discovery = {
        "circuits": [],      # circuit.json files
        "cpa": [],           # pathway_interaction.json files
        "logit_lens": [],    # logit lens results
        "crossarch": [],     # cross-architecture comparison
        "intervention": [],  # intervention_results.json files
        "topk": [],          # topk_ablation_results.json files
        "multi_bench": [],   # multi-benchmark results
        "chair": [],         # CHAIR results
    }

    for root, dirs, files in os.walk(results_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, results_dir)

            if fname == "circuit.json":
                discovery["circuits"].append(fpath)
            elif fname == "pathway_interaction.json":
                discovery["cpa"].append(fpath)
            elif "logit_lens" in fname or "logitlens" in rel.lower():
                if fname.endswith(".json"):
                    discovery["logit_lens"].append(fpath)
            elif "crossarch" in rel.lower() and fname.endswith(".json"):
                discovery["crossarch"].append(fpath)
            elif fname == "topk_ablation_results.json":
                discovery["topk"].append(fpath)
            elif fname == "intervention_results.json":
                # Distinguish multi-benchmark from single
                if "multi_benchmark" in rel or "pope_" in rel or "amber" in rel:
                    discovery["multi_bench"].append(fpath)
                else:
                    discovery["intervention"].append(fpath)
            elif "chair" in rel.lower() and fname.endswith(".json"):
                discovery["chair"].append(fpath)

    return discovery


def print_section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ═══════════════════════════════════════════════════════════════
# PHASE 1: CIRCUIT EXTRACTION
# ═══════════════════════════════════════════════════════════════

def summarize_circuits(files):
    print_section("PHASE 1: CIRCUIT EXTRACTION (Tables 1 & 2)")

    results = []
    for fpath in files:
        data = load_json(fpath)
        if not data:
            continue

        parent = os.path.basename(os.path.dirname(fpath))
        model = data.get("model", parent)

        components = data.get("components", [])
        # Filter to component-level (not head-level)
        comps = [c for c in components if c.get("type") != "head"]
        grnd = [c for c in comps if c["cohen_d"] > 0]
        hall = [c for c in comps if c["cohen_d"] < 0]

        n_samples = data.get("n_samples", "?")
        halluc_rate = data.get("halluc_rate", data.get("hallucination_rate", "?"))

        print(f"\n  {model} ({parent})")
        print(f"    samples={n_samples}  halluc_rate={halluc_rate}")
        print(f"    significant components: {len(comps)} "
              f"(grounding={len(grnd)}, hallucination={len(hall)})")

        # Top components by |d|
        top = sorted(comps, key=lambda x: abs(x["cohen_d"]), reverse=True)[:5]
        for c in top:
            d = c["cohen_d"]
            pathway = "grounding" if d > 0 else "halluc"
            print(f"      L{c['layer']}.{c['type']:4s}  d={d:+.2f}  [{pathway}]")

        results.append({
            "model": model, "dir": parent,
            "n_components": len(comps),
            "n_grounding": len(grnd), "n_halluc": len(hall),
            "path": fpath,
        })

    return results


# ═══════════════════════════════════════════════════════════════
# PHASE 2: CPA
# ═══════════════════════════════════════════════════════════════

def summarize_cpa(files):
    print_section("PHASE 2: Conditional Pathway Analysis (Table 3)")

    for fpath in files:
        data = load_json(fpath)
        if not data:
            continue

        parent = os.path.basename(os.path.dirname(fpath))
        print(f"\n  {parent}")

        for pname in ["visual", "halluc", "grounding", "hallucination"]:
            if pname not in data:
                continue
            pw = data[pname]
            k = pw.get("n_components", "?")
            sc = pw.get("stats_correct", {})
            sh = pw.get("stats_halluc", {})
            diff = pw.get("differential", {})

            print(f"    [{pname.upper()} pathway]  k={k}")
            if sc:
                print(f"      correct:  interaction={sc.get('mean',0):+.2f}  "
                      f"d={sc.get('cohen_d',0):.2f}  p={sc.get('p',1):.4f}")
            if sh:
                print(f"      halluc:   interaction={sh.get('mean',0):+.2f}  "
                      f"d={sh.get('cohen_d',0):.2f}  p={sh.get('p',1):.4f}")
            if diff:
                print(f"      differential: d={diff.get('d',0):.2f}  "
                      f"p={diff.get('p',1):.4f}")


# ═══════════════════════════════════════════════════════════════
# PHASE 5: INTERVENTION (POPE-adv only, Table 6)
# ═══════════════════════════════════════════════════════════════

def summarize_intervention(files):
    print_section("PHASE 5: INTERVENTION — POPE-adv (Table 6)")

    print(f"\n  {'Model':<25s} {'Config':<20s} {'Acc':>8s} {'Hall':>8s} "
          f"{'ΔAcc':>8s} {'ΔHall':>8s} {'Rel↓':>8s}")
    print(f"  {THIN}")

    for fpath in files:
        data = load_json(fpath)
        if not data:
            continue

        parent = os.path.basename(os.path.dirname(fpath))

        # Handle nested benchmark structure
        for bench_name, bdata in data.items():
            if not isinstance(bdata, dict) or "baseline" not in bdata:
                continue

            bl = bdata["baseline"]
            ba = bl.get("accuracy", 0)
            bh = bl.get("halluc_rate", 0)
            print(f"  {parent:<25s} {'baseline':<20s} {ba:>8.1%} {bh:>8.1%} "
                  f"{'':>8s} {'':>8s} {'':>8s}")

            for k, v in bdata.items():
                if not k.startswith("targeted"):
                    continue
                ta = v.get("accuracy", 0)
                th = v.get("halluc_rate", 0)
                da = ta - ba
                dh = th - bh
                rel = (dh / bh * 100) if bh > 0 else 0
                s = v.get("scale", "?")
                label = f"s={s}"
                print(f"  {'':<25s} {label:<20s} {ta:>8.1%} {th:>8.1%} "
                      f"{da:>+8.1%} {dh:>+8.1%} {rel:>7.0f}%")


# ═══════════════════════════════════════════════════════════════
# PHASE 6: TOP-K ABLATION (Table 7)
# ═══════════════════════════════════════════════════════════════

def summarize_topk(files):
    print_section("PHASE 6: TOP-K ABLATION (Table 7)")

    for fpath in files:
        data = load_json(fpath)
        if not data:
            continue

        parent = os.path.basename(os.path.dirname(fpath))
        print(f"\n  {parent}")

        for bench_name, bdata in data.items():
            if not isinstance(bdata, dict) or "baseline" not in bdata:
                continue

            bl = bdata["baseline"]
            ba = bl.get("accuracy", 0)
            bh = bl.get("halluc_rate", 0)
            print(f"    [{bench_name}]")
            print(f"      {'k':<10s} {'Acc':>8s} {'Hall':>8s} {'ΔAcc':>8s} {'ΔHall':>8s}")
            print(f"      {'baseline':<10s} {ba:>8.1%} {bh:>8.1%}")

            for k in sorted(bdata.keys()):
                if not k.startswith("top_"):
                    continue
                v = bdata[k]
                ta = v.get("accuracy", 0)
                th = v.get("halluc_rate", 0)
                da = ta - ba
                dh = th - bh
                print(f"      {k:<10s} {ta:>8.1%} {th:>8.1%} {da:>+8.1%} {dh:>+8.1%}")


# ═══════════════════════════════════════════════════════════════
# PHASE 7: CROSS-BENCHMARK (Table 8)
# ═══════════════════════════════════════════════════════════════

def summarize_multi_benchmark(files, results_dir):
    print_section("PHASE 7: CROSS-BENCHMARK GENERALIZATION (Table 8)")

    # Also try the dedicated summary file
    summary_path = os.path.join(results_dir, "multi_benchmark", "summary_table.json")
    summary = load_json(summary_path)

    if summary:
        # Use the pre-computed summary
        all_benches = set()
        for model_data in summary.values():
            all_benches.update(model_data.keys())
        benches = sorted(all_benches)

        for bench in benches:
            print(f"\n  {bench}:")
            print(f"    {'Model':<25s} {'Base Acc':>10s} {'Base Hall':>10s} "
                  f"{'Int Acc':>10s} {'Int Hall':>10s} "
                  f"{'ΔAcc':>8s} {'ΔHall':>8s} {'Rel↓':>8s}")
            print(f"    {THIN}")

            for model_key in sorted(summary.keys()):
                if bench not in summary[model_key]:
                    continue
                r = summary[model_key][bench]
                ba = r["baseline_acc"]
                bh = r["baseline_halluc"]
                ta = r["targeted_acc"]
                th = r["targeted_halluc"]
                da = ta - ba
                dh = th - bh
                rel = (dh / bh * 100) if bh > 0 else 0
                print(f"    {model_key:<25s} {ba:>10.1%} {bh:>10.1%} "
                      f"{ta:>10.1%} {th:>10.1%} "
                      f"{da:>+8.1%} {dh:>+8.1%} {rel:>7.0f}%")
        return

    # Fallback: parse individual files
    if not files:
        print("\n  No multi-benchmark results found.")
        print("  Run: bash run_multi_benchmark.sh")
        return

    # Group by model
    by_model = defaultdict(dict)
    for fpath in files:
        data = load_json(fpath)
        if not data:
            continue
        # Infer model from path: multi_benchmark/<model>/<bench>/intervention_results.json
        parts = fpath.split(os.sep)
        model_key = None
        bench_key = None
        for i, p in enumerate(parts):
            if p == "multi_benchmark" and i + 2 < len(parts):
                model_key = parts[i + 1]
                bench_key = parts[i + 2]
                break

        if not model_key:
            continue

        for bname, bdata in data.items():
            if not isinstance(bdata, dict) or "baseline" not in bdata:
                continue
            bl = bdata["baseline"]
            targeted = None
            for k, v in bdata.items():
                if "targeted" in k:
                    targeted = v
                    break
            if targeted:
                by_model[model_key][bench_key or bname] = {
                    "baseline_acc": bl.get("accuracy", 0),
                    "baseline_halluc": bl.get("halluc_rate", 0),
                    "targeted_acc": targeted.get("accuracy", 0),
                    "targeted_halluc": targeted.get("halluc_rate", 0),
                }

    # Print
    all_benches = set()
    for md in by_model.values():
        all_benches.update(md.keys())

    for bench in sorted(all_benches):
        print(f"\n  {bench}:")
        print(f"    {'Model':<25s} {'Base Acc':>10s} {'Base Hall':>10s} "
              f"{'Int Acc':>10s} {'Int Hall':>10s} "
              f"{'ΔHall':>8s} {'Rel↓':>8s}")
        print(f"    {THIN}")
        for mk in sorted(by_model.keys()):
            if bench not in by_model[mk]:
                continue
            r = by_model[mk][bench]
            ba = r["baseline_acc"]
            bh = r["baseline_halluc"]
            ta = r["targeted_acc"]
            th = r["targeted_halluc"]
            dh = th - bh
            rel = (dh / bh * 100) if bh > 0 else 0
            print(f"    {mk:<25s} {ba:>10.1%} {bh:>10.1%} "
                  f"{ta:>10.1%} {th:>10.1%} "
                  f"{dh:>+8.1%} {rel:>7.0f}%")


# ═══════════════════════════════════════════════════════════════
# FILE INVENTORY
# ═══════════════════════════════════════════════════════════════

def print_inventory(discovery):
    print_section("FILE INVENTORY")
    total = 0
    for category, files in discovery.items():
        if files:
            print(f"\n  {category} ({len(files)} files):")
            for f in files:
                size = os.path.getsize(f) / 1024
                print(f"    {os.path.relpath(f, 'results/'):<60s} ({size:.1f}KB)")
                total += 1
    print(f"\n  Total: {total} result files found")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Summarize all experiment results")
    parser.add_argument("--results", type=str, default="results/",
                        help="Root results directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Save JSON summary to this path")
    parser.add_argument("--inventory-only", action="store_true",
                        help="Only print file inventory")
    args = parser.parse_args()

    print(f"\n{'#' * 80}")
    print(f"#  COMPREHENSIVE RESULTS SUMMARY")
    print(f"#  Results dir: {os.path.abspath(args.results)}")
    print(f"{'#' * 80}")

    discovery = auto_discover(args.results)

    if args.inventory_only:
        print_inventory(discovery)
        return

    # Run all summaries
    if discovery["circuits"]:
        summarize_circuits(discovery["circuits"])

    if discovery["cpa"]:
        summarize_cpa(discovery["cpa"])

    if discovery["intervention"]:
        summarize_intervention(discovery["intervention"])

    if discovery["topk"]:
        summarize_topk(discovery["topk"])

    if discovery["multi_bench"] or os.path.exists(
            os.path.join(args.results, "multi_benchmark", "summary_table.json")):
        summarize_multi_benchmark(discovery["multi_bench"], args.results)

    print_inventory(discovery)

    print(f"\n{'#' * 80}")
    print(f"#  DONE")
    print(f"{'#' * 80}\n")


if __name__ == "__main__":
    main()