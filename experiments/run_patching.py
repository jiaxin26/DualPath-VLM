"""
Run Activation Patching Experiments

Usage:
  # POPE 1000 samples with checkpointing (resumes if interrupted)
  PYTHONPATH=. python experiments/run_patching.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --benchmark pope --split adversarial --n-samples 1000 \
    --granularity component \
    --image-dir data/coco/val2014/ \
    --data-path data/pope/ \
    --output results/qwen3vl_v2/ \
    --ckpt-interval 50

  # MME existence benchmark
  PYTHONPATH=. python experiments/run_patching.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --benchmark mme --n-samples 1000 \
    --image-dir data/mme/images/ \
    --data-path data/mme/ \
    --output results/qwen3vl_mme/

  # Resume interrupted run (just re-run the same command)
  PYTHONPATH=. python experiments/run_patching.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --benchmark pope --split adversarial --n-samples 1000 \
    --output results/qwen3vl_v2/   # detects checkpoint, resumes

  # Cross-architecture comparison (unchanged)
  PYTHONPATH=. python experiments/run_patching.py \
    --compare results/qwen3vl_v2/ results/llava_v2/ \
    --output results/comparison_v2/
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("run_patching")


# ════════════════════════════════════════════════════════════════════
# DATA LOADING (backward-compatible + multi-benchmark)
# ════════════════════════════════════════════════════════════════════

def load_data(benchmark: str, data_path: str, split: str = "adversarial",
              n_samples: Optional[int] = None) -> List[dict]:
    """Load data from any supported benchmark.

    Returns list of dicts with keys: image, question, answer, benchmark.
    Backward-compatible: samples also have 'text' and 'label' keys.
    """
    from experiments.benchmarks import load_benchmark
    samples = load_benchmark(benchmark, data_path,
                             n_samples=n_samples, split=split)
    # Add backward-compatible keys
    for s in samples:
        if "text" not in s:
            s["text"] = s["question"]
        if "label" not in s:
            s["label"] = s["answer"]
    return samples


def load_pope(pope_path: str, split: str = "adversarial",
              n_samples: Optional[int] = None) -> List[dict]:
    return load_data("pope", pope_path, split=split, n_samples=n_samples)


def load_model(model_id: str, device_map: str = "auto",
               compile_model: bool = False):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    logger.info(f"Loading model: {model_id}")
    attn_impl = "sdpa"
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        logger.info("Using FlashAttention-2")
    except ImportError:
        logger.info("Using SDPA attention")

    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16,
        device_map=device_map, attn_implementation=attn_impl,
        trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id,
                                              trust_remote_code=True)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info(f"Model loaded: {n_params:.1f}B params")

    if compile_model:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("torch.compile enabled (reduce-overhead mode)")
        except Exception as e:
            logger.warning(f"torch.compile failed, continuing without: {e}")

    return model, processor


def create_patcher(model, processor, model_id: str, config):
    """Create the appropriate patcher based on model architecture."""
    model_lower = model_id.lower()

    if "qwen" in model_lower:
        from patching.qwen_patcher import QwenActivationPatcher
        return QwenActivationPatcher(model, processor, config)

    elif "llava" in model_lower and "llama" in model_lower:
        from patching.llama_patcher import LLaMAActivationPatcher
        return LLaMAActivationPatcher(model, processor, config)

    elif "llava" in model_lower:
        from patching.llava_patcher import LLaVAActivationPatcher
        return LLaVAActivationPatcher(model, processor, config)

    elif "internvl" in model_lower:
        from patching.internvl_patcher import InternVLActivationPatcher
        return InternVLActivationPatcher(model, processor, config)

    elif "llama" in model_lower or "mllama" in model_lower:
        from patching.llama_patcher import Llama32VisionPatcher
        return Llama32VisionPatcher(model, processor, config)

    else:
        logger.warning(f"Unknown model {model_id}, trying Qwen adapter")
        from patching.qwen_patcher import QwenActivationPatcher
        return QwenActivationPatcher(model, processor, config)


# ════════════════════════════════════════════════════════════════════
# CHECKPOINT / RESUME
# ════════════════════════════════════════════════════════════════════

CKPT_FILENAME = "_checkpoint.pkl"


def save_checkpoint(output_dir: str, results: list, completed_indices: list,
                    meta: dict):
    """Save incremental checkpoint.

    Uses atomic write (write to tmp then rename) to avoid corruption.
    """
    ckpt_path = os.path.join(output_dir, CKPT_FILENAME)
    tmp_path = ckpt_path + ".tmp"
    payload = {
        "results": results,
        "completed_indices": completed_indices,
        "meta": meta,
        "timestamp": time.time(),
    }
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, ckpt_path)  # atomic on POSIX
    logger.info(f"Checkpoint saved: {len(results)} results → {ckpt_path}")


def load_checkpoint(output_dir: str, meta: dict) -> tuple:
    """Load checkpoint if it exists and matches current run config.

    Returns (results_list, completed_indices_set) or ([], set()) if no
    valid checkpoint.
    """
    ckpt_path = os.path.join(output_dir, CKPT_FILENAME)
    if not os.path.exists(ckpt_path):
        return [], set()

    try:
        with open(ckpt_path, "rb") as f:
            payload = pickle.load(f)
    except Exception as e:
        logger.warning(f"Checkpoint load failed: {e}. Starting fresh.")
        return [], set()

    # Validate: same model, benchmark, granularity
    ckpt_meta = payload.get("meta", {})
    for key in ("model", "benchmark", "granularity", "split"):
        if ckpt_meta.get(key) != meta.get(key):
            logger.warning(
                f"Checkpoint mismatch on '{key}': "
                f"{ckpt_meta.get(key)} vs {meta.get(key)}. Starting fresh.")
            return [], set()

    results = payload["results"]
    completed = set(payload.get("completed_indices", []))
    logger.info(
        f"Resuming from checkpoint: {len(results)} results completed")
    return results, completed


# ════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ════════════════════════════════════════════════════════════════════

def run_patching_experiment(
    model, processor, model_id: str,
    data: List[dict],
    image_dir: str, output_dir: str,
    granularity: str = "component",
    corrupt_mode: str = "noise",
    noise_std_mult: float = 3.0,
    benchmark: str = "pope",
    split: str = "adversarial",
    ckpt_interval: int = 50,
):
    """
    Saves checkpoint every ckpt_interval samples. If the output_dir
    contains a valid checkpoint from a matching run, resumes from it.
    """
    import torch
    from PIL import Image
    from patching.core import PatchingConfig, BatchPatchingResult, PatchingResult
    from analysis.circuits import extract_circuit, save_circuit

    os.makedirs(output_dir, exist_ok=True)

    config = PatchingConfig(
        corrupt_mode=corrupt_mode,
        noise_std_mult=noise_std_mult,
        granularity=granularity,
    )
    patcher = create_patcher(model, processor, model_id, config)

    # Check for checkpoint
    run_meta = {
        "model": model_id,
        "benchmark": benchmark,
        "split": split,
        "granularity": granularity,
        "corrupt_mode": corrupt_mode,
        "noise_std_mult": noise_std_mult,
        "n_samples_requested": len(data),
    }
    results, completed_set = load_checkpoint(output_dir, run_meta)
    completed_indices = list(completed_set)

    errors = 0
    t_start = time.time()
    samples_done_this_session = 0

    for idx in range(len(data)):
        # Skip already-completed samples
        if idx in completed_set:
            continue

        sample = data[idx]
        question = sample.get("text") or sample.get("question", "")
        gt_answer = (sample.get("label") or sample.get("answer", "")).lower()
        image_file = sample.get("image", "")
        image_path = os.path.join(image_dir, image_file)

        if not os.path.exists(image_path):
            if samples_done_this_session == 0:
                logger.error(f"Image not found: {image_path}")
            errors += 1
            completed_indices.append(idx)
            completed_set.add(idx)
            continue

        img = Image.open(image_path).convert("RGB")
        prompt = question + " Answer with yes or no."

        if gt_answer == "yes":
            correct_tok, incorrect_tok = "Yes", "No"
        else:
            correct_tok, incorrect_tok = "No", "Yes"

        try:
            result = patcher.patch_sample(
                img, prompt,
                correct_token=correct_tok,
                incorrect_token=incorrect_tok,
                gt_answer=gt_answer,
            )
            result.question = question
            results.append(result)
            completed_indices.append(idx)
            completed_set.add(idx)
            samples_done_this_session += 1

            status = "HALLUC" if result.is_hallucination else "OK"
            remaining = len(data) - len(completed_set)
            elapsed = time.time() - t_start
            rate = samples_done_this_session / elapsed if elapsed > 0 else 0
            eta = remaining / rate if rate > 0 else 0

            if samples_done_this_session % 10 == 0 or samples_done_this_session == 1:
                logger.info(
                    f"[{len(completed_set)}/{len(data)}] {status} | "
                    f"ld={result.clean_logit_diff:+.2f} | "
                    f"te={result.total_effect:.3f} | "
                    f"{result.time_seconds:.1f}s | "
                    f"ETA {eta/60:.0f}min "
                    f"({len(results)} results, {errors} err)")

        except Exception as e:
            logger.error(f"Error on sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            errors += 1
            completed_indices.append(idx)
            completed_set.add(idx)
            continue

        # Checkpoint
        if ckpt_interval > 0 and samples_done_this_session % ckpt_interval == 0:
            save_checkpoint(output_dir, results, completed_indices, run_meta)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

    total_time = time.time() - t_start
    logger.info(f"Completed: {len(results)} samples in {total_time/60:.1f}min "
                f"({errors} errors)")

    # Aggregate
    batch = BatchPatchingResult(results=results)

    # Summary
    print("\n" + "=" * 70)
    print(f"ACTIVATION PATCHING RESULTS — {model_id} ({benchmark})")
    print("=" * 70)
    print(f"  Samples:       {len(results)} "
          f"({batch.n_correct} correct, {batch.n_hallucination} halluc)")
    print(f"  Halluc rate:   {batch.n_hallucination / max(len(results), 1):.1%}")
    print(f"  Granularity:   {granularity}")
    print(f"  Time:          {total_time/60:.1f} min "
          f"({total_time/max(len(results),1):.1f} s/sample)")

    # Extract circuit
    model_short = model_id.split("/")[-1]
    circuit = extract_circuit(batch, model_short)

    print(f"\n  Circuit: {circuit.n_components} significant components")
    for c in circuit.top_k(10):
        print(f"    {c.name:>10s}  d={c.differential:+.4f}  "
              f"(correct={c.effect_correct:.4f}  "
              f"halluc={c.effect_halluc:.4f})  "
              f"d={c.cohen_d:.2f}  p={c.p_value:.4f}")

    # Save
    save_circuit(circuit, os.path.join(output_dir, "circuit.json"))

    with open(os.path.join(output_dir, "batch_results.pkl"), "wb") as f:
        pickle.dump(batch, f)

    summary = {
        "model": model_id,
        "benchmark": benchmark,
        "split": split,
        "n_samples": len(results),
        "n_correct": batch.n_correct,
        "n_hallucination": batch.n_hallucination,
        "halluc_rate": batch.n_hallucination / max(len(results), 1),
        "granularity": granularity,
        "total_time_min": total_time / 60,
        "n_circuit_components": circuit.n_components,
        "top10_components": [
            {"name": c.name, "diff": c.differential,
             "correct": c.effect_correct, "halluc": c.effect_halluc,
             "cohen_d": c.cohen_d, "p": c.p_value}
            for c in circuit.top_k(10)
        ],
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if circuit.component_matrix_correct is not None:
        np.save(os.path.join(output_dir, "comp_correct.npy"),
                circuit.component_matrix_correct)
    if circuit.component_matrix_halluc is not None:
        np.save(os.path.join(output_dir, "comp_halluc.npy"),
                circuit.component_matrix_halluc)
    if circuit.head_matrix_correct is not None:
        np.save(os.path.join(output_dir, "head_correct.npy"),
                circuit.head_matrix_correct)
    if circuit.head_matrix_halluc is not None:
        np.save(os.path.join(output_dir, "head_halluc.npy"),
                circuit.head_matrix_halluc)

    # Remove checkpoint on successful completion
    ckpt_path = os.path.join(output_dir, CKPT_FILENAME)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        logger.info("Checkpoint removed (run complete)")

    logger.info(f"Results saved to {output_dir}")
    return batch, circuit


# ════════════════════════════════════════════════════════════════════
# CROSS-ARCHITECTURE COMPARISON
# ════════════════════════════════════════════════════════════════════

def run_comparison(result_dirs: List[str], output_dir: str):
    """Compare circuits extracted from different models."""
    from analysis.circuits import (
        HallucinationCircuit, compare_architectures, save_comparison)

    circuits = []
    for rd in result_dirs:
        cpath = os.path.join(rd, "circuit.json")
        if not os.path.exists(cpath):
            logger.error(f"No circuit.json found in {rd}")
            continue
        with open(cpath) as f:
            data = json.load(f)

        from analysis.circuits import CircuitComponent
        components = [
            CircuitComponent(
                layer=c["layer"], component_type=c["type"],
                head_idx=c.get("head_idx", -1),
                effect_correct=c["effect_correct"],
                effect_halluc=c["effect_halluc"],
                differential=c["differential"],
                p_value=c["p_value"], cohen_d=c["cohen_d"])
            for c in data["components"]
        ]

        comp_c = _try_load_npy(os.path.join(rd, "comp_correct.npy"))
        comp_h = _try_load_npy(os.path.join(rd, "comp_halluc.npy"))
        head_c = _try_load_npy(os.path.join(rd, "head_correct.npy"))
        head_h = _try_load_npy(os.path.join(rd, "head_halluc.npy"))

        circuit = HallucinationCircuit(
            model_name=data["model_name"],
            n_layers=data["n_layers"],
            n_heads=data["n_heads"],
            components=components,
            component_matrix_correct=comp_c,
            component_matrix_halluc=comp_h,
            head_matrix_correct=head_c,
            head_matrix_halluc=head_h,
        )
        circuits.append(circuit)
        logger.info(f"Loaded circuit: {circuit.model_name} "
                    f"({circuit.n_components} components)")

    if len(circuits) < 2:
        logger.error("Need at least 2 circuits for comparison")
        return

    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(circuits)):
        for j in range(i + 1, len(circuits)):
            comp = compare_architectures(circuits[i], circuits[j])
            name = f"{circuits[i].model_name}_vs_{circuits[j].model_name}"

            print(f"\n{'=' * 60}")
            print(f"COMPARISON: {comp.model_a} vs {comp.model_b}")
            print(f"{'=' * 60}")
            print(f"  Layer profile correlation: r={comp.layer_profile_corr:.3f} "
                  f"(p={comp.layer_profile_p:.4f})")
            print(f"  Component overlap (Jaccard): {comp.component_overlap:.3f}")
            print(f"  Interpretation: {comp.interpretation}")

            save_comparison(comp, os.path.join(output_dir, f"{name}.json"))


def _try_load_npy(path: str):
    if os.path.exists(path):
        return np.load(path)
    return None


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Activation patching for VLM hallucination circuits (v2)")

    # Mode
    parser.add_argument("--compare", nargs="+", default=None,
                        help="Paths to result dirs for cross-arch comparison")

    # Model
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile for faster forward passes")

    # Data
    parser.add_argument("--benchmark", type=str, default="pope",
                        choices=["pope", "mme", "amber"],
                        help="Which benchmark to use")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to benchmark data dir")
    parser.add_argument("--pope-path", type=str, default=None,
                        help="(Legacy) Alias for --data-path with POPE")
    parser.add_argument("--image-dir", type=str,
                        default="data/coco/val2014/")
    parser.add_argument("--split", type=str, default="adversarial",
                        choices=["random", "popular", "adversarial"])
    parser.add_argument("--n-samples", type=int, default=200)

    # Patching config
    parser.add_argument("--granularity", type=str, default="component",
                        choices=["component", "head"])
    parser.add_argument("--corrupt-mode", type=str, default="noise",
                        choices=["noise", "zero", "mean"])
    parser.add_argument("--noise-std", type=float, default=3.0)

    # Checkpoint
    parser.add_argument("--ckpt-interval", type=int, default=50,
                        help="Save checkpoint every N samples (0=disabled)")

    # Output
    parser.add_argument("--output", type=str, default="results/patching/")

    args = parser.parse_args()

    if args.compare:
        run_comparison(args.compare, args.output)
        return

    # Resolve data path
    data_path = args.data_path or args.pope_path
    if data_path is None:
        defaults = {"pope": "data/pope/", "mme": "data/mme/",
                    "amber": "data/amber/"}
        data_path = defaults.get(args.benchmark, "data/pope/")

    model, processor = load_model(args.model, compile_model=args.compile)
    data = load_data(args.benchmark, data_path,
                     split=args.split, n_samples=args.n_samples)

    run_patching_experiment(
        model, processor, args.model,
        data,
        image_dir=args.image_dir,
        output_dir=args.output,
        granularity=args.granularity,
        corrupt_mode=args.corrupt_mode,
        noise_std_mult=args.noise_std,
        benchmark=args.benchmark,
        split=args.split,
        ckpt_interval=args.ckpt_interval,
    )


if __name__ == "__main__":
    main()