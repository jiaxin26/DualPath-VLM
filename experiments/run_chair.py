"""
CHAIR Evaluation: Caption Hallucination Assessment with Image Relevance
========================================================================

Generates free-form captions for COCO val2014 images, then measures
hallucination using CHAIR metrics (Rohrbach et al., 2018):

  CHAIR_i = # hallucinated objects / # total mentioned objects
  CHAIR_s = # sentences with ≥1 hallucination / # total sentences

Supports:
  - All 3 models (Qwen3-VL, LLaVA-v1.6, InternVL3)
  - Baseline (no intervention) and targeted intervention conditions
  - Comparison output for paper tables

Usage:
  # Baseline evaluation (all models)
  PYTHONPATH=. python experiments/run_chair.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --coco-dir data/coco/ \
    --n-images 500 \
    --output results/chair_qwen/

  # With intervention
  PYTHONPATH=. python experiments/run_chair.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --coco-dir data/coco/ \
    --n-images 500 \
    --circuit results/qwen3vl_component/circuit.json \
    --scale 0.5 --min-d 0.7 \
    --output results/chair_qwen/

  # All 3 models baseline + intervention
  for m in "Qwen/Qwen3-VL-8B-Instruct" "llava-hf/llava-v1.6-mistral-7b-hf" "OpenGVLab/InternVL3-8B-hf"; do
    short=$(echo $m | tr '/' '_')
    PYTHONPATH=. python experiments/run_chair.py \
      --model "$m" --coco-dir data/coco/ --n-images 500 \
      --output "results/chair_${short}/"
  done
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("chair")


# ════════════════════════════════════════════════════════════════════
# COCO OBJECT SYNONYMS (80 categories + common synonyms)
# ════════════════════════════════════════════════════════════════════

# Maps COCO category names to sets of synonyms used in captions.
# Based on the standard CHAIR implementation.
COCO_SYNONYMS = {
    "person": {"person", "people", "man", "woman", "boy", "girl", "child",
               "kid", "baby", "lady", "guy", "gentleman", "player",
               "skier", "snowboarder", "surfer", "rider", "pedestrian",
               "passenger", "worker", "officer", "soldier", "adult",
               "teenager", "toddler", "infant", "couple", "crowd",
               "skateboarder", "cyclist"},
    "bicycle": {"bicycle", "bike", "cycling"},
    "car": {"car", "automobile", "vehicle", "sedan", "suv", "van",
            "taxi", "cab"},
    "motorcycle": {"motorcycle", "motorbike", "scooter"},
    "airplane": {"airplane", "plane", "aircraft", "jet"},
    "bus": {"bus"},
    "train": {"train", "locomotive"},
    "truck": {"truck", "lorry", "pickup"},
    "boat": {"boat", "ship", "vessel", "canoe", "kayak", "sailboat",
             "yacht", "ferry"},
    "traffic light": {"traffic light", "stoplight", "traffic signal"},
    "fire hydrant": {"fire hydrant", "hydrant"},
    "stop sign": {"stop sign"},
    "parking meter": {"parking meter"},
    "bench": {"bench", "park bench"},
    "bird": {"bird", "parrot", "pigeon", "duck", "seagull", "crow",
             "eagle", "hawk", "owl", "goose", "swan", "sparrow",
             "robin", "pelican", "flamingo"},
    "cat": {"cat", "kitten", "kitty", "feline"},
    "dog": {"dog", "puppy", "canine", "pup", "hound", "terrier",
            "poodle", "retriever", "labrador", "bulldog", "beagle",
            "shepherd", "collie", "dalmatian"},
    "horse": {"horse", "pony", "stallion", "mare", "foal", "colt",
              "equestrian"},
    "sheep": {"sheep", "lamb"},
    "cow": {"cow", "cattle", "bull", "calf", "ox", "steer", "heifer",
            "bovine"},
    "elephant": {"elephant"},
    "bear": {"bear", "teddy bear", "polar bear", "grizzly"},
    "zebra": {"zebra"},
    "giraffe": {"giraffe"},
    "backpack": {"backpack", "knapsack", "rucksack", "bag pack"},
    "umbrella": {"umbrella", "parasol"},
    "handbag": {"handbag", "purse", "clutch", "bag"},
    "tie": {"tie", "necktie", "bow tie"},
    "suitcase": {"suitcase", "luggage", "baggage", "briefcase",
                 "trunk"},
    "frisbee": {"frisbee", "disc", "flying disc"},
    "skis": {"skis", "ski"},
    "snowboard": {"snowboard"},
    "sports ball": {"ball", "sports ball", "baseball", "basketball",
                    "soccer ball", "football", "tennis ball",
                    "volleyball"},
    "kite": {"kite"},
    "baseball bat": {"baseball bat", "bat"},
    "baseball glove": {"baseball glove", "glove", "mitt"},
    "skateboard": {"skateboard"},
    "surfboard": {"surfboard"},
    "tennis racket": {"tennis racket", "racket", "racquet"},
    "bottle": {"bottle", "water bottle"},
    "wine glass": {"wine glass", "glass", "goblet", "champagne glass"},
    "cup": {"cup", "mug", "coffee cup", "teacup"},
    "fork": {"fork"},
    "knife": {"knife", "blade"},
    "spoon": {"spoon"},
    "bowl": {"bowl"},
    "banana": {"banana"},
    "apple": {"apple"},
    "sandwich": {"sandwich", "sub", "hoagie", "burger", "hamburger"},
    "orange": {"orange", "tangerine", "clementine"},
    "broccoli": {"broccoli"},
    "carrot": {"carrot"},
    "hot dog": {"hot dog", "hotdog", "sausage"},
    "pizza": {"pizza"},
    "donut": {"donut", "doughnut"},
    "cake": {"cake", "cupcake", "pastry"},
    "chair": {"chair", "seat", "stool", "armchair", "recliner"},
    "couch": {"couch", "sofa", "loveseat", "settee"},
    "potted plant": {"potted plant", "plant", "houseplant", "flower pot",
                     "flowerpot"},
    "bed": {"bed", "mattress", "bunk bed"},
    "dining table": {"dining table", "table", "desk", "counter",
                     "countertop"},
    "toilet": {"toilet", "restroom", "bathroom"},
    "tv": {"tv", "television", "monitor", "screen", "display"},
    "laptop": {"laptop", "computer", "notebook"},
    "mouse": {"mouse", "computer mouse"},
    "remote": {"remote", "remote control", "controller"},
    "keyboard": {"keyboard"},
    "cell phone": {"cell phone", "phone", "cellphone", "mobile phone",
                   "smartphone", "mobile", "iphone"},
    "microwave": {"microwave", "microwave oven"},
    "oven": {"oven", "stove"},
    "toaster": {"toaster"},
    "sink": {"sink", "basin"},
    "refrigerator": {"refrigerator", "fridge"},
    "book": {"book", "textbook", "novel", "magazine"},
    "clock": {"clock", "watch", "alarm clock", "wall clock"},
    "vase": {"vase"},
    "scissors": {"scissors", "shears"},
    "teddy bear": {"teddy bear", "stuffed animal", "plush",
                   "stuffed bear"},
    "hair drier": {"hair drier", "hair dryer", "blow dryer"},
    "toothbrush": {"toothbrush"},
}

# Build reverse map: synonym → COCO category
_SYNONYM_TO_CATEGORY = {}
for cat, syns in COCO_SYNONYMS.items():
    for syn in syns:
        _SYNONYM_TO_CATEGORY[syn.lower()] = cat


# ════════════════════════════════════════════════════════════════════
# COCO ANNOTATIONS
# ════════════════════════════════════════════════════════════════════

def load_coco_annotations(coco_dir: str) -> Dict[int, Set[str]]:
    """Load COCO annotations and return {image_id: set of category names}.

    Looks for annotations at:
      coco_dir/annotations/instances_val2014.json
    """
    ann_path = os.path.join(coco_dir, "annotations", "instances_val2014.json")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(
            f"COCO annotations not found at {ann_path}. "
            "Download from http://images.cocodataset.org/annotations/"
            "annotations_trainval2014.zip")

    logger.info(f"Loading COCO annotations from {ann_path}")
    with open(ann_path) as f:
        coco = json.load(f)

    # Build category id → name map
    cat_id_to_name = {}
    for cat in coco["categories"]:
        cat_id_to_name[cat["id"]] = cat["name"]

    # Build image_id → set of category names
    image_objects = defaultdict(set)
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        cat_name = cat_id_to_name[ann["category_id"]]
        image_objects[img_id].add(cat_name)

    logger.info(f"Loaded annotations for {len(image_objects)} images, "
                f"{len(cat_id_to_name)} categories")
    return dict(image_objects)


def get_image_id_from_filename(filename: str) -> int:
    """Extract COCO image ID from filename like 'COCO_val2014_000000012345.jpg'."""
    match = re.search(r'(\d{12})', filename)
    if match:
        return int(match.group(1))
    # Try just extracting any number
    match = re.search(r'(\d+)', os.path.basename(filename))
    if match:
        return int(match.group(1))
    return -1


# ════════════════════════════════════════════════════════════════════
# CAPTION GENERATION
# ════════════════════════════════════════════════════════════════════

def generate_caption(model, processor, model_id: str, image: Image.Image,
                     max_new_tokens: int = 100) -> str:
    """Generate a caption for an image."""
    prompt = "Describe this image in one or two sentences."

    model_lower = model_id.lower()
    device = next(model.parameters()).device

    if "qwen" in model_lower:
        from patching.qwen_patcher import _prepare_inputs
        inputs = _prepare_inputs(processor, image, prompt, device)
    elif "llava" in model_lower:
        from patching.llava_patcher import _prepare_inputs
        inputs = _prepare_inputs(processor, image, prompt, device)
    elif "internvl" in model_lower:
        from patching.internvl_patcher import _prepare_inputs
        inputs = _prepare_inputs(processor, image, prompt, device)
    else:
        inputs = processor(text=prompt, images=image,
                           return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None)

    input_len = inputs["input_ids"].shape[1]
    caption = processor.decode(
        output_ids[0][input_len:], skip_special_tokens=True).strip()

    return caption


# ════════════════════════════════════════════════════════════════════
# CHAIR EVALUATION
# ════════════════════════════════════════════════════════════════════

def extract_objects_from_caption(caption: str) -> List[str]:
    """Extract COCO object mentions from a caption string.

    Returns list of COCO category names found in the caption.
    Uses multi-word matching (longest match first).
    """
    caption_lower = caption.lower()
    # Remove punctuation except hyphens
    caption_clean = re.sub(r'[^\w\s-]', ' ', caption_lower)

    found = []

    # Sort synonyms by length (longest first) to match multi-word phrases
    all_synonyms = sorted(_SYNONYM_TO_CATEGORY.keys(),
                          key=len, reverse=True)

    # Track which positions have been matched
    matched_positions = set()

    for synonym in all_synonyms:
        # Word boundary matching
        pattern = r'\b' + re.escape(synonym) + r'\b'
        for match in re.finditer(pattern, caption_clean):
            start, end = match.start(), match.end()
            # Check no overlap with already matched
            if not any(p in matched_positions for p in range(start, end)):
                category = _SYNONYM_TO_CATEGORY[synonym]
                found.append(category)
                matched_positions.update(range(start, end))

    return found


def compute_chair(captions: List[dict],
                  image_objects: Dict[int, Set[str]]) -> Dict[str, float]:
    """Compute CHAIR metrics from generated captions.

    Args:
        captions: list of {"image_id": int, "caption": str}
        image_objects: {image_id: set of COCO category names}

    Returns:
        dict with CHAIR_i, CHAIR_s, and supporting statistics
    """
    total_objects = 0
    halluc_objects = 0
    total_sentences = 0
    halluc_sentences = 0
    n_images = 0
    per_image_chair = []

    for item in captions:
        img_id = item["image_id"]
        caption = item["caption"]
        gt_objects = image_objects.get(img_id, set())

        # Extract mentioned objects
        mentioned = extract_objects_from_caption(caption)
        mentioned_unique = list(set(mentioned))

        # Count hallucinated objects (mentioned but not in GT)
        img_halluc = 0
        for obj in mentioned_unique:
            if obj not in gt_objects:
                halluc_objects += 1
                img_halluc += 1
            total_objects += 1

        # Sentence-level: split caption into sentences
        sentences = re.split(r'[.!?]+', caption)
        sentences = [s.strip() for s in sentences if s.strip()]

        for sent in sentences:
            total_sentences += 1
            sent_objects = extract_objects_from_caption(sent)
            sent_halluc = any(obj not in gt_objects
                              for obj in set(sent_objects))
            if sent_halluc:
                halluc_sentences += 1

        # Per-image CHAIR
        if mentioned_unique:
            per_image_chair.append(img_halluc / len(mentioned_unique))
        else:
            per_image_chair.append(0.0)

        n_images += 1

    chair_i = halluc_objects / max(total_objects, 1)
    chair_s = halluc_sentences / max(total_sentences, 1)
    avg_objects = total_objects / max(n_images, 1)

    return {
        "CHAIR_i": chair_i,
        "CHAIR_s": chair_s,
        "total_objects_mentioned": total_objects,
        "halluc_objects": halluc_objects,
        "total_sentences": total_sentences,
        "halluc_sentences": halluc_sentences,
        "n_images": n_images,
        "avg_objects_per_image": avg_objects,
        "mean_per_image_chair": float(np.mean(per_image_chair)),
        "std_per_image_chair": float(np.std(per_image_chair)),
    }


# ════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ════════════════════════════════════════════════════════════════════

def select_images(coco_dir: str, image_objects: Dict[int, Set[str]],
                  n_images: int, seed: int = 42) -> List[dict]:
    """Select random COCO val images that have annotations."""
    image_dir = os.path.join(coco_dir, "val2014")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    # List available images
    available = []
    for fname in sorted(os.listdir(image_dir)):
        if not fname.endswith((".jpg", ".jpeg", ".png")):
            continue
        img_id = get_image_id_from_filename(fname)
        if img_id in image_objects:
            available.append({"filename": fname, "image_id": img_id,
                              "path": os.path.join(image_dir, fname)})

    logger.info(f"Found {len(available)} annotated images in {image_dir}")

    rng = random.Random(seed)
    selected = rng.sample(available, min(n_images, len(available)))
    logger.info(f"Selected {len(selected)} images for evaluation")

    return selected


def run_evaluation(model, processor, model_id: str,
                   images: List[dict], image_objects: Dict[int, Set[str]],
                   condition_name: str = "baseline",
                   max_new_tokens: int = 100) -> dict:
    """Generate captions and compute CHAIR."""
    logger.info(f"{'=' * 60}")
    logger.info(f"Generating captions: {condition_name}")
    logger.info(f"{'=' * 60}")

    captions = []
    t0 = time.time()

    for idx, img_info in enumerate(images):
        try:
            img = Image.open(img_info["path"]).convert("RGB")
            caption = generate_caption(
                model, processor, model_id, img,
                max_new_tokens=max_new_tokens)
            captions.append({
                "image_id": img_info["image_id"],
                "filename": img_info["filename"],
                "caption": caption,
            })

            if (idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                eta = (len(images) - idx - 1) / rate
                logger.info(f"  [{idx+1}/{len(images)}] "
                            f"{rate:.1f} img/s, ETA {eta/60:.1f}min")

        except Exception as e:
            logger.error(f"Error on image {idx} ({img_info['filename']}): {e}")
            continue

    elapsed = time.time() - t0
    logger.info(f"Generated {len(captions)} captions in {elapsed/60:.1f}min")

    # Compute CHAIR
    results = compute_chair(captions, image_objects)
    results["condition"] = condition_name
    results["model"] = model_id
    results["time_min"] = elapsed / 60

    print(f"\n  {condition_name}:")
    print(f"    CHAIR_i = {results['CHAIR_i']:.4f} "
          f"({results['halluc_objects']}/{results['total_objects_mentioned']})")
    print(f"    CHAIR_s = {results['CHAIR_s']:.4f} "
          f"({results['halluc_sentences']}/{results['total_sentences']})")
    print(f"    Avg objects/image = {results['avg_objects_per_image']:.1f}")

    return results, captions


def main():
    parser = argparse.ArgumentParser(
        description="CHAIR evaluation for VLM hallucination")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--coco-dir", type=str, default="data/coco/",
                        help="COCO directory (must contain val2014/ and "
                             "annotations/instances_val2014.json)")
    parser.add_argument("--n-images", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--output", type=str, default="results/chair/")

    # Intervention (optional)
    parser.add_argument("--circuit", type=str, default=None,
                        help="Path to circuit.json for intervention")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--min-d", type=float, default=0.4)

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Load model
    from experiments.run_patching import load_model
    model, processor = load_model(args.model)

    # Load COCO annotations
    image_objects = load_coco_annotations(args.coco_dir)

    # Select images
    images = select_images(args.coco_dir, image_objects,
                           args.n_images, args.seed)

    all_results = {}

    # ── Condition 1: Baseline ──────────────────────────────────────
    baseline_results, baseline_captions = run_evaluation(
        model, processor, args.model, images, image_objects,
        condition_name="baseline",
        max_new_tokens=args.max_new_tokens)
    all_results["baseline"] = baseline_results

    # Save captions
    cap_path = os.path.join(args.output, "captions_baseline.json")
    with open(cap_path, "w") as f:
        json.dump(baseline_captions, f, indent=2)

    # ── Condition 2: With intervention (if circuit provided) ───────
    if args.circuit and os.path.exists(args.circuit):
        from experiments.run_intervention import (
            ComponentIntervention, load_halluc_targets)

        targets = load_halluc_targets(
            args.circuit, pathway="halluc", min_abs_d=args.min_d)

        if targets:
            intervention = ComponentIntervention(
                model, args.model, targets, scale=args.scale)
            intervention.attach()

            intervened_results, intervened_captions = run_evaluation(
                model, processor, args.model, images, image_objects,
                condition_name=f"intervention_s{args.scale}",
                max_new_tokens=args.max_new_tokens)
            all_results["intervention"] = intervened_results

            intervention.detach()

            # Save
            cap_path = os.path.join(args.output, "captions_intervention.json")
            with open(cap_path, "w") as f:
                json.dump(intervened_captions, f, indent=2)

            # Delta
            d_i = intervened_results["CHAIR_i"] - baseline_results["CHAIR_i"]
            d_s = intervened_results["CHAIR_s"] - baseline_results["CHAIR_s"]
            print(f"\n  Intervention effect:")
            print(f"    ΔCHAIR_i = {d_i:+.4f}")
            print(f"    ΔCHAIR_s = {d_s:+.4f}")
            all_results["delta_CHAIR_i"] = d_i
            all_results["delta_CHAIR_s"] = d_s

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"CHAIR RESULTS — {args.model}")
    print(f"{'=' * 60}")
    for cond, res in all_results.items():
        if isinstance(res, dict) and "CHAIR_i" in res:
            print(f"  {res['condition']:30s} "
                  f"CHAIR_i={res['CHAIR_i']:.4f}  "
                  f"CHAIR_s={res['CHAIR_s']:.4f}")

    # ── Save ───────────────────────────────────────────────────────
    out_path = os.path.join(args.output, "chair_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to {out_path}")

    # LaTeX row
    tex_path = os.path.join(args.output, "chair_table_row.tex")
    model_short = args.model.split("/")[-1]
    with open(tex_path, "w") as f:
        f.write(f"% CHAIR results for {model_short}\n")
        b = all_results["baseline"]
        f.write(f"  {model_short} & Baseline "
                f"& {b['CHAIR_i']:.1%} & {b['CHAIR_s']:.1%} "
                f"& {b['avg_objects_per_image']:.1f} \\\\\n")
        if "intervention" in all_results:
            iv = all_results["intervention"]
            f.write(f"  {model_short} & Intervention "
                    f"& {iv['CHAIR_i']:.1%} & {iv['CHAIR_s']:.1%} "
                    f"& {iv['avg_objects_per_image']:.1f} \\\\\n")
    logger.info(f"LaTeX row saved to {tex_path}")


if __name__ == "__main__":
    main()
