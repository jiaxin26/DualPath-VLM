"""
Multi-Benchmark Data Loading for Hallucination Circuit Experiments
===================================================================

Unified interface for loading object-existence samples from:
  - POPE (adversarial / popular / random splits)
  - MME existence subset
  - AMBER discriminative subset

All loaders return List[dict] with a common schema:
  {
    "image": str,          # relative path to image file
    "question": str,       # the question text
    "answer": str,         # ground truth: "yes" or "no"
    "benchmark": str,      # source benchmark name
    "split": str,          # split name (if applicable)
    "meta": dict,          # benchmark-specific metadata
  }

Setup instructions
------------------

POPE:
  git clone https://github.com/AoiDragon/POPE.git data/pope_repo
  cp data/pope_repo/output/coco/*.json data/pope/

MME:
  # Download from https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models
  # Place images in data/mme/images/
  # Place existence.txt in data/mme/

AMBER:
  # Download from https://github.com/junyangwang0410/AMBER
  # Place in data/amber/
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
# COMMON SCHEMA
# ════════════════════════════════════════════════════════════════════

def make_sample(image: str, question: str, answer: str,
                benchmark: str, split: str = "",
                meta: dict = None) -> dict:
    """Create a sample in the unified schema."""
    return {
        "image": image,
        "question": question,
        "answer": answer.strip().lower(),
        "benchmark": benchmark,
        "split": split,
        "meta": meta or {},
    }


# ════════════════════════════════════════════════════════════════════
# POPE
# ════════════════════════════════════════════════════════════════════

def load_pope(pope_path: str, split: str = "adversarial",
              n_samples: Optional[int] = None) -> List[dict]:
    """Load POPE benchmark samples.

    Tries multiple filename patterns for compatibility.
    """
    candidates = [
        f"coco_pope_{split}.json",
        f"pope_{split}.jsonl",
        f"coco_{split}.jsonl",
        f"{split}.jsonl",
    ]
    pope_file = None
    for name in candidates:
        path = os.path.join(pope_path, name)
        if os.path.exists(path):
            pope_file = path
            break

    if pope_file is None:
        raise FileNotFoundError(
            f"POPE file not found in {pope_path} for split '{split}'. "
            f"Tried: {candidates}. "
            "Download from https://github.com/AoiDragon/POPE")

    data = []
    with open(pope_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            question = raw.get("text") or raw.get("question", "")
            answer = raw.get("label") or raw.get("answer", "")
            image = raw.get("image", "")

            data.append(make_sample(
                image=image,
                question=question,
                answer=answer,
                benchmark="pope",
                split=split,
                meta={"raw": raw},
            ))
            if n_samples and len(data) >= n_samples:
                break

    logger.info(f"POPE ({split}): loaded {len(data)} samples from {pope_file}")
    return data


# ════════════════════════════════════════════════════════════════════
# MME EXISTENCE SUBSET
# ════════════════════════════════════════════════════════════════════

def load_mme_existence(mme_path: str,
                       n_samples: Optional[int] = None) -> List[dict]:
    """Load MME existence subset.

    MME format (existence.txt):
      Each line: <image_filename> <question> <answer>
      where answer is "Yes" or "No"

    Also supports the JSON-lines variant some repos use.
    """
    # Try txt format first
    txt_path = os.path.join(mme_path, "existence.txt")
    jsonl_path = os.path.join(mme_path, "existence.jsonl")

    data = []

    if os.path.exists(txt_path):
        with open(txt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: "image.jpg\tIs there a xxx in the image?\tYes"
                # or space-separated with image first
                parts = line.split("\t")
                if len(parts) >= 3:
                    image, question, answer = parts[0], parts[1], parts[2]
                elif len(parts) == 2:
                    # Some versions: "image.jpg question? answer"
                    image = parts[0]
                    rest = parts[1]
                    # answer is last word
                    tokens = rest.rsplit(" ", 1)
                    question = tokens[0] if len(tokens) > 1 else rest
                    answer = tokens[-1]
                else:
                    continue

                data.append(make_sample(
                    image=image.strip(),
                    question=question.strip(),
                    answer=answer.strip(),
                    benchmark="mme",
                    split="existence",
                ))
                if n_samples and len(data) >= n_samples:
                    break

    elif os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                data.append(make_sample(
                    image=raw.get("image", ""),
                    question=raw.get("question", raw.get("text", "")),
                    answer=raw.get("answer", raw.get("label", "")),
                    benchmark="mme",
                    split="existence",
                    meta=raw,
                ))
                if n_samples and len(data) >= n_samples:
                    break
    else:
        raise FileNotFoundError(
            f"MME existence file not found at {txt_path} or {jsonl_path}. "
            "Download from https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models")

    logger.info(f"MME existence: loaded {len(data)} samples")
    return data


# ════════════════════════════════════════════════════════════════════
# AMBER DISCRIMINATIVE SUBSET
# ════════════════════════════════════════════════════════════════════

def _infer_amber_type(question: str) -> str:
    """Infer AMBER hallucination type from question text.

    AMBER discriminative questions follow clear patterns:
      - existence: "Is there a/one/an X in this image?"
      - attribute: "Is the X [adj]?" / "Does the X [verb]?"
      - relation:  "Is there direct contact between X and Y?"
                   "Is X on/near/beside/above Y?"
    """
    q = question.lower().strip()

    # Relation patterns (check first — more specific)
    relation_keywords = [
        "between", "direct contact",
        "next to", "beside", "near",
        "on top of", "above", "below", "under",
        "in front of", "behind",
        "to the left", "to the right",
        "inside", "outside",
    ]
    for kw in relation_keywords:
        if kw in q:
            return "relation"

    # Existence patterns
    existence_patterns = [
        "is there a ", "is there an ", "is there one ",
        "are there ", "is there any ",
    ]
    for pat in existence_patterns:
        if q.startswith(pat):
            return "existence"

    # Attribute patterns (what's left: "is the X adj?", "does the X verb?")
    if q.startswith(("is the ", "are the ", "does the ", "do the ",
                     "does a ", "does this ")):
        return "attribute"

    # Fallback: if the question asks about a property, it's attribute;
    # if it asks about presence, it's existence
    if "is there" in q:
        return "existence"

    return "attribute"  # conservative fallback


def load_amber(amber_path: str,
               n_samples: Optional[int] = None,
               halluc_types: Optional[List[str]] = None) -> List[dict]:
    """Load AMBER discriminative subset.

    AMBER provides JSON with fields:
      id, image, query, answer (1=yes, 0=no), type
    The 'type' field indicates hallucination category:
      - "existence": object existence (yes/no)
      - "attribute": attribute correctness (color, shape, etc.)
      - "relation":  spatial/semantic relation between objects

    If 'type' is absent or None, it is inferred from the question text.

    Args:
        amber_path: path to AMBER data directory
        n_samples: max samples to load (None = all)
        halluc_types: which hallucination types to include.
            None or [] = all types. Examples:
            ["existence"] = object existence only (original behavior)
            ["attribute", "relation"] = fine-grained only
            ["existence", "attribute", "relation"] = all types
    """
    # Try multiple file patterns
    candidates = [
        "data/query/query_discriminative.json",
        "query_discriminative.json",
        "discriminative.json",
        "amber_discriminative.jsonl",
    ]
    amber_file = None
    for name in candidates:
        path = os.path.join(amber_path, name)
        if os.path.exists(path):
            amber_file = path
            break

    if amber_file is None:
        raise FileNotFoundError(
            f"AMBER file not found in {amber_path}. "
            f"Tried: {candidates}. "
            "Download from https://github.com/junyangwang0410/AMBER")

    with open(amber_file) as f:
        if amber_file.endswith(".jsonl"):
            raw_data = [json.loads(line) for line in f if line.strip()]
        else:
            raw_data = json.load(f)

    # Normalize type filter
    if halluc_types:
        # Expand aliases
        type_set = set()
        for t in halluc_types:
            t = t.lower().strip()
            if t in ("existence", "object"):
                type_set.add("existence")
                type_set.add("object")
            else:
                type_set.add(t)
    else:
        type_set = None  # accept all types

    data = []
    type_counts = {}
    n_inferred = 0
    for item in raw_data:
        # AMBER uses 1/0 for yes/no
        answer_raw = item.get("answer", "")
        if isinstance(answer_raw, int):
            answer = "yes" if answer_raw == 1 else "no"
        else:
            answer = str(answer_raw).strip().lower()
            if answer not in ("yes", "no"):
                answer = "yes" if answer in ("1", "true") else "no"

        question = item.get("query", item.get("question", ""))
        image = item.get("image", "")

        # Get type — infer from question text if missing or None
        item_type = item.get("type")
        if item_type is None or item_type == "":
            item_type = _infer_amber_type(question)
            n_inferred += 1

        # Type filtering
        if type_set is not None and item_type not in type_set:
            continue

        # Build split name that reflects the type filter
        if halluc_types and len(halluc_types) == 1:
            split_name = f"discriminative_{halluc_types[0]}"
        else:
            split_name = "discriminative_all"

        data.append(make_sample(
            image=image,
            question=question,
            answer=answer,
            benchmark="amber",
            split=split_name,
            meta={"id": item.get("id"), "type": item_type},
        ))

    # Shuffle before truncating to avoid ordering bias
    # (AMBER's raw data groups all "no" answers first)
    rng = random.Random(42)
    rng.shuffle(data)

    if n_samples and len(data) > n_samples:
        data = data[:n_samples]

    # Count types after final selection
    for d in data:
        t = d["meta"]["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    # Also log answer balance
    ans_counts = {}
    for d in data:
        a = d["answer"]
        ans_counts[a] = ans_counts.get(a, 0) + 1

    type_str = ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items()))
    ans_str = ", ".join(f"{k}={v}" for k, v in sorted(ans_counts.items()))
    logger.info(f"AMBER discriminative: loaded {len(data)} samples "
                f"(types: {type_str}) (answers: {ans_str})")
    if n_inferred > 0:
        logger.info(f"  ({n_inferred} types inferred from question text)")
    return data


# ════════════════════════════════════════════════════════════════════
# MMHAL-BENCH
# ════════════════════════════════════════════════════════════════════

def load_mmhal(mmhal_path: str,
               n_samples: Optional[int] = None) -> List[dict]:
    """Load MMHal-Bench evaluation data.

    MMHal-Bench (Sun et al., 2023) contains 96 image-question pairs
    covering 8 question categories × 12 object topics, designed to
    test diverse hallucination types beyond simple object existence.

    Data format (mmhal-bench_with_image.jsonl or response_template.json):
      {
        "image": <base64 or path>,
        "question": str,
        "gt_answer": str,
        "image_content": [list of objects/attributes in image],
        "category": str (e.g., "attribute", "adversarial", ...),
      }

    For yes/no evaluation, we convert open-ended questions to binary
    format by checking if the model's answer aligns with ground truth.
    For open-ended evaluation (GPT-4 scoring), use run_mmhal_openended.py.

    Setup:
      # Download from HuggingFace
      wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/mmhal-bench_with_image.jsonl
      # or from RLHF-V repo
      # Place in data/mmhal/
    """
    candidates = [
        "mmhal-bench_with_image.jsonl",
        "response_template.json",
        "mmhal_bench.json",
        "mmhal.jsonl",
    ]
    mmhal_file = None
    for name in candidates:
        path = os.path.join(mmhal_path, name)
        if os.path.exists(path):
            mmhal_file = path
            break

    if mmhal_file is None:
        raise FileNotFoundError(
            f"MMHal-Bench file not found in {mmhal_path}. "
            f"Tried: {candidates}. "
            "Download: wget https://huggingface.co/OpenGVLab/InternVL/"
            "resolve/main/mmhal-bench_with_image.jsonl")

    raw_data = []
    with open(mmhal_file) as f:
        if mmhal_file.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    raw_data.append(json.loads(line))
        else:
            raw_data = json.load(f)

    data = []
    for idx, item in enumerate(raw_data):
        question = item.get("question", item.get("text", ""))
        gt_answer = item.get("gt_answer", item.get("answer", ""))
        category = item.get("category", "unknown")
        image_content = item.get("image_content", [])

        # Handle image: could be a path or base64
        image = item.get("image_path", item.get("image", ""))
        # If image is base64, store the index as filename; the runner
        # will need to decode it
        if isinstance(image, str) and len(image) > 500:
            # Likely base64 — save reference
            image = f"mmhal_{idx}.jpg"  # placeholder, actual b64 in meta

        data.append(make_sample(
            image=image,
            question=question,
            answer=gt_answer,
            benchmark="mmhal",
            split=category,
            meta={
                "id": idx,
                "category": category,
                "image_content": image_content,
                "gt_answer": gt_answer,
                "raw": item,  # keep full item for GPT eval
            },
        ))
        if n_samples and len(data) >= n_samples:
            break

    # Log category distribution
    cats = {}
    for d in data:
        c = d["meta"]["category"]
        cats[c] = cats.get(c, 0) + 1
    cat_str = ", ".join(f"{k}={v}" for k, v in sorted(cats.items()))
    logger.info(f"MMHal-Bench: loaded {len(data)} samples ({cat_str})")
    return data


# ════════════════════════════════════════════════════════════════════
# UNIFIED LOADER
# ════════════════════════════════════════════════════════════════════

BENCHMARK_LOADERS = {
    "pope": load_pope,
    "mme": load_mme_existence,
    "amber": load_amber,
    "mmhal": load_mmhal,
}

# Default image directories per benchmark
DEFAULT_IMAGE_DIRS = {
    "pope": "data/coco/val2014/",
    "mme": "data/mme/images/",
    "amber": "data/amber/image/",
    "mmhal": "data/mmhal/images/",
}


def load_benchmark(name: str, data_path: str,
                   n_samples: Optional[int] = None,
                   split: str = "adversarial",
                   halluc_types: Optional[List[str]] = None,
                   ) -> List[dict]:
    """Load samples from any supported benchmark.

    Args:
        name: benchmark name ("pope", "mme", "amber", "mmhal")
        data_path: path to benchmark data directory
        n_samples: max samples to load (None = all)
        split: split name (only used for POPE)
        halluc_types: hallucination types to include (only used for AMBER).
            None = all types. ["existence"] = object only.
            ["attribute", "relation"] = fine-grained only.

    Returns:
        List of sample dicts in unified schema
    """
    name = name.lower()
    if name not in BENCHMARK_LOADERS:
        raise ValueError(
            f"Unknown benchmark: {name}. "
            f"Supported: {list(BENCHMARK_LOADERS.keys())}")

    if name == "pope":
        return load_pope(data_path, split=split, n_samples=n_samples)
    elif name == "mme":
        return load_mme_existence(data_path, n_samples=n_samples)
    elif name == "amber":
        return load_amber(data_path, n_samples=n_samples,
                          halluc_types=halluc_types)
    elif name == "mmhal":
        return load_mmhal(data_path, n_samples=n_samples)
    else:
        return BENCHMARK_LOADERS[name](data_path, n_samples=n_samples)


def resolve_image_path(sample: dict, image_dirs: Dict[str, str]) -> str:
    """Get full image path for a sample, using benchmark-specific dirs."""
    bench = sample["benchmark"]
    base_dir = image_dirs.get(bench, DEFAULT_IMAGE_DIRS.get(bench, ""))
    return os.path.join(base_dir, sample["image"])