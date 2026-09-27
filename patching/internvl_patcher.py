"""
InternVL3 Activation Patcher
==============================

Model-specific adapter for InternVL3 (ViT-MLP-LLM paradigm).
Uses the HF-native version (OpenGVLab/InternVL3-8B-hf) which
follows standard transformers conventions.

Architecture (InternVL3-8B-hf):
  InternVLForConditionalGeneration
    ├── vision_model  (InternViT-300M)
    ├── mlp1          (MLP projector: ViT hidden → LLM hidden)
    └── language_model (Qwen2.5-based)
        ├── model
        │   ├── embed_tokens
        │   ├── layers  (ModuleList of decoder layers)
        │   └── norm
        └── lm_head

Key differences from Qwen3-VL:
  - Separate ViT + MLP projector (not embed-concat)
  - Visual tokens marked by <IMG_CONTEXT> or pixel_values presence
  - Qwen2.5 backbone (not Qwen3), but layer structure identical

HF model ID: OpenGVLab/InternVL3-8B-hf
  (NOT OpenGVLab/InternVL3-8B — that one needs custom code)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .core import ActivationPatcher, PatchingConfig

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
# MODEL DISCOVERY
# ════════════════════════════════════════════════════════════════════

def _get_decoder_layers(model) -> nn.ModuleList:
    """Locate decoder layers in InternVL3."""
    for path in [
        # HF-native InternVL3: model.model.language_model.layers
        # (from top-level ForConditionalGeneration, "model" = InternVLModel,
        #  "language_model" = Qwen2Model which contains "layers")
        ("model", "language_model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("model", "language_model", "model", "layers"),
        ("model", "layers"),
    ]:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            if isinstance(obj, nn.ModuleList) and len(obj) > 0:
                logger.info(f"InternVL3: found decoder layers via {'.'.join(path)} "
                            f"({len(obj)} layers)")
                return obj
        except (AttributeError, IndexError, TypeError):
            continue
    raise RuntimeError("Could not locate decoder layers in InternVL3 model. "
                       "Make sure you're using OpenGVLab/InternVL3-*-hf")


def _get_inner_model(model):
    """Get the inner language model (for ViT-skip forward)."""
    for path in [
        # HF-native InternVL3: model.model.language_model is Qwen2Model
        ("model", "language_model"),
        ("language_model", "model"),
        ("language_model",),
        ("model", "language_model", "model"),
    ]:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            if hasattr(obj, "layers") and hasattr(obj, "norm"):
                return obj
        except (AttributeError, TypeError):
            continue
    logger.warning("Could not find inner model for InternVL3 — ViT-skip disabled")
    return None


def _get_lm_head(model):
    """Get the lm_head for logit computation."""
    for path in [
        # HF-native InternVL3: lm_head at top level
        ("lm_head",),
        ("language_model", "lm_head"),
        ("model", "lm_head"),
    ]:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            if hasattr(obj, "weight"):
                return obj
        except (AttributeError, TypeError):
            continue
    raise RuntimeError("Could not find lm_head in InternVL3 model")


def _get_head_config(model) -> Tuple[int, int, int]:
    """Returns (n_heads, n_kv_heads, head_dim)."""
    config = model.config
    # Try text_config first (HF-native), then top-level
    for cfg in [getattr(config, "text_config", None),
                getattr(config, "llm_config", None),
                config]:
        if cfg is None:
            continue
        n_heads = getattr(cfg, "num_attention_heads", None)
        n_kv = getattr(cfg, "num_key_value_heads", n_heads)
        hidden = getattr(cfg, "hidden_size", None)
        head_dim = getattr(cfg, "head_dim", None)
        if n_heads and hidden:
            if head_dim is None:
                head_dim = hidden // n_heads
            return n_heads, n_kv or n_heads, head_dim
    # Fallback: infer from weights
    try:
        layers = _get_decoder_layers(model)
        q_w = layers[0].self_attn.q_proj.weight
        k_w = layers[0].self_attn.k_proj.weight
        head_dim = 128
        return q_w.shape[0] // head_dim, k_w.shape[0] // head_dim, head_dim
    except Exception:
        return 32, 8, 128


# ════════════════════════════════════════════════════════════════════
# INPUT PREPARATION
# ════════════════════════════════════════════════════════════════════

def _prepare_inputs(processor, image, prompt: str, device):
    """Prepare inputs for InternVL3 using HF chat template."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs.to(device)


def _find_visual_positions(processor, input_ids: torch.Tensor) -> List[int]:
    """Find positions of visual tokens in InternVL3 input.

    InternVL3 uses special tokens to mark image regions.
    Common markers: <IMG_CONTEXT>, <img>, image_pad tokens.
    """
    tokenizer = processor.tokenizer
    unk_id = getattr(tokenizer, "unk_token_id", None)
    ids = input_ids[0].cpu().tolist()

    def _tid(name):
        t = tokenizer.convert_tokens_to_ids(name)
        return None if (t is None or t == unk_id) else t

    # Try InternVL-specific tokens
    for name in ["<IMG_CONTEXT>", "<img>", "<image_pad>",
                  "<|image_pad|>", "<|placeholder|>"]:
        pid = _tid(name)
        if pid is not None and pid in ids:
            positions = [i for i, t in enumerate(ids) if t == pid]
            if positions:
                logger.debug(f"InternVL3: found {len(positions)} visual tokens "
                             f"via {name} (id={pid})")
                return positions

    # Try start/end marker approach
    for start_name, end_name in [
        ("<img>", "</img>"),
        ("<image>", "</image>"),
        ("<|vision_start|>", "<|vision_end|>"),
    ]:
        sid, eid = _tid(start_name), _tid(end_name)
        if sid is not None and sid in ids:
            pos, inside = [], False
            for i, t in enumerate(ids):
                if t == sid:
                    inside = True
                    continue
                if eid and t == eid:
                    inside = False
                    continue
                if inside:
                    pos.append(i)
            if pos:
                logger.debug(f"InternVL3: found {len(pos)} visual tokens "
                             f"via {start_name}..{end_name}")
                return pos

    # Fallback: look for the most common non-text token in a contiguous block
    # (InternVL typically inserts a block of repeated pad tokens for the image)
    from collections import Counter
    text_tokens = set(tokenizer.encode("yes no the a is", add_special_tokens=False))
    special = set()
    for sid in [tokenizer.bos_token_id, tokenizer.eos_token_id,
                tokenizer.pad_token_id, unk_id]:
        if sid is not None:
            special.add(sid)

    # Count tokens that aren't common text or special tokens
    candidate_counts = Counter()
    for i, t in enumerate(ids):
        if t not in text_tokens and t not in special:
            candidate_counts[t] += 1

    if candidate_counts:
        most_common_id, count = candidate_counts.most_common(1)[0]
        if count >= 10:  # likely image tokens
            positions = [i for i, t in enumerate(ids) if t == most_common_id]
            logger.info(f"InternVL3: found {len(positions)} visual tokens "
                        f"via frequency heuristic (token_id={most_common_id})")
            return positions

    logger.warning("InternVL3: could not detect visual positions")
    return []


# ════════════════════════════════════════════════════════════════════
# PATCHER
# ════════════════════════════════════════════════════════════════════

class InternVLActivationPatcher(ActivationPatcher):
    """Activation patcher for InternVL3 (HF-native version)."""

    def __init__(self, model, processor, config: PatchingConfig = None):
        self._model = model
        self._processor = processor
        super().__init__(model, processor, config)

    def _prepare_inputs(self, image, prompt: str) -> dict:
        return _prepare_inputs(self._processor, image, prompt,
                               self._model.device)

    def _find_visual_positions(self, input_ids: torch.Tensor) -> List[int]:
        return _find_visual_positions(self._processor, input_ids)

    def _get_decoder_layers(self) -> nn.ModuleList:
        return _get_decoder_layers(self._model)

    def _get_inner_model(self):
        return _get_inner_model(self._model)

    def _get_lm_head(self):
        return _get_lm_head(self._model)

    def _get_head_config(self) -> Tuple[int, int, int]:
        return _get_head_config(self._model)