"""
Qwen3-VL / Qwen2.5-VL Activation Patcher
==========================================

Model-specific adapter for Qwen-family VLMs. Handles ViT-skip,
inner-model capture, visual position detection, and
_deepstack_process bypass.

Architecture (Qwen3-VL):
  Qwen3VLForConditionalGeneration
    ├── visual  (ViT — skipped during patching)
    ├── model   (Qwen3VLTextModel: embed + layers + norm)
    └── lm_head
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
    for path in [
        ("model", "language_model", "layers"),
        ("model", "model", "layers"),
        ("model", "layers"),
        ("language_model", "model", "layers"),
    ]:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            if isinstance(obj, nn.ModuleList) and len(obj) > 0:
                return obj
        except (AttributeError, IndexError, TypeError):
            continue
    raise RuntimeError("Could not locate decoder layers in Qwen model.")


def _get_inner_model(model):
    for path in [
        ("model", "language_model"),
        ("model",),
        ("model", "model"),
        ("language_model", "model"),
        ("language_model",),
    ]:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            if hasattr(obj, "layers") and hasattr(obj, "norm"):
                return obj
        except (AttributeError, TypeError):
            continue
    logger.warning("Could not find inner model for Qwen — ViT-skip disabled")
    return None


def _get_lm_head(model):
    for attr in ["lm_head", "output"]:
        if hasattr(model, attr):
            return getattr(model, attr)
    raise RuntimeError("Could not find lm_head")


def _get_head_config(model) -> Tuple[int, int, int]:
    config = model.config
    for cfg in [getattr(config, "text_config", None), config]:
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
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    try:
        from qwen_vl_utils import process_vision_info
        try:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                add_vision_id=True)
        except TypeError:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt")
    except ImportError:
        inputs = processor(text=prompt, images=image, return_tensors="pt")
    return inputs.to(device)


def _find_visual_positions(processor, input_ids: torch.Tensor) -> List[int]:
    tokenizer = processor.tokenizer
    unk_id = getattr(tokenizer, "unk_token_id", None)
    ids = input_ids[0].cpu().tolist()

    def _tid(name):
        t = tokenizer.convert_tokens_to_ids(name)
        return None if (t is None or t == unk_id) else t

    # Try direct pad token
    for name in ["<|image_pad|>", "<|placeholder|>", "<|vision_pad|>"]:
        pid = _tid(name)
        if pid is not None and pid in ids:
            return [i for i, t in enumerate(ids) if t == pid]

    # Try start/end markers
    for sn, en in [("<|vision_start|>", "<|vision_end|>"),
                    ("<|image_start|>", "<|image_end|>")]:
        sid, eid = _tid(sn), _tid(en)
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
                return pos

    logger.warning("Could not detect visual positions in Qwen input.")
    return []


# ════════════════════════════════════════════════════════════════════
# PATCHER
# ════════════════════════════════════════════════════════════════════

class QwenActivationPatcher(ActivationPatcher):
    """Activation patcher for Qwen3-VL / Qwen2.5-VL."""

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


class Qwen25VLPatcher(QwenActivationPatcher):
    """Alias — Qwen2.5-VL uses the same architecture."""
    pass
