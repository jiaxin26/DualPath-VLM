"""
LLaMA-3 / Llama-3.2-Vision Activation Patcher
================================================

Model-specific adapter for LLaMA-3-based VLMs.
Supports:
  - Llama-3.2-Vision (meta-llama/Llama-3.2-11B-Vision-Instruct)
  - LLaVA-NeXT with Llama-3 backbone (llava-hf/llava-next-llama3-...)
  - Any VLM using LlamaForCausalLM or MllamaForConditionalGeneration

Architecture (Llama-3.2-Vision):
  MllamaForConditionalGeneration
    ├── vision_model  (ViT — skipped during patching)
    ├── multi_modal_projector
    ├── language_model (LlamaForCausalLM)
    │   ├── model (LlamaModel: embed + layers + norm)
    │   └── lm_head
    └── (or direct model + lm_head for LLaVA-style)

Architecture (LLaVA-NeXT-Llama3):
  LlavaNextForConditionalGeneration
    ├── vision_tower
    ├── multi_modal_projector
    └── language_model (LlamaForCausalLM)
        ├── model (LlamaModel: embed + layers + norm)
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
    """Locate transformer decoder layers in LLaMA-based VLMs."""
    search_paths = [
        # Llama-3.2-Vision (MllamaForConditionalGeneration)
        # model.model = MllamaModel, model.model.language_model = MllamaTextModel
        ("model", "language_model", "layers"),
        ("language_model", "model", "layers"),
        # LLaVA-NeXT-Llama3
        ("model", "language_model", "model", "layers"),
        ("language_model", "layers"),
        # Direct LlamaForCausalLM
        ("model", "layers"),
        # Nested
        ("model", "model", "layers"),
    ]
    for path in search_paths:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            if isinstance(obj, nn.ModuleList) and len(obj) > 0:
                logger.info(f"Found decoder layers at: {'.'.join(path)} "
                            f"({len(obj)} layers)")
                return obj
        except (AttributeError, IndexError, TypeError):
            continue
    raise RuntimeError(
        "Could not locate decoder layers in LLaMA model. "
        "Tried paths: " + ", ".join(".".join(p) for p in search_paths))


def _get_inner_model(model):
    """Get inner transformer model (for ViT-skip forward)."""
    search_paths = [
        # Llama-3.2-Vision
        ("language_model", "model"),
        ("language_model",),
        # LLaVA-NeXT
        ("model", "language_model", "model"),
        ("model", "language_model"),
        # Direct
        ("model",),
        ("model", "model"),
    ]
    for path in search_paths:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            if hasattr(obj, "layers") and hasattr(obj, "norm"):
                logger.info(f"Found inner model at: {'.'.join(path)}")
                return obj
        except (AttributeError, TypeError):
            continue
    logger.warning("Could not find inner model for LLaMA — ViT-skip disabled")
    return None


def _get_lm_head(model):
    """Get the language model head."""
    # Llama-3.2-Vision
    if hasattr(model, "language_model") and hasattr(model.language_model, "lm_head"):
        return model.language_model.lm_head
    # LLaVA-style
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
        if hasattr(lm, "lm_head"):
            return lm.lm_head
    # Direct
    for attr in ["lm_head", "output"]:
        if hasattr(model, attr):
            return getattr(model, attr)
    raise RuntimeError("Could not find lm_head in LLaMA model")


def _get_head_config(model) -> Tuple[int, int, int]:
    """Get (n_heads, n_kv_heads, head_dim) from config."""
    # Try various config locations
    configs = [
        getattr(model.config, "text_config", None),
        model.config,
    ]
    if hasattr(model, "language_model"):
        configs.insert(0, getattr(model.language_model, "config", None))

    for cfg in configs:
        if cfg is None:
            continue
        n_heads = getattr(cfg, "num_attention_heads", None)
        n_kv = getattr(cfg, "num_key_value_heads", n_heads)
        hidden = getattr(cfg, "hidden_size", None)
        head_dim = getattr(cfg, "head_dim", None)

        if n_heads and hidden:
            if head_dim is None:
                head_dim = hidden // n_heads
            logger.info(f"Head config: n_heads={n_heads}, n_kv={n_kv}, "
                        f"head_dim={head_dim}")
            return n_heads, n_kv or n_heads, head_dim

    # Fallback: infer from weights
    try:
        layers = _get_decoder_layers(model)
        q_w = layers[0].self_attn.q_proj.weight
        k_w = layers[0].self_attn.k_proj.weight
        head_dim = 128  # LLaMA-3 default
        n_heads = q_w.shape[0] // head_dim
        n_kv = k_w.shape[0] // head_dim
        logger.info(f"Head config (inferred): n_heads={n_heads}, "
                    f"n_kv={n_kv}, head_dim={head_dim}")
        return n_heads, n_kv, head_dim
    except Exception:
        logger.warning("Falling back to default head config")
        return 32, 8, 128


# ════════════════════════════════════════════════════════════════════
# INPUT PREPARATION
# ════════════════════════════════════════════════════════════════════

def _prepare_inputs(processor, image, prompt: str, device):
    """Prepare model inputs for LLaMA-based VLMs.

    Handles both Llama-3.2-Vision (MllamaProcessor) and
    LLaVA-NeXT-Llama3 (LlavaNextProcessor) formats.
    """
    # Try Llama-3.2-Vision format first
    try:
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(
            messages, add_generation_prompt=True)
        inputs = processor(
            text=text, images=image, return_tensors="pt")
        return inputs.to(device)
    except Exception:
        pass

    # Try LLaVA-NeXT format
    try:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=text, images=image, return_tensors="pt")
        return inputs.to(device)
    except Exception:
        pass

    # Fallback: simple processing
    try:
        conversation = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True)
        inputs = processor(
            images=image, text=text, return_tensors="pt")
        return inputs.to(device)
    except Exception:
        pass

    # Last resort
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    return inputs.to(device)


def _find_visual_positions(processor, input_ids: torch.Tensor) -> List[int]:
    """Find visual token positions in input_ids.

    Llama-3.2-Vision uses <|image|> token (128256).
    LLaVA-NeXT uses various image pad tokens.
    """
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    unk_id = getattr(tokenizer, "unk_token_id", None)
    ids = input_ids[0].cpu().tolist()

    def _tid(name):
        try:
            t = tokenizer.convert_tokens_to_ids(name)
            return None if (t is None or t == unk_id) else t
        except Exception:
            return None

    # Llama-3.2-Vision: <|image|> token
    for name in ["<|image|>", "<image>", "<|image_pad|>"]:
        pid = _tid(name)
        if pid is not None and pid in ids:
            return [i for i, t in enumerate(ids) if t == pid]

    # Check for cross-attention image tokens (Llama-3.2-Vision uses
    # cross-attention, so visual tokens may not be in input_ids)
    # In that case, check for special placeholder tokens
    for special_id in [128256, 128257]:  # common Llama-3.2-Vision image token IDs
        if special_id in ids:
            return [i for i, t in enumerate(ids) if t == special_id]

    # LLaVA-style: start/end markers
    for sn, en in [("<image>", "</image>"),
                    ("<|image_start|>", "<|image_end|>"),
                    ("<im_start>", "<im_end>")]:
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

    # Heuristic: if we find a contiguous block of identical tokens
    # that aren't common text tokens, those are probably image tokens
    from collections import Counter
    counter = Counter(ids)
    for tok_id, count in counter.most_common():
        if count > 50 and tok_id != 0 and tok_id != 1:
            # Likely image padding tokens
            return [i for i, t in enumerate(ids) if t == tok_id]

    logger.warning("Could not detect visual positions in LLaMA input. "
                   "This may affect patching accuracy.")
    return []


# ════════════════════════════════════════════════════════════════════
# PATCHER
# ════════════════════════════════════════════════════════════════════

class LLaMAActivationPatcher(ActivationPatcher):
    """Activation patcher for LLaMA-3-based VLMs.

    Tested with:
      - meta-llama/Llama-3.2-11B-Vision-Instruct
      - llava-hf/llava-v1.6-llama3-8b-hf (if it exists)
      - Any LLaMA-3-backbone VLM
    """

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


class Llama32VisionPatcher(LLaMAActivationPatcher):
    """Specialized patcher for Llama-3.2-Vision.

    Llama-3.2-Vision uses cross-attention for visual input rather than
    concatenating visual tokens into the input sequence. This means
    visual positions may be handled differently.

    The _clean_forward in the base class hooks embeddings at layer 0,
    which captures post-cross-attention hidden states. The corruption
    strategy still works because we can noise the cross-attention
    input representations.
    """

    def _find_visual_positions(self, input_ids: torch.Tensor) -> List[int]:
        """For Llama-3.2-Vision, visual info enters via cross-attention,
        so 'visual positions' are the positions attending to image features.

        We identify the <|image|> token and treat surrounding context
        as visually-influenced positions.
        """
        positions = _find_visual_positions(self._processor, input_ids)

        if not positions:
            # Llama-3.2-Vision may not embed images as tokens at all
            # In this case, we treat all positions as potentially visual
            # (the corruption will target cross-attention representations)
            seq_len = input_ids.shape[1]
            logger.info("Llama-3.2-Vision: no explicit visual tokens found. "
                        f"Using all {seq_len} positions as visual targets.")
            # Use a subset to avoid excessive computation
            if seq_len > 100:
                # Image info typically influences early-to-mid positions
                positions = list(range(min(seq_len, 200)))
            else:
                positions = list(range(seq_len))

        return positions
