"""
LLaVA Activation Patcher
==================================

Critical difference from Qwen: LLaVA's input_ids contain a single
<image> placeholder token (e.g. ID=32000).  During forward(), the
model's prepare_inputs_labels_for_multimodal() replaces this with
~576 projected CLIP patch embeddings.  So:

  input_ids seq_len:  ~45   (text + 1 <image>)
  layer-0 embeds:     ~620  (text + 576 visual)

We CANNOT detect visual positions from input_ids alone.  We must:
  1. Find <image> position in input_ids (pre-expansion)
  2. Run forward → capture layer-0 embeds (post-expansion)
  3. Compute n_visual = expanded_len - original_len + n_placeholders
  4. Visual positions = [img_pos ... img_pos + n_visual - 1]

This patcher overrides _clean_forward to handle this.

Supports: LLaVA-1.5, LLaVA-NeXT (v1.6), LLaVA-OneVision.

Architecture (HuggingFace LlavaNext):
  LlavaNextForConditionalGeneration
    ├── vision_tower   (CLIP/SigLIP ViT)
    ├── multi_modal_projector  (MLP)
    ├── language_model (LlamaForCausalLM / MistralForCausalLM)
    │   ├── model  (LlamaModel: embed_tokens + layers + norm)
    │   └── lm_head
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .core import ActivationPatcher, CleanCache, PatchingConfig

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
# MODEL DISCOVERY
# ════════════════════════════════════════════════════════════════════

def _get_decoder_layers(model) -> nn.ModuleList:
    for path in [
        ("model", "language_model", "layers"),   # LlavaNext: model.language_model = MistralModel
        ("language_model", "model", "layers"),    # Some HF versions
        ("model", "model", "layers"),
        ("model", "layers"),
    ]:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            if isinstance(obj, nn.ModuleList) and len(obj) > 0:
                return obj
        except (AttributeError, IndexError, TypeError):
            continue
    raise RuntimeError("Could not locate decoder layers in LLaVA model.")


def _get_inner_model(model):
    """Inner model = the transformer decoder (layers + norm).

    For LLaVA, this is model.language_model.model (e.g. LlamaModel).
    We skip model.language_model (LlamaForCausalLM) because that
    includes lm_head and we want to call the decoder directly.
    """
    for path in [
        ("model", "language_model"),    # LlavaNext: MistralModel has .layers + .norm
        ("language_model", "model"),
        ("model", "model"),
        ("model",),
    ]:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            if hasattr(obj, "layers") and hasattr(obj, "norm"):
                return obj
        except (AttributeError, TypeError):
            continue
    logger.warning("Could not find inner model for LLaVA — ViT-skip disabled")
    return None


def _get_lm_head(model):
    for path in [("language_model", "lm_head"), ("lm_head",)]:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("Could not find lm_head in LLaVA model")


def _get_head_config(model) -> Tuple[int, int, int]:
    # Try text_config first (LlavaNextConfig wraps the LM config)
    for cfg_attr in ["text_config", "language_config"]:
        cfg = getattr(model.config, cfg_attr, None)
        if cfg is not None:
            n_heads = getattr(cfg, "num_attention_heads", None)
            n_kv = getattr(cfg, "num_key_value_heads", n_heads)
            hidden = getattr(cfg, "hidden_size", None)
            head_dim = getattr(cfg, "head_dim", None)
            if n_heads and hidden:
                if head_dim is None:
                    head_dim = hidden // n_heads
                return n_heads, n_kv or n_heads, head_dim
    # Try the language_model's own config
    lm = getattr(model, "language_model", None)
    if lm is not None:
        cfg = lm.config
        n_heads = getattr(cfg, "num_attention_heads", None)
        n_kv = getattr(cfg, "num_key_value_heads", n_heads)
        hidden = getattr(cfg, "hidden_size", None)
        if n_heads and hidden:
            return n_heads, n_kv or n_heads, hidden // n_heads
    return 32, 8, 128


# ════════════════════════════════════════════════════════════════════
# INPUT PREPARATION
# ════════════════════════════════════════════════════════════════════

def _prepare_inputs(processor, image, prompt: str, device):
    """Prepare inputs for LLaVA family processors."""
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt},
    ]}]
    try:
        text = processor.apply_chat_template(
            messages, add_generation_prompt=True)
        inputs = processor(
            text=text, images=image, return_tensors="pt")
    except Exception:
        # Fallback for older LLaVA-1.5
        conv_prompt = f"<image>\n{prompt}"
        inputs = processor(
            text=conv_prompt, images=image, return_tensors="pt")
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()}


def _find_image_placeholder_pos(model, input_ids: torch.Tensor
                                ) -> Tuple[List[int], int]:
    """Find <image> placeholder positions in input_ids (PRE-expansion).

    Returns:
        (positions, image_token_id)
    """
    ids = input_ids[0].cpu().tolist()

    # Method 1: config.image_token_index (most reliable)
    img_tok_id = getattr(model.config, "image_token_index", None)
    if img_tok_id is not None and img_tok_id in ids:
        pos = [i for i, t in enumerate(ids) if t == img_tok_id]
        return pos, img_tok_id

    # Method 2: try known special token IDs
    for candidate_id in [32000, 32001, 64000]:
        if candidate_id in ids:
            pos = [i for i, t in enumerate(ids) if t == candidate_id]
            return pos, candidate_id

    logger.error("Could not find image placeholder in LLaVA input_ids. "
                 f"Unique IDs: {sorted(set(ids))[:20]}")
    return [], -1


# ════════════════════════════════════════════════════════════════════
# PATCHER
# ════════════════════════════════════════════════════════════════════

class LLaVAActivationPatcher(ActivationPatcher):
    """Activation patcher for LLaVA family models.

    Overrides _clean_forward to handle LLaVA's dynamic visual token
    expansion (input_ids has 1 <image> token → layer-0 has ~576).
    """

    def __init__(self, model, processor, config: PatchingConfig = None):
        self._model = model
        self._processor = processor
        super().__init__(model, processor, config)

    # ── overrides ─────────────────────────────────────────────────

    def _prepare_inputs(self, image, prompt: str) -> dict:
        return _prepare_inputs(self._processor, image, prompt,
                               self._model.device)

    def _find_visual_positions(self, input_ids: torch.Tensor) -> List[int]:
        # Base class calls this, but for LLaVA it only returns
        # PLACEHOLDER positions (pre-expansion).  The real visual
        # positions are computed in our overridden _clean_forward.
        pos, _ = _find_image_placeholder_pos(self._model, input_ids)
        return pos

    def _get_decoder_layers(self) -> nn.ModuleList:
        return _get_decoder_layers(self._model)

    def _get_inner_model(self):
        return _get_inner_model(self._model)

    def _get_lm_head(self):
        return _get_lm_head(self._model)

    def _get_head_config(self) -> Tuple[int, int, int]:
        return _get_head_config(self._model)

    # ── override _clean_forward for post-expansion detection ──────

    @torch.no_grad()
    def _clean_forward(self, image, prompt: str) -> CleanCache:
        """LLaVA-specific clean forward with post-expansion visual detection.

        Key difference from base:
          1. Find <image> placeholder position in input_ids
          2. Run forward → capture layer-0 embeds (post-expansion)
          3. Compute actual visual positions from seq_len difference
        """
        inputs = self._prepare_inputs(image, prompt)
        input_ids = inputs["input_ids"]
        pre_seq_len = input_ids.shape[1]

        # Find placeholder position(s) before expansion
        placeholder_pos, img_tok_id = _find_image_placeholder_pos(
            self.model, input_ids)
        n_placeholders = len(placeholder_pos)

        if not placeholder_pos:
            logger.error("No image placeholders found — visual positions empty")

        # Storage for hooks
        captured_layer0 = {}
        captured_inner = {}
        captured_attn = {}
        captured_mlp = {}
        captured_pre_o = {}
        handles = []

        # Hook inner model kwargs (ViT-skip)
        inner = self._inner_model
        if inner is not None:
            def cap_inner(module, args, kwargs):
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor):
                        captured_inner[k] = v.detach().clone()
                    elif v is not None:
                        captured_inner[k] = v
                if args and len(args) >= 2 and isinstance(args[1], torch.Tensor):
                    if "inputs_embeds" not in captured_inner:
                        captured_inner["inputs_embeds"] = args[1].detach().clone()
                return None
            handles.append(inner.register_forward_pre_hook(
                cap_inner, with_kwargs=True))

        # Hook layer 0 for post-expansion embeds
        def cap_layer0(module, args, kwargs):
            hs = args[0] if args else kwargs.get("hidden_states")
            if hs is not None:
                captured_layer0["embeds"] = hs.detach().clone()
            return None
        handles.append(self._decoder_layers[0].register_forward_pre_hook(
            cap_layer0, with_kwargs=True))

        # Hook each layer's attn and mlp outputs
        for l_idx in range(self._n_layers):
            layer = self._decoder_layers[l_idx]

            def make_attn_hook(idx):
                def hook(module, inp, output):
                    out = output[0] if isinstance(output, tuple) else output
                    captured_attn[idx] = out.detach().clone()
                    return None
                return hook
            handles.append(layer.self_attn.register_forward_hook(
                make_attn_hook(l_idx)))

            def make_mlp_hook(idx):
                def hook(module, inp, output):
                    captured_mlp[idx] = output.detach().clone()
                    return None
                return hook
            handles.append(layer.mlp.register_forward_hook(
                make_mlp_hook(l_idx)))

            if self.config.granularity == "head" and hasattr(layer.self_attn, "o_proj"):
                def make_pre_o_hook(idx):
                    def hook(module, args):
                        captured_pre_o[idx] = args[0].detach().clone()
                        return None
                    return hook
                handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
                    make_pre_o_hook(l_idx)))

        # Run forward
        try:
            outputs = self.model(**inputs)
        finally:
            for h in handles:
                h.remove()

        # Get embeddings
        embeds = captured_layer0.get("embeds")
        if embeds is None:
            embeds = captured_inner.get("inputs_embeds")

        # ── CRITICAL: compute actual visual positions post-expansion ──
        if embeds is not None and placeholder_pos:
            post_seq_len = embeds.shape[1]
            n_visual = post_seq_len - pre_seq_len + n_placeholders
            # The first placeholder was at placeholder_pos[0] in the
            # original sequence.  After expansion, it occupies
            # [placeholder_pos[0] .. placeholder_pos[0] + n_visual - 1]
            first_img = placeholder_pos[0]
            vis_pos = list(range(first_img, first_img + n_visual))

            logger.info(f"LLaVA visual detection: input_ids={pre_seq_len}, "
                        f"embeds={post_seq_len}, n_visual={n_visual}, "
                        f"positions=[{first_img}..{first_img + n_visual - 1}]")

            # Sanity check: visual tokens should have distinct norms
            if n_visual > 0 and n_visual < post_seq_len:
                vis_norms = embeds[0, vis_pos, :].float().norm(dim=-1)
                text_pos = [i for i in range(post_seq_len) if i not in set(vis_pos)]
                if text_pos:
                    txt_norms = embeds[0, text_pos, :].float().norm(dim=-1)
                    logger.info(f"  Embed norms — visual: {vis_norms.mean():.1f} "
                                f"(std {vis_norms.std():.1f}), "
                                f"text: {txt_norms.mean():.1f} "
                                f"(std {txt_norms.std():.1f})")
        else:
            vis_pos = []
            logger.warning("Could not determine visual positions")

        # Build cache
        logits_last = outputs.logits[:, -1:, :]
        attn_list = [captured_attn.get(l) for l in range(self._n_layers)]
        mlp_list = [captured_mlp.get(l) for l in range(self._n_layers)]
        pre_o_list = [captured_pre_o.get(l) for l in range(self._n_layers)]

        logger.info(f"Clean cache: {len(vis_pos)} visual tokens, "
                    f"embeds {embeds.shape if embeds is not None else None}, "
                    f"attn at {sum(1 for a in attn_list if a is not None)} layers")

        return CleanCache(
            logits_last=logits_last,
            embeds=embeds,
            inner_kwargs=captured_inner,
            visual_positions=vis_pos,
            input_ids=input_ids,
            attn_out=attn_list,
            mlp_out=mlp_list,
            pre_o_proj=pre_o_list,
        )

    # ── override _build_inner_kwargs for LLaVA ────────────────────

    def _build_inner_kwargs(self, embeds: torch.Tensor,
                            inner_kwargs: dict) -> dict:
        """Build kwargs for LLaVA's inner language model.

        LLaVA's language model (LlamaModel) doesn't have visual-specific
        keys like Qwen.  We skip input_ids and provide inputs_embeds.
        Also skip pixel_values / image_sizes that the outer model uses.
        """
        SKIP_KEYS = {
            "input_ids", "inputs_embeds",
            "pixel_values", "pixel_values_videos",
            "image_sizes", "images",
        }
        kwargs = {}
        for k, v in inner_kwargs.items():
            if k in SKIP_KEYS:
                continue
            kwargs[k] = v
        kwargs["inputs_embeds"] = embeds
        kwargs["use_cache"] = False
        return kwargs


class LLaVANextPatcher(LLaVAActivationPatcher):
    """Alias for LLaVA-NeXT / LLaVA-1.6."""
    pass