"""
Activation Patching for VLM Hallucination Circuit Discovery
=============================================================

Core framework: capture clean activations, corrupt visual input,
patch individual components (layer × {attn, mlp, head}), measure
causal effect on output logits.

Protocol (Meng et al. 2022 adapted for VLMs):
  1. Clean run:   real image → full forward → cache all layer activations
  2. Corrupt run: noise on visual embeddings → forward → get baseline logits
  3. Patch run:   corrupt + restore component C at layer L → measure shift

The "indirect effect" of (L, C) = how much patching shifts output
toward the clean answer.  Large effect → C is a critical visual
information pathway.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════

@dataclass
class PatchingConfig:
    """Configuration for a patching experiment."""
    corrupt_mode: str = "noise"        # "noise" | "zero" | "mean"
    noise_std_mult: float = 3.0        # multiplier on visual token std
    granularity: str = "component"     # "component" (attn+mlp) | "head"
    patch_positions: str = "all"       # "all" | "visual" | "last"
    seed: int = 42


@dataclass
class CleanCache:
    """Cached activations from a clean forward pass."""
    logits_last: torch.Tensor          # (1, 1, vocab) — logits at last position
    embeds: torch.Tensor               # (1, seq, hidden) — layer-0 input
    inner_kwargs: dict                  # kwargs for inner model (ViT-skip)
    visual_positions: List[int]
    input_ids: torch.Tensor

    # Per-layer activations: list of length n_layers
    attn_out: List[torch.Tensor] = field(default_factory=list)
    mlp_out: List[torch.Tensor] = field(default_factory=list)
    pre_o_proj: List[Optional[torch.Tensor]] = field(default_factory=list)

    @property
    def n_layers(self) -> int:
        return len(self.attn_out)

    @property
    def n_visual(self) -> int:
        return len(self.visual_positions)


@dataclass
class PatchingResult:
    """Results from patching one (image, prompt) sample."""
    n_layers: int
    n_heads: int

    # Component effects: (n_layers, 2) where [:, 0]=attn, [:, 1]=mlp
    component_effects: np.ndarray

    # Head effects: (n_layers, n_heads) — None if granularity != "head"
    head_effects: Optional[np.ndarray] = None

    # Logit differences (correct - incorrect answer token)
    clean_logit_diff: float = 0.0
    corrupt_logit_diff: float = 0.0
    total_effect: float = 0.0   # clean - corrupt

    # Sample metadata
    question: str = ""
    gt_answer: str = ""
    model_answer: str = ""
    is_hallucination: bool = False
    n_visual_tokens: int = 0
    time_seconds: float = 0.0

    @property
    def normalized_component(self) -> np.ndarray:
        """Component effects normalized by |total effect|."""
        te = abs(self.total_effect)
        if te < 1e-8:
            return np.zeros_like(self.component_effects)
        return self.component_effects / te

    @property
    def normalized_heads(self) -> Optional[np.ndarray]:
        """Head effects normalized by |total effect|."""
        if self.head_effects is None:
            return None
        te = abs(self.total_effect)
        if te < 1e-8:
            return np.zeros_like(self.head_effects)
        return self.head_effects / te


@dataclass
class BatchPatchingResult:
    """Aggregated results across a batch of samples."""
    results: List[PatchingResult]

    # Aggregated by sample category
    correct_results: List[PatchingResult] = field(default_factory=list)
    hallucination_results: List[PatchingResult] = field(default_factory=list)

    def __post_init__(self):
        self.correct_results = [r for r in self.results if not r.is_hallucination]
        self.hallucination_results = [r for r in self.results if r.is_hallucination]

    @property
    def n_correct(self) -> int:
        return len(self.correct_results)

    @property
    def n_hallucination(self) -> int:
        return len(self.hallucination_results)

    def mean_component_effects(self, category: str = "all") -> np.ndarray:
        """Mean component effects for a category."""
        rs = self._select(category)
        if not rs:
            return np.array([])
        return np.mean([r.normalized_component for r in rs], axis=0)

    def mean_head_effects(self, category: str = "all") -> Optional[np.ndarray]:
        """Mean head effects for a category."""
        rs = self._select(category)
        heads = [r.normalized_heads for r in rs if r.normalized_heads is not None]
        if not heads:
            return None
        return np.mean(heads, axis=0)

    def _select(self, category: str) -> List[PatchingResult]:
        if category == "all":
            return self.results
        elif category == "correct":
            return self.correct_results
        elif category == "hallucination":
            return self.hallucination_results
        raise ValueError(f"Unknown category: {category}")


@dataclass
class PathwayInteractionResult:
    """Aggregated pathway interaction results across samples."""
    pathway_name: str
    n_components: int
    targets: List[Tuple[int, str, int]]

    # Per-sample measurements
    sample_results: List[dict] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self.sample_results)

    @property
    def n_correct(self) -> int:
        return sum(1 for r in self.sample_results if not r["is_hallucination"])

    @property
    def n_halluc(self) -> int:
        return sum(1 for r in self.sample_results if r["is_hallucination"])

    def mean_interaction(self, category: str = "all") -> float:
        rs = self._select(category)
        if not rs:
            return 0.0
        return float(np.mean([r["interaction"] for r in rs]))

    def mean_normalized_interaction(self, category: str = "all") -> float:
        rs = self._select(category)
        if not rs:
            return 0.0
        return float(np.mean([r["normalized_interaction"] for r in rs]))

    def interaction_stats(self, category: str = "all") -> dict:
        """Compute t-test on interaction ≠ 0 for a category."""
        rs = self._select(category)
        if len(rs) < 3:
            return {"mean": 0.0, "std": 0.0, "t": 0.0, "p": 1.0,
                    "cohen_d": 0.0, "n": len(rs)}
        vals = np.array([r["interaction"] for r in rs])
        t, p = sp_stats.ttest_1samp(vals, 0.0)
        d = vals.mean() / (vals.std(ddof=1) + 1e-12)
        return {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)),
            "t": float(t),
            "p": float(p),
            "cohen_d": float(d),
            "n": len(vals),
        }

    def differential_interaction(self) -> dict:
        """Test whether interaction differs between correct and halluc samples."""
        correct = [r["interaction"] for r in self.sample_results
                   if not r["is_hallucination"]]
        halluc = [r["interaction"] for r in self.sample_results
                  if r["is_hallucination"]]
        if len(correct) < 3 or len(halluc) < 3:
            return {"t": 0.0, "p": 1.0, "d": 0.0,
                    "mean_correct": 0.0, "mean_halluc": 0.0}
        c, h = np.array(correct), np.array(halluc)
        t, p = sp_stats.ttest_ind(c, h, equal_var=False)
        pooled = np.sqrt((c.var(ddof=1) + h.var(ddof=1)) / 2)
        d = (c.mean() - h.mean()) / (pooled + 1e-12)
        return {
            "t": float(t), "p": float(p), "d": float(d),
            "mean_correct": float(c.mean()),
            "mean_halluc": float(h.mean()),
        }

    def _select(self, category: str) -> list:
        if category == "all":
            return self.sample_results
        elif category == "correct":
            return [r for r in self.sample_results if not r["is_hallucination"]]
        elif category == "hallucination":
            return [r for r in self.sample_results if r["is_hallucination"]]
        raise ValueError(f"Unknown: {category}")


# ════════════════════════════════════════════════════════════════════
# STATISTICS
# ════════════════════════════════════════════════════════════════════

def component_significance(batch: BatchPatchingResult, alpha: float = 0.05
                           ) -> Dict[str, np.ndarray]:
    """Test which components differ between correct and hallucination samples.

    For each (layer, component), runs a two-sample t-test on normalized
    patching effects between correct and hallucination groups.

    Returns dict with keys:
      - t_stats: (n_layers, 2)
      - p_values: (n_layers, 2)
      - effect_sizes: (n_layers, 2) — Cohen's d
      - significant: (n_layers, 2) bool after BH correction
    """
    if batch.n_correct < 3 or batch.n_hallucination < 3:
        logger.warning("Too few samples for significance testing "
                       f"(correct={batch.n_correct}, "
                       f"halluc={batch.n_hallucination})")
        n = batch.results[0].n_layers if batch.results else 0
        shape = (n, 2)
        return {"t_stats": np.zeros(shape), "p_values": np.ones(shape),
                "effect_sizes": np.zeros(shape),
                "significant": np.zeros(shape, dtype=bool)}

    # corr, hall: (n_samples, n_layers, 2)
    corr = np.array([r.normalized_component for r in batch.correct_results])
    hall = np.array([r.normalized_component for r in batch.hallucination_results])

    n_layers, n_comp = corr.shape[1], corr.shape[2]
    t_stats = np.zeros((n_layers, n_comp))
    p_values = np.ones((n_layers, n_comp))
    effect_sizes = np.zeros((n_layers, n_comp))

    for l in range(n_layers):
        for c in range(n_comp):
            c_vals = corr[:, l, c]
            h_vals = hall[:, l, c]
            t, p = sp_stats.ttest_ind(c_vals, h_vals, equal_var=False)
            t_stats[l, c] = t
            p_values[l, c] = p
            # Cohen's d
            pooled_std = np.sqrt((c_vals.var(ddof=1) + h_vals.var(ddof=1)) / 2)
            if pooled_std > 1e-12:
                effect_sizes[l, c] = (c_vals.mean() - h_vals.mean()) / pooled_std
            else:
                effect_sizes[l, c] = 0.0

    # BH correction on flattened p-values
    flat_p = p_values.flatten()
    _, sig_flat = benjamini_hochberg(flat_p, alpha)
    significant = sig_flat.reshape(n_layers, n_comp)

    return {"t_stats": t_stats, "p_values": p_values,
            "effect_sizes": effect_sizes, "significant": significant}


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """BH FDR correction. Returns (adjusted_p, significant_mask)."""
    m = len(p_values)
    if m == 0:
        return np.array([]), np.array([], dtype=bool)
    idx = np.argsort(p_values)
    sorted_p = p_values[idx]
    significant = np.zeros(m, dtype=bool)
    max_k = -1
    for k in range(m):
        if sorted_p[k] <= (k + 1) / m * alpha:
            max_k = k
    if max_k >= 0:
        significant[idx[:max_k + 1]] = True
    adj_p = np.ones(m)
    for k in range(m - 1, -1, -1):
        i = idx[k]
        adj_p[i] = min(sorted_p[k] * m / (k + 1), 1.0)
        if k < m - 1:
            adj_p[i] = min(adj_p[i], adj_p[idx[k + 1]])
    return adj_p, significant


# ════════════════════════════════════════════════════════════════════
# BASE PATCHER
# ════════════════════════════════════════════════════════════════════

class ActivationPatcher:
    """
    Base class for activation patching in VLMs.

    Subclasses implement model-specific methods:
      - _prepare_inputs(image, prompt) → model inputs dict
      - _find_visual_positions(input_ids) → list[int]
      - _get_decoder_layers() → nn.ModuleList
      - _get_inner_model() → inner transformer model
      - _get_lm_head() → lm_head linear
      - _get_head_config() → (n_heads, n_kv_heads, head_dim)
    """

    def __init__(self, model, processor, config: PatchingConfig = None):
        self.model = model
        self.processor = processor
        self.config = config or PatchingConfig()
        self._decoder_layers = self._get_decoder_layers()
        self._inner_model = self._get_inner_model()
        self._lm_head = self._get_lm_head()
        self._n_heads, self._n_kv_heads, self._head_dim = self._get_head_config()
        self._n_layers = len(self._decoder_layers)
        logger.info(f"ActivationPatcher: {self._n_layers} layers, "
                    f"{self._n_heads} heads, head_dim={self._head_dim}")

    # ── public API ────────────────────────────────────────────────

    @torch.no_grad()
    def patch_sample(self, image, prompt: str,
                     correct_token: str, incorrect_token: str,
                     gt_answer: str = "",
                     ) -> PatchingResult:
        """Run full activation patching on one sample.

        Args:
            image: PIL Image
            prompt: question text (with answer instruction)
            correct_token: token for the correct answer ("Yes"/"No")
            incorrect_token: token for the incorrect answer
            gt_answer: ground truth label
        """
        t0 = time.time()

        # Resolve answer token ids
        correct_ids = self._answer_token_ids(correct_token)
        incorrect_ids = self._answer_token_ids(incorrect_token)

        # Step 1: clean forward → cache activations
        clean = self._clean_forward(image, prompt)

        # Step 2: corrupt forward → baseline logits
        corrupt_embeds = self._make_corrupt_embeds(clean)
        corrupt_logits = self._forward_with_embeds(
            corrupt_embeds, clean.inner_kwargs)

        # Logit diffs
        clean_ld = self._logit_diff(clean.logits_last, correct_ids, incorrect_ids)
        corrupt_ld = self._logit_diff(corrupt_logits, correct_ids, incorrect_ids)
        total_effect = clean_ld - corrupt_ld

        # Step 3: patch each component
        comp_effects = np.zeros((self._n_layers, 2))  # attn, mlp
        head_effects = None

        for layer_idx in range(self._n_layers):
            # Patch attention
            patched_attn = self._forward_patched(
                corrupt_embeds, clean, layer_idx, "attn")
            comp_effects[layer_idx, 0] = (
                self._logit_diff(patched_attn, correct_ids, incorrect_ids)
                - corrupt_ld
            )

            # Patch MLP
            patched_mlp = self._forward_patched(
                corrupt_embeds, clean, layer_idx, "mlp")
            comp_effects[layer_idx, 1] = (
                self._logit_diff(patched_mlp, correct_ids, incorrect_ids)
                - corrupt_ld
            )

        # Head-level patching (optional)
        if self.config.granularity == "head" and clean.pre_o_proj:
            head_effects = np.zeros((self._n_layers, self._n_heads))
            for layer_idx in range(self._n_layers):
                if clean.pre_o_proj[layer_idx] is None:
                    continue
                for head_idx in range(self._n_heads):
                    patched = self._forward_patched(
                        corrupt_embeds, clean, layer_idx, "head",
                        head_idx=head_idx)
                    head_effects[layer_idx, head_idx] = (
                        self._logit_diff(patched, correct_ids, incorrect_ids)
                        - corrupt_ld
                    )
                if (layer_idx + 1) % 7 == 0:
                    logger.debug(f"  Head patching: {layer_idx+1}/{self._n_layers}")

        dt = time.time() - t0

        # Determine if model hallucinates
        model_correct = clean_ld > 0
        is_hallucination = not model_correct

        return PatchingResult(
            n_layers=self._n_layers,
            n_heads=self._n_heads,
            component_effects=comp_effects,
            head_effects=head_effects,
            clean_logit_diff=float(clean_ld),
            corrupt_logit_diff=float(corrupt_ld),
            total_effect=float(total_effect),
            question=prompt,
            gt_answer=gt_answer,
            is_hallucination=is_hallucination,
            n_visual_tokens=clean.n_visual,
            time_seconds=dt,
        )

    # ── clean forward with activation capture ─────────────────────

    def _clean_forward(self, image, prompt: str) -> CleanCache:
        """Full forward pass, caching all internal activations."""
        inputs = self._prepare_inputs(image, prompt)
        vis_pos = self._find_visual_positions(inputs["input_ids"])

        # Storage for hooks
        captured_layer0 = {}
        captured_inner = {}
        captured_attn = {}
        captured_mlp = {}
        captured_pre_o = {}
        handles = []

        # Hook inner model to capture kwargs (for ViT-skip)
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

        # Hook layer 0 to capture embeddings (post-ViT merge)
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

            # Attention output — handle both self_attn and cross_attn
            # (Llama-3.2-Vision interleaves MllamaSelfAttentionDecoderLayer
            #  and MllamaCrossAttentionDecoderLayer)
            attn_module = getattr(layer, "self_attn", None) or \
                          getattr(layer, "cross_attn", None)

            if attn_module is not None:
                def make_attn_hook(idx):
                    def hook(module, inp, output):
                        out = output[0] if isinstance(output, tuple) else output
                        captured_attn[idx] = out.detach().clone()
                        return None
                    return hook
                handles.append(attn_module.register_forward_hook(
                    make_attn_hook(l_idx)))

            # MLP output
            if hasattr(layer, "mlp"):
                def make_mlp_hook(idx):
                    def hook(module, inp, output):
                        captured_mlp[idx] = output.detach().clone()
                        return None
                    return hook
                handles.append(layer.mlp.register_forward_hook(
                    make_mlp_hook(l_idx)))

            # Pre-o_proj (for head-level patching)
            if self.config.granularity == "head" and attn_module is not None \
                    and hasattr(attn_module, "o_proj"):
                def make_pre_o_hook(idx):
                    def hook(module, args):
                        captured_pre_o[idx] = args[0].detach().clone()
                        return None
                    return hook
                handles.append(attn_module.o_proj.register_forward_pre_hook(
                    make_pre_o_hook(l_idx)))

        # Run forward
        try:
            outputs = self.model(**inputs)
        finally:
            for h in handles:
                h.remove()

        # Build cache
        embeds = captured_layer0.get("embeds")
        if embeds is None:
            embeds = captured_inner.get("inputs_embeds")

        logits_last = outputs.logits[:, -1:, :]

        attn_list = [captured_attn.get(l) for l in range(self._n_layers)]
        mlp_list = [captured_mlp.get(l) for l in range(self._n_layers)]
        pre_o_list = [captured_pre_o.get(l) for l in range(self._n_layers)]

        logger.info(f"Clean cache: {len(vis_pos)} visual tokens, "
                    f"embeds {embeds.shape if embeds is not None else None}, "
                    f"captured attn at {sum(1 for a in attn_list if a is not None)} layers")

        return CleanCache(
            logits_last=logits_last,
            embeds=embeds,
            inner_kwargs=captured_inner,
            visual_positions=vis_pos,
            input_ids=inputs["input_ids"],
            attn_out=attn_list,
            mlp_out=mlp_list,
            pre_o_proj=pre_o_list,
        )

    # ── corruption ────────────────────────────────────────────────

    def _make_corrupt_embeds(self, clean: CleanCache) -> torch.Tensor:
        """Create corrupted embeddings by noising visual positions."""
        embeds = clean.embeds.clone()
        vis_pos = clean.visual_positions

        if not vis_pos:
            return embeds

        vis_h = embeds[:, vis_pos, :]
        mode = self.config.corrupt_mode

        if mode == "noise":
            std = vis_h.std().item() * self.config.noise_std_mult
            rng = torch.Generator(device=embeds.device)
            rng.manual_seed(self.config.seed)
            noise = torch.randn(
                vis_h.shape, generator=rng,
                device=embeds.device, dtype=embeds.dtype) * std
            embeds[:, vis_pos, :] = vis_h + noise

        elif mode == "zero":
            embeds[:, vis_pos, :] = 0.0

        elif mode == "mean":
            mean_tok = vis_h.mean(dim=1, keepdim=True)
            embeds[:, vis_pos, :] = mean_tok.expand_as(vis_h)

        return embeds

    # ── forward helpers ───────────────────────────────────────────

    def _forward_with_embeds(self, embeds: torch.Tensor,
                             inner_kwargs: dict) -> torch.Tensor:
        """Forward through inner model (ViT-skip). Returns last-pos logits."""
        kwargs = self._build_inner_kwargs(embeds, inner_kwargs)
        hs = self._call_inner(kwargs)
        logits = self._lm_head(hs)
        return logits[:, -1:, :]  # (1, 1, vocab)

    def _forward_patched(self, corrupt_embeds: torch.Tensor,
                         clean: CleanCache, layer_idx: int,
                         component: str,
                         head_idx: int = -1) -> torch.Tensor:
        """Forward with corruption + one component patched from clean run.

        Args:
            corrupt_embeds: pre-computed corrupted embeddings
            clean: clean run cache
            layer_idx: which layer to patch
            component: "attn" | "mlp" | "head"
            head_idx: which head (only for component="head")

        Returns:
            logits at last position: (1, 1, vocab)
        """
        handles = []
        layer = self._decoder_layers[layer_idx]
        attn_module = getattr(layer, "self_attn", None) or \
                      getattr(layer, "cross_attn", None)

        if component == "attn":
            clean_val = clean.attn_out[layer_idx]
            if clean_val is None or attn_module is None:
                return self._forward_with_embeds(
                    corrupt_embeds, clean.inner_kwargs)

            def patch_hook(module, inp, output):
                # Defensive device cast: with device_map="auto" on large models,
                # layers may live on different GPUs.
                # Same-layer same-run normally matches, but cast to be safe.
                ref = output[0] if isinstance(output, tuple) else output
                cv = clean_val.to(device=ref.device, dtype=ref.dtype)
                if isinstance(output, tuple):
                    return (cv,) + output[1:]
                return cv
            handles.append(attn_module.register_forward_hook(patch_hook))

        elif component == "mlp":
            clean_val = clean.mlp_out[layer_idx]
            if clean_val is None or not hasattr(layer, "mlp"):
                return self._forward_with_embeds(
                    corrupt_embeds, clean.inner_kwargs)

            def patch_hook(module, inp, output):
                # Defensive device cast (see attn branch above).
                ref = output[0] if isinstance(output, tuple) else output
                return clean_val.to(device=ref.device, dtype=ref.dtype)
            handles.append(layer.mlp.register_forward_hook(patch_hook))

        elif component == "head":
            clean_pre_o = clean.pre_o_proj[layer_idx]
            if clean_pre_o is None or attn_module is None or \
                    not hasattr(attn_module, "o_proj"):
                return self._forward_with_embeds(
                    corrupt_embeds, clean.inner_kwargs)

            head_dim = self._head_dim
            n_heads = self._n_heads

            def patch_head_hook(module, args):
                pre_o = args[0]  # (B, seq, n_heads * head_dim)
                B, S, D = pre_o.shape
                pre_o_r = pre_o.view(B, S, n_heads, head_dim).clone()
                # Defensive device/dtype cast for multi-GPU sharded models.
                clean_r = clean_pre_o.to(
                    device=pre_o.device, dtype=pre_o.dtype
                ).view(1, S, n_heads, head_dim)
                pre_o_r[:, :, head_idx, :] = clean_r[:, :, head_idx, :]
                return (pre_o_r.reshape(B, S, D),)
            handles.append(
                attn_module.o_proj.register_forward_pre_hook(patch_head_hook))

        try:
            logits = self._forward_with_embeds(
                corrupt_embeds, clean.inner_kwargs)
        finally:
            for h in handles:
                h.remove()

        return logits

    # ── multi-component (pathway) patching ───────────────────────

    def _forward_patched_multi(self, corrupt_embeds: torch.Tensor,
                                clean: CleanCache,
                                targets: List[Tuple[int, str, int]]
                                ) -> torch.Tensor:
        """Forward with corruption + MULTIPLE components patched simultaneously.

        This is the key extension for Conditional Pathway Analysis (CPA).
        Instead of patching one component at a time, we restore an entire
        pathway (set of components) from the clean run, allowing us to
        measure joint causal effects and pathway interactions.

        Args:
            corrupt_embeds: pre-computed corrupted embeddings
            clean: clean run cache
            targets: list of (layer_idx, component_type, head_idx) tuples
                     component_type: "attn" | "mlp" | "head"
                     head_idx: -1 for attn/mlp, specific index for head

        Returns:
            logits at last position: (1, 1, vocab)
        """
        handles = []

        for layer_idx, component, head_idx in targets:
            layer = self._decoder_layers[layer_idx]
            attn_module = getattr(layer, "self_attn", None) or \
                          getattr(layer, "cross_attn", None)

            if component == "attn":
                clean_val = clean.attn_out[layer_idx]
                if clean_val is None or attn_module is None:
                    continue

                def make_attn_hook(cv=clean_val):
                    def hook(module, inp, output):
                        # Defensive device cast for sharded large models.
                        ref = output[0] if isinstance(output, tuple) else output
                        cv2 = cv.to(device=ref.device, dtype=ref.dtype)
                        if isinstance(output, tuple):
                            return (cv2,) + output[1:]
                        return cv2
                    return hook
                handles.append(
                    attn_module.register_forward_hook(make_attn_hook()))

            elif component == "mlp":
                clean_val = clean.mlp_out[layer_idx]
                if clean_val is None or not hasattr(layer, "mlp"):
                    continue

                def make_mlp_hook(cv=clean_val):
                    def hook(module, inp, output):
                        ref = output[0] if isinstance(output, tuple) else output
                        return cv.to(device=ref.device, dtype=ref.dtype)
                    return hook
                handles.append(
                    layer.mlp.register_forward_hook(make_mlp_hook()))

            elif component == "head":
                clean_pre_o = clean.pre_o_proj[layer_idx]
                if clean_pre_o is None or attn_module is None or \
                        not hasattr(attn_module, "o_proj"):
                    continue

                head_dim = self._head_dim
                n_heads = self._n_heads

                def make_head_hook(cp=clean_pre_o, hi=head_idx,
                                   hd=head_dim, nh=n_heads):
                    def hook(module, args):
                        pre_o = args[0]
                        B, S, D = pre_o.shape
                        pre_o_r = pre_o.view(B, S, nh, hd).clone()
                        clean_r = cp.to(
                            device=pre_o.device, dtype=pre_o.dtype
                        ).view(1, S, nh, hd)
                        pre_o_r[:, :, hi, :] = clean_r[:, :, hi, :]
                        return (pre_o_r.reshape(B, S, D),)
                    return hook
                handles.append(
                    attn_module.o_proj.register_forward_pre_hook(
                        make_head_hook()))

        try:
            logits = self._forward_with_embeds(
                corrupt_embeds, clean.inner_kwargs)
        finally:
            for h in handles:
                h.remove()

        return logits

    @torch.no_grad()
    def patch_pathway_interaction(self, image, prompt: str,
                                   correct_token: str,
                                   incorrect_token: str,
                                   pathway_targets: List[Tuple[int, str, int]],
                                   gt_answer: str = "",
                                   ) -> dict:
        """Measure pathway-level causal effects and interaction terms.

        Computes:
          - IE(pathway):  joint effect of restoring all pathway components
          - Σ IE(ci):     sum of individual component effects
          - Interaction:  IE(pathway) - Σ IE(ci)  (synergy if > 0, redundancy if < 0)
          - Normalized interaction: Interaction / |IE(pathway)|

        A significant interaction term proves the pathway is a genuine
        functional unit, not just a bag of independently-acting components.

        Args:
            image: PIL Image
            prompt: question text
            correct_token, incorrect_token: answer tokens
            pathway_targets: list of (layer_idx, component_type, head_idx)
            gt_answer: ground truth label

        Returns:
            dict with all effect measurements
        """
        correct_ids = self._answer_token_ids(correct_token)
        incorrect_ids = self._answer_token_ids(incorrect_token)

        # Step 1: clean forward
        clean = self._clean_forward(image, prompt)

        # Step 2: corrupt baseline
        corrupt_embeds = self._make_corrupt_embeds(clean)
        corrupt_logits = self._forward_with_embeds(
            corrupt_embeds, clean.inner_kwargs)
        corrupt_ld = self._logit_diff(corrupt_logits, correct_ids, incorrect_ids)
        clean_ld = self._logit_diff(clean.logits_last, correct_ids, incorrect_ids)

        # Step 3: individual effects — Σ IE(ci)
        individual_effects = []
        for target in pathway_targets:
            layer_idx, component, head_idx = target
            patched = self._forward_patched(
                corrupt_embeds, clean, layer_idx, component,
                head_idx=head_idx)
            ie = self._logit_diff(patched, correct_ids, incorrect_ids) - corrupt_ld
            individual_effects.append(ie)

        sum_individual = sum(individual_effects)

        # Step 4: joint pathway effect — IE(pathway)
        joint_logits = self._forward_patched_multi(
            corrupt_embeds, clean, pathway_targets)
        joint_ie = self._logit_diff(
            joint_logits, correct_ids, incorrect_ids) - corrupt_ld

        # Step 5: interaction
        interaction = joint_ie - sum_individual
        norm_interaction = (interaction / abs(joint_ie)
                            if abs(joint_ie) > 1e-8 else 0.0)

        model_correct = clean_ld > 0

        return {
            "joint_ie": float(joint_ie),
            "sum_individual_ie": float(sum_individual),
            "interaction": float(interaction),
            "normalized_interaction": float(norm_interaction),
            "individual_effects": [float(e) for e in individual_effects],
            "n_components": len(pathway_targets),
            "clean_logit_diff": float(clean_ld),
            "corrupt_logit_diff": float(corrupt_ld),
            "total_effect": float(clean_ld - corrupt_ld),
            "is_hallucination": not model_correct,
            "gt_answer": gt_answer,
        }

    @torch.no_grad()
    def logit_lens_profile(self, image, prompt: str,
                            correct_token: str,
                            incorrect_token: str,
                            ) -> dict:
        """Apply logit lens at every layer to trace how the model builds
        its answer through the network.

        At each layer, projects the hidden state through lm_head to get
        intermediate logits, then records:
          - P(correct) and P(incorrect) at each layer
          - rank of correct and incorrect tokens
          - logit_diff trajectory

        This validates the dual-pathway hypothesis: grounding-pathway
        layers should progressively increase P(correct), while
        hallucination-pathway layers may boost P(incorrect).

        Returns:
            dict with per-layer measurements
        """
        correct_ids = self._answer_token_ids(correct_token)
        incorrect_ids = self._answer_token_ids(incorrect_token)

        inputs = self._prepare_inputs(image, prompt)

        # Capture hidden states at every layer
        hidden_states = {}
        handles = []

        for l_idx in range(self._n_layers):
            layer = self._decoder_layers[l_idx]

            def make_hook(idx):
                def hook(module, input, output):
                    # output is typically (hidden_states, ...) tuple
                    hs = output[0] if isinstance(output, tuple) else output
                    hidden_states[idx] = hs.detach()
                    return None
                return hook
            handles.append(
                layer.register_forward_hook(make_hook(l_idx)))

        try:
            outputs = self.model(**inputs)
        finally:
            for h in handles:
                h.remove()

        # Get the final norm (for proper projection)
        final_norm = None
        inner = self._get_inner_model()
        if inner is not None and hasattr(inner, "norm"):
            final_norm = inner.norm
        elif hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            final_norm = self.model.model.norm

        lm_head = self._lm_head

        # Per-layer logit lens
        layer_data = []
        for l_idx in range(self._n_layers):
            hs = hidden_states.get(l_idx)
            if hs is None:
                layer_data.append(None)
                continue

            # Take last-position hidden state
            hs_last = hs[:, -1:, :]  # (1, 1, hidden)

            # For sharded large models, hs lives on the captured layer's
            # device but final_norm / lm_head live on the last device.
            # Move hs to wherever lm_head is.
            target_device = lm_head.weight.device
            if hs_last.device != target_device:
                hs_last = hs_last.to(target_device)

            # Apply final norm before projection (critical for accuracy)
            if final_norm is not None:
                hs_normed = final_norm(hs_last)
            else:
                hs_normed = hs_last

            # Project through lm_head
            logits = lm_head(hs_normed)  # (1, 1, vocab)
            probs = torch.softmax(logits[0, -1, :].float(), dim=-1)
            lgs = logits[0, -1, :].float()

            # Correct / incorrect token stats
            p_correct = max(probs[tid].item() for tid in correct_ids)
            p_incorrect = max(probs[tid].item() for tid in incorrect_ids)
            l_correct = max(lgs[tid].item() for tid in correct_ids)
            l_incorrect = max(lgs[tid].item() for tid in incorrect_ids)

            # Ranks
            sorted_indices = torch.argsort(lgs, descending=True)
            rank_map = {idx.item(): rank
                        for rank, idx in enumerate(sorted_indices[:500])}
            rank_correct = min(rank_map.get(tid, 500) for tid in correct_ids)
            rank_incorrect = min(rank_map.get(tid, 500) for tid in incorrect_ids)

            layer_data.append({
                "layer": l_idx,
                "depth": l_idx / max(self._n_layers - 1, 1),
                "p_correct": float(p_correct),
                "p_incorrect": float(p_incorrect),
                "logit_correct": float(l_correct),
                "logit_incorrect": float(l_incorrect),
                "logit_diff": float(l_correct - l_incorrect),
                "rank_correct": int(rank_correct),
                "rank_incorrect": int(rank_incorrect),
            })

        # Final output
        final_logits = outputs.logits[:, -1:, :]
        final_ld = self._logit_diff(final_logits, correct_ids, incorrect_ids)

        return {
            "layers": [d for d in layer_data if d is not None],
            "final_logit_diff": float(final_ld),
            "is_hallucination": final_ld <= 0,
            "n_layers": self._n_layers,
        }

    # ── utilities ─────────────────────────────────────────────────

    def _build_inner_kwargs(self, embeds: torch.Tensor,
                            inner_kwargs: dict) -> dict:
        """Build kwargs for inner model from cached values + new embeds."""
        SKIP_KEYS = {
            "input_ids", "inputs_embeds",
            "pixel_values", "pixel_values_videos",
            "image_grid_thw", "video_grid_thw",
            "mm_token_type_ids",
        }
        kwargs = {}
        for k, v in inner_kwargs.items():
            if k in SKIP_KEYS:
                continue
            kwargs[k] = v
        kwargs["inputs_embeds"] = embeds
        kwargs["use_cache"] = False
        return kwargs

    def _call_inner(self, kwargs: dict) -> torch.Tensor:
        """Call inner model, handle _deepstack_process monkey-patch."""
        original = getattr(self._inner_model, "_deepstack_process", None)
        def noop(hidden_states, *a, **kw):
            return hidden_states
        if original is not None:
            self._inner_model._deepstack_process = noop
        try:
            outputs = self._inner_model(**kwargs)
            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state
            return outputs[0]
        finally:
            if original is not None:
                self._inner_model._deepstack_process = original

    def _logit_diff(self, logits: torch.Tensor,
                    correct_ids: List[int],
                    incorrect_ids: List[int]) -> float:
        """Compute logit(correct) - logit(incorrect) at last position."""
        lgs = logits[0, -1, :].float()
        c = max(lgs[tid].item() for tid in correct_ids)
        i = max(lgs[tid].item() for tid in incorrect_ids)
        return c - i

    def _answer_token_ids(self, answer: str) -> List[int]:
        """Get token IDs for an answer string, trying multiple casing."""
        tokenizer = self.processor.tokenizer
        ids = set()
        for variant in [answer, answer.lower(), answer.upper(),
                        answer.capitalize(), f" {answer}", f" {answer.lower()}"]:
            toks = tokenizer.encode(variant, add_special_tokens=False)
            if toks:
                ids.add(toks[0])
        if not ids:
            raise ValueError(f"Cannot tokenize answer: {answer!r}")
        return list(ids)

    # ── subclass interface (model-specific) ───────────────────────

    def _prepare_inputs(self, image, prompt: str) -> dict:
        raise NotImplementedError

    def _find_visual_positions(self, input_ids: torch.Tensor) -> List[int]:
        raise NotImplementedError

    def _get_decoder_layers(self):
        raise NotImplementedError

    def _get_inner_model(self):
        raise NotImplementedError

    def _get_lm_head(self):
        raise NotImplementedError

    def _get_head_config(self) -> Tuple[int, int, int]:
        """Returns (n_heads, n_kv_heads, head_dim)."""
        raise NotImplementedError