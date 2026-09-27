"""
Hallucination Circuit Extraction & Cross-Architecture Comparison
================================================================

Given BatchPatchingResults from multiple models, this module:
  1. Identifies "hallucination circuits" — components with significant
     differential patching effects between correct and hallucinating samples
  2. Compares circuits across architectures to answer:
     "Do different VLMs develop the same hallucination pathways?"
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats
from scipy.spatial.distance import cosine as cosine_dist

from patching.core import BatchPatchingResult, PatchingResult, component_significance

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
# CIRCUIT DEFINITION
# ════════════════════════════════════════════════════════════════════

@dataclass
class CircuitComponent:
    """One component in a hallucination circuit."""
    layer: int
    component_type: str    # "attn" | "mlp" | "head"
    head_idx: int = -1     # only for head-type
    effect_correct: float = 0.0     # mean normalized effect (correct samples)
    effect_halluc: float = 0.0      # mean normalized effect (halluc samples)
    differential: float = 0.0       # correct - halluc
    p_value: float = 1.0
    cohen_d: float = 0.0

    @property
    def name(self) -> str:
        if self.component_type == "head":
            return f"L{self.layer}.H{self.head_idx}"
        return f"L{self.layer}.{self.component_type}"


@dataclass
class HallucinationCircuit:
    """A set of components that differentially process visual information
    for correct vs hallucinated outputs."""
    model_name: str
    n_layers: int
    n_heads: int
    components: List[CircuitComponent] = field(default_factory=list)

    # Raw matrices for visualization
    component_matrix_correct: Optional[np.ndarray] = None   # (n_layers, 2)
    component_matrix_halluc: Optional[np.ndarray] = None
    head_matrix_correct: Optional[np.ndarray] = None        # (n_layers, n_heads)
    head_matrix_halluc: Optional[np.ndarray] = None

    @property
    def n_components(self) -> int:
        return len(self.components)

    @property
    def attn_components(self) -> List[CircuitComponent]:
        return [c for c in self.components if c.component_type == "attn"]

    @property
    def mlp_components(self) -> List[CircuitComponent]:
        return [c for c in self.components if c.component_type == "mlp"]

    @property
    def head_components(self) -> List[CircuitComponent]:
        return [c for c in self.components if c.component_type == "head"]

    def top_k(self, k: int = 10) -> List[CircuitComponent]:
        """Top-k components by absolute differential effect."""
        return sorted(self.components,
                      key=lambda c: abs(c.differential), reverse=True)[:k]

    def layer_profile(self) -> np.ndarray:
        """Per-layer aggregate differential effect: (n_layers,)."""
        profile = np.zeros(self.n_layers)
        for c in self.components:
            profile[c.layer] += abs(c.differential)
        return profile


# ════════════════════════════════════════════════════════════════════
# CIRCUIT EXTRACTION
# ════════════════════════════════════════════════════════════════════

def extract_circuit(batch: BatchPatchingResult,
                    model_name: str,
                    alpha: float = 0.05,
                    min_effect_size: float = 0.3,
                    ) -> HallucinationCircuit:
    """Extract the hallucination circuit from patching results.

    A component is included in the circuit if:
      1. Its differential effect (correct - halluc) is statistically
         significant after BH correction (p < alpha), AND
      2. The effect size (Cohen's d) exceeds min_effect_size.

    Args:
        batch: aggregated patching results
        model_name: identifier for the model
        alpha: FDR threshold for BH correction
        min_effect_size: minimum Cohen's d to be included
    """
    if not batch.results:
        return HallucinationCircuit(model_name=model_name, n_layers=0, n_heads=0)

    n_layers = batch.results[0].n_layers
    n_heads = batch.results[0].n_heads

    # Compute mean matrices per category
    comp_correct = batch.mean_component_effects("correct")
    comp_halluc = batch.mean_component_effects("hallucination")

    head_correct = batch.mean_head_effects("correct")
    head_halluc = batch.mean_head_effects("hallucination")

    # Statistical tests at component level
    sig_results = component_significance(batch, alpha)

    components = []

    # Component-level (attn + mlp)
    comp_names = ["attn", "mlp"]
    if comp_correct.size > 0 and comp_halluc.size > 0:
        for l in range(n_layers):
            for c_idx, c_name in enumerate(comp_names):
                p = sig_results["p_values"][l, c_idx]
                d = sig_results["effect_sizes"][l, c_idx]
                is_sig = sig_results["significant"][l, c_idx]

                comp = CircuitComponent(
                    layer=l,
                    component_type=c_name,
                    effect_correct=float(comp_correct[l, c_idx]),
                    effect_halluc=float(comp_halluc[l, c_idx]),
                    differential=float(comp_correct[l, c_idx] - comp_halluc[l, c_idx]),
                    p_value=float(p),
                    cohen_d=float(d),
                )
                if is_sig and abs(d) >= min_effect_size:
                    components.append(comp)

    # Head-level (if available)
    if head_correct is not None and head_halluc is not None:
        correct_heads_all = np.array([
            r.normalized_heads for r in batch.correct_results
            if r.normalized_heads is not None])
        halluc_heads_all = np.array([
            r.normalized_heads for r in batch.hallucination_results
            if r.normalized_heads is not None])

        if correct_heads_all.shape[0] >= 3 and halluc_heads_all.shape[0] >= 3:
            for l in range(n_layers):
                for h in range(n_heads):
                    c_vals = correct_heads_all[:, l, h]
                    h_vals = halluc_heads_all[:, l, h]
                    t, p = sp_stats.ttest_ind(c_vals, h_vals, equal_var=False)
                    pooled = np.sqrt(
                        (c_vals.var(ddof=1) + h_vals.var(ddof=1)) / 2)
                    d = ((c_vals.mean() - h_vals.mean()) / pooled
                         if pooled > 1e-12 else 0.0)
                    if p < alpha and abs(d) >= min_effect_size:
                        components.append(CircuitComponent(
                            layer=l,
                            component_type="head",
                            head_idx=h,
                            effect_correct=float(head_correct[l, h]),
                            effect_halluc=float(head_halluc[l, h]),
                            differential=float(head_correct[l, h] - head_halluc[l, h]),
                            p_value=float(p),
                            cohen_d=float(d),
                        ))

    circuit = HallucinationCircuit(
        model_name=model_name,
        n_layers=n_layers,
        n_heads=n_heads,
        components=components,
        component_matrix_correct=comp_correct if comp_correct.size else None,
        component_matrix_halluc=comp_halluc if comp_halluc.size else None,
        head_matrix_correct=head_correct,
        head_matrix_halluc=head_halluc,
    )

    logger.info(f"Circuit for {model_name}: {circuit.n_components} components "
                f"({len(circuit.attn_components)} attn, "
                f"{len(circuit.mlp_components)} mlp, "
                f"{len(circuit.head_components)} head)")

    return circuit


# ════════════════════════════════════════════════════════════════════
# CROSS-ARCHITECTURE COMPARISON
# ════════════════════════════════════════════════════════════════════

@dataclass
class ArchitectureComparison:
    """Comparison of hallucination circuits across two architectures."""
    model_a: str
    model_b: str

    # Layer-profile similarity (after depth normalization)
    layer_profile_corr: float = 0.0     # Pearson correlation
    layer_profile_p: float = 1.0

    # Component overlap (Jaccard on significant components)
    component_overlap: float = 0.0

    # Per-layer cosine similarity of effect vectors
    layer_cosines: Optional[np.ndarray] = None

    # Summary
    interpretation: str = ""


def compare_architectures(circuit_a: HallucinationCircuit,
                          circuit_b: HallucinationCircuit,
                          ) -> ArchitectureComparison:
    """Compare hallucination circuits across two architectures.

    Since models may have different numbers of layers, we normalize
    layer indices to [0, 1] (relative depth) before comparing.
    """
    comp = ArchitectureComparison(
        model_a=circuit_a.model_name,
        model_b=circuit_b.model_name)

    # 1. Layer profile correlation (depth-normalized)
    prof_a = circuit_a.layer_profile()
    prof_b = circuit_b.layer_profile()

    if len(prof_a) > 0 and len(prof_b) > 0:
        # Resample to common resolution (100 bins)
        bins = 100
        norm_a = _resample_profile(prof_a, bins)
        norm_b = _resample_profile(prof_b, bins)
        r, p = sp_stats.pearsonr(norm_a, norm_b)
        comp.layer_profile_corr = float(r)
        comp.layer_profile_p = float(p)

    # 2. Component overlap via normalized layer ranges
    sig_a = _normalize_layers(
        [(c.layer, c.component_type) for c in circuit_a.components],
        circuit_a.n_layers)
    sig_b = _normalize_layers(
        [(c.layer, c.component_type) for c in circuit_b.components],
        circuit_b.n_layers)
    if sig_a and sig_b:
        # Discretize to thirds: early (0-0.33), mid (0.33-0.66), late (0.66-1)
        set_a = _discretize_components(sig_a)
        set_b = _discretize_components(sig_b)
        intersection = set_a & set_b
        union = set_a | set_b
        comp.component_overlap = len(intersection) / len(union) if union else 0.0

    # 3. Per-layer cosine similarity of component effect vectors
    if (circuit_a.component_matrix_correct is not None
            and circuit_b.component_matrix_correct is not None):
        diff_a = circuit_a.component_matrix_correct - (
            circuit_a.component_matrix_halluc
            if circuit_a.component_matrix_halluc is not None
            else np.zeros_like(circuit_a.component_matrix_correct))
        diff_b = circuit_b.component_matrix_correct - (
            circuit_b.component_matrix_halluc
            if circuit_b.component_matrix_halluc is not None
            else np.zeros_like(circuit_b.component_matrix_correct))

        bins = min(circuit_a.n_layers, circuit_b.n_layers)
        resamp_a = _resample_matrix(diff_a, bins)
        resamp_b = _resample_matrix(diff_b, bins)

        cosines = np.zeros(bins)
        for i in range(bins):
            va, vb = resamp_a[i], resamp_b[i]
            norm_p = np.linalg.norm(va) * np.linalg.norm(vb)
            cosines[i] = np.dot(va, vb) / norm_p if norm_p > 1e-12 else 0.0
        comp.layer_cosines = cosines

    # 4. Interpretation
    r = comp.layer_profile_corr
    j = comp.component_overlap
    if r > 0.7 and j > 0.5:
        comp.interpretation = (
            "CONVERGENT: Both architectures develop similar hallucination "
            "pathways — hallucination is likely driven by training data/objective, "
            "not architecture.")
    elif r < 0.3 and j < 0.2:
        comp.interpretation = (
            "DIVERGENT: Architectures develop distinct hallucination mechanisms. "
            "Architecture design significantly shapes hallucination pathways — "
            "targeted architectural interventions may help.")
    else:
        comp.interpretation = (
            "MIXED: Partial overlap in hallucination circuits. Some shared "
            "patterns (likely driven by shared training data) coexist with "
            "architecture-specific pathways.")

    return comp


# ════════════════════════════════════════════════════════════════════
# SERIALIZATION
# ════════════════════════════════════════════════════════════════════

def save_circuit(circuit: HallucinationCircuit, path: str):
    """Save circuit to JSON."""
    data = {
        "model_name": circuit.model_name,
        "n_layers": circuit.n_layers,
        "n_heads": circuit.n_heads,
        "n_components": circuit.n_components,
        "components": [
            {"layer": c.layer, "type": c.component_type,
             "head_idx": c.head_idx,
             "effect_correct": c.effect_correct,
             "effect_halluc": c.effect_halluc,
             "differential": c.differential,
             "p_value": c.p_value, "cohen_d": c.cohen_d}
            for c in circuit.components
        ],
        "component_matrix_correct": (
            circuit.component_matrix_correct.tolist()
            if circuit.component_matrix_correct is not None else None),
        "component_matrix_halluc": (
            circuit.component_matrix_halluc.tolist()
            if circuit.component_matrix_halluc is not None else None),
        "head_matrix_correct": (
            circuit.head_matrix_correct.tolist()
            if circuit.head_matrix_correct is not None else None),
        "head_matrix_halluc": (
            circuit.head_matrix_halluc.tolist()
            if circuit.head_matrix_halluc is not None else None),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved circuit to {path}")


def save_comparison(comp: ArchitectureComparison, path: str):
    """Save comparison to JSON."""
    data = {
        "model_a": comp.model_a,
        "model_b": comp.model_b,
        "layer_profile_corr": comp.layer_profile_corr,
        "layer_profile_p": comp.layer_profile_p,
        "component_overlap": comp.component_overlap,
        "layer_cosines": (
            comp.layer_cosines.tolist()
            if comp.layer_cosines is not None else None),
        "interpretation": comp.interpretation,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ════════════════════════════════════════════════════════════════════

def _resample_profile(profile: np.ndarray, bins: int) -> np.ndarray:
    """Resample a layer profile to fixed number of bins."""
    n = len(profile)
    if n == bins:
        return profile
    indices = np.linspace(0, n - 1, bins)
    return np.interp(indices, np.arange(n), profile)


def _resample_matrix(matrix: np.ndarray, bins: int) -> np.ndarray:
    """Resample rows (layers) of a matrix to fixed bins."""
    n_layers, n_cols = matrix.shape
    if n_layers == bins:
        return matrix
    result = np.zeros((bins, n_cols))
    for c in range(n_cols):
        result[:, c] = _resample_profile(matrix[:, c], bins)
    return result


def _normalize_layers(components: List[Tuple[int, str]],
                      n_layers: int) -> List[Tuple[float, str]]:
    """Normalize layer indices to [0, 1]."""
    if n_layers <= 1:
        return [(0.5, t) for _, t in components]
    return [(l / (n_layers - 1), t) for l, t in components]


def _discretize_components(normalized: List[Tuple[float, str]]
                           ) -> set:
    """Discretize normalized layer positions into thirds."""
    result = set()
    for depth, ctype in normalized:
        if depth < 0.33:
            region = "early"
        elif depth < 0.66:
            region = "mid"
        else:
            region = "late"
        result.add((region, ctype))
    return result
