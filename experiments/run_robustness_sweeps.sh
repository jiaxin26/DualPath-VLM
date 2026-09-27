#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  run_robustness_sweeps.sh — Robustness experiments driver
# ═══════════════════════════════════════════════════════════════════
#
# Experiments covered:
#   A. Noise level sweep     (corruption = 2σ, 3σ, 4σ)
#   B. Corruption type sweep (noise, zero, mean)
#   C. Seed variance         (3 seeds)
#   D. Top-k per-sample      (for later bootstrap)
#
# Budget (all on H200; Qwen is the cheapest ~2h/1000 samples):
#   A:  ~18 GPU-h  (2 models × 3 noise levels × 3h avg)
#   B:  ~12 GPU-h  (2 models × 3 types × 2h avg, overlaps with A)
#   C:  ~18 GPU-h  (2 models × 3 seeds × 3h avg)
#   D:  ~6  GPU-h  (LLaVA only, 7 k values × ~0.5h each)
#   Total: ~50 GPU-h. Cut in half by using only Qwen (fastest).
#
# ─────────────────────────────────────────────────────────────────────
#
# Usage:
#   bash experiments/run_robustness_sweeps.sh            # run everything
#   bash experiments/run_robustness_sweeps.sh A          # noise only
#   bash experiments/run_robustness_sweeps.sh A B C D    # explicit set
#   MODELS="qwen" bash experiments/run_robustness_sweeps.sh A  # single model
#
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

# ── Config ───────────────────────────────────────────────────────
RESULTS="${RESULTS:-results/}"
ROBUSTNESS="${ROBUSTNESS:-results/robustness/}"
POPE_PATH="${POPE_PATH:-data/pope/}"
IMAGE_DIR="${IMAGE_DIR:-data/coco/val2014/}"
N_SAMPLES="${N_SAMPLES:-500}"
N_CPA="${N_CPA:-200}"                # for CPA on sensitivity runs
N_INTERVENTION="${N_INTERVENTION:-500}"

# Default models for each experiment (Qwen = fastest, InternVL8B = tight circuit)
# Override with: MODELS="qwen llava" bash ...
MODELS="${MODELS:-qwen internvl8b}"

declare -A MODEL_IDS=(
    [qwen]="Qwen/Qwen3-VL-8B-Instruct"
    [llava]="llava-hf/llava-v1.6-mistral-7b-hf"
    [internvl8b]="OpenGVLab/InternVL3-8B-hf"
    [llama]="meta-llama/Llama-3.2-11B-Vision-Instruct"
    [internvl14b]="OpenGVLab/InternVL3-14B-hf"
)

# Which experiments to run
EXPS="${@:-A B C D}"

mkdir -p "$ROBUSTNESS"

hdr() { echo; echo "═════════════════════════════════════════════════════════"; echo "  $1"; echo "═════════════════════════════════════════════════════════"; }

# ─────────────────────────────────────────────────────────────────
#  A. NOISE LEVEL SWEEP
# ─────────────────────────────────────────────────────────────────
if [[ " $EXPS " =~ " A " ]]; then
    hdr "A. Noise level sweep"
    for mkey in $MODELS; do
        model="${MODEL_IDS[$mkey]}"
        for std in 2.0 4.0; do
            out="${ROBUSTNESS}/noise/${mkey}_std${std}/"
            echo "  → ${mkey}  noise=${std}σ  →  $out"
            python experiments/run_patching.py \
                --model "$model" \
                --benchmark pope --split adversarial \
                --n-samples "$N_SAMPLES" \
                --pope-path "$POPE_PATH" --image-dir "$IMAGE_DIR" \
                --granularity component \
                --corrupt-mode noise --noise-std "$std" \
                --output "$out"
        done
    done

    # Then compare circuits: post-hoc
    echo "  → Comparing circuits across noise levels..."
    for mkey in $MODELS; do
        out_dirs=""
        for std in 2.0 3.0 4.0; do
            d="${ROBUSTNESS}/noise/${mkey}_std${std}/"
            [[ "$std" == "3.0" ]] && d="${RESULTS}/${mkey}_component/"  # baseline
            [[ -d "$d" ]] && out_dirs+=" $d"
        done
        if [[ -n "$out_dirs" ]]; then
            python experiments/run_patching.py \
                --compare $out_dirs \
                --output "${ROBUSTNESS}/noise/compare_${mkey}/"
        fi
    done
fi

# ─────────────────────────────────────────────────────────────────
#  B. CORRUPTION TYPE SWEEP  (noise, zero, mean)
# ─────────────────────────────────────────────────────────────────
if [[ " $EXPS " =~ " B " ]]; then
    hdr "B. Corruption type sweep"
    for mkey in $MODELS; do
        model="${MODEL_IDS[$mkey]}"
        for mode in zero mean; do
            out="${ROBUSTNESS}/corrupt/${mkey}_${mode}/"
            echo "  → ${mkey}  mode=${mode}  →  $out"
            python experiments/run_patching.py \
                --model "$model" \
                --benchmark pope --split adversarial \
                --n-samples "$N_SAMPLES" \
                --pope-path "$POPE_PATH" --image-dir "$IMAGE_DIR" \
                --granularity component \
                --corrupt-mode "$mode" \
                --output "$out"
        done
    done
fi

# ─────────────────────────────────────────────────────────────────
#  C. SEED VARIANCE  (3 seeds)
# ─────────────────────────────────────────────────────────────────
if [[ " $EXPS " =~ " C " ]]; then
    hdr "C. Seed variance (requires --seed CLI patch, see header)"
    for mkey in $MODELS; do
        model="${MODEL_IDS[$mkey]}"
        for seed in 0 1 123; do
            out="${ROBUSTNESS}/seed/${mkey}_seed${seed}/"
            echo "  → ${mkey}  seed=${seed}  →  $out"
            python experiments/run_patching.py \
                --model "$model" \
                --benchmark pope --split adversarial \
                --n-samples "$N_SAMPLES" \
                --pope-path "$POPE_PATH" --image-dir "$IMAGE_DIR" \
                --granularity component \
                --corrupt-mode noise --noise-std 3.0 \
                --seed "$seed" \
                --output "$out"
        done
    done
fi

# ─────────────────────────────────────────────────────────────────
#  D. TOP-K PER-SAMPLE  (for bootstrap; LLaVA is the only model that
#     uses top-k selection in the paper)
# ─────────────────────────────────────────────────────────────────
if [[ " $EXPS " =~ " D " ]]; then
    hdr "D. Top-k per-sample evaluation (LLaVA)"
    circuit="${RESULTS}/llava_component/circuit.json"
    if [[ ! -f "$circuit" ]]; then
        echo "  ⚠ Missing $circuit — run Phase 1 for LLaVA first, skipping D"
    else
        python experiments/eval_topk_per_sample.py \
            --model "${MODEL_IDS[llava]}" \
            --circuit "$circuit" \
            --pope-path "$POPE_PATH" --image-dir "$IMAGE_DIR" \
            --split adversarial \
            --n-samples "$N_INTERVENTION" \
            --k-values 3 5 8 10 15 20 30 \
            --scale 0.5 \
            --output "${ROBUSTNESS}/topk_per_sample_llava/"
    fi
fi

# ─────────────────────────────────────────────────────────────────
#  Post-hoc analyses (fast, no GPU). Run these last.
# ─────────────────────────────────────────────────────────────────
hdr "POST-HOC analyses"

echo "  → d-threshold sweep (all models)"
python experiments/sweep_d_threshold.py \
    --results-dir "$RESULTS" \
    --thresholds 0.2 0.25 0.3 0.35 0.4 \
    --output "${ROBUSTNESS}/d_threshold/"

echo "  → TE-filter sweep (all models)"
python experiments/sweep_te_filter.py \
    --results-dir "$RESULTS" \
    --te-thresholds 0.0 0.05 0.1 0.15 0.2 \
    --output "${ROBUSTNESS}/te_filter/"

if [[ -d "${ROBUSTNESS}/topk_per_sample_llava/" ]]; then
    echo "  → Bootstrap top-k (LLaVA)"
    python experiments/bootstrap_topk.py \
        --results-dir "${ROBUSTNESS}/topk_per_sample_llava/" \
        --n-bootstrap 1000 \
        --n-selection 100 \
        --paper-k 10
fi

echo
echo "═════════════════════════════════════════════════════════"
echo "  All robustness experiments complete."
echo "  Summarize results from: $ROBUSTNESS"
echo "═════════════════════════════════════════════════════════"
