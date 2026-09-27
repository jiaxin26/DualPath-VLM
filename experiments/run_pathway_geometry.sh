#!/bin/bash
# run_pathway_geometry.sh
#
# One-shot driver for the multi-direction follow-up experiment.
# Collects raw pathway activations on each model, then runs the
# CPU-side SVD/PR analysis. Set ONLY_MODELS to subset.
#
# Quick mode (~5 h on H200, 3 representative models):
#   ONLY_MODELS="llama llava qwen" bash experiments/run_pathway_geometry.sh
#
# Full mode (~12 h on H200):
#   bash experiments/run_pathway_geometry.sh

set -euo pipefail

POPE_PATH="${POPE_PATH:-data/pope/}"
IMAGE_DIR="${IMAGE_DIR:-data/coco/val2014/}"
RESULTS="${RESULTS:-results}"
N_SAMPLES="${N_SAMPLES:-1000}"
ONLY_MODELS="${ONLY_MODELS:-qwen llava llama internvl8b internvl14b}"

declare -A MODELS=(
    [qwen]="Qwen/Qwen3-VL-8B-Instruct"
    [llava]="llava-hf/llava-v1.6-mistral-7b-hf"
    [internvl8b]="OpenGVLab/InternVL3-8B-hf"
    [llama]="meta-llama/Llama-3.2-11B-Vision-Instruct"
    [internvl14b]="OpenGVLab/InternVL3-14B-hf"
)

declare -A CIRCUITS=(
    [qwen]="${RESULTS}/qwen_component/circuit.json"
    [llava]="${RESULTS}/llava_component/circuit.json"
    [internvl8b]="${RESULTS}/internvl8b_component/circuit.json"
    [llama]="${RESULTS}/llama_component/circuit.json"
    [internvl14b]="${RESULTS}/internvl14b_component/circuit.json"
)

# ── Phase 1: GPU collection per model ─────────────────────────
COLLECTED=()
for key in $ONLY_MODELS; do
    model="${MODELS[$key]}"
    circuit="${CIRCUITS[$key]}"
    out_dir="${RESULTS}/geometry_${key}"

    if [[ ! -f "$circuit" ]]; then
        echo "[skip] $key: circuit not found at $circuit"
        echo "       Run reproduce.sh phase 1+2 for $key first."
        continue
    fi

    if [[ -f "${out_dir}/geometry_acts.pt" ]]; then
        echo "[exists] $key: ${out_dir}/geometry_acts.pt — skipping collection"
    else
        echo "================================================"
        echo "[collect] $key  ($model)"
        echo "================================================"
        PYTHONPATH=. python experiments/collect_pathway_geometry.py \
            --model "$model" \
            --circuit "$circuit" \
            --pope-path "$POPE_PATH" --image-dir "$IMAGE_DIR" \
            --n-samples "$N_SAMPLES" \
            --output "$out_dir"
    fi
    COLLECTED+=("${out_dir}/geometry_acts.pt")
done

# ── Phase 2: CPU analysis over all collected models ──────────
if [[ ${#COLLECTED[@]} -eq 0 ]]; then
    echo "Nothing collected; exiting."
    exit 1
fi

echo "================================================"
echo "[analyze] aggregate over ${#COLLECTED[@]} models"
echo "================================================"
python experiments/analyze_pathway_geometry.py \
    --inputs "${COLLECTED[@]}" \
    --output "${RESULTS}/geometry_summary/"

echo
echo "Done. See ${RESULTS}/geometry_summary/per_pathway.csv for headlines."
