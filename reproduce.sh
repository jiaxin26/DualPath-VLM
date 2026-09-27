#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  reproduce.sh — Full Reproduction Pipeline
# ═══════════════════════════════════════════════════════════════════
#
#  Phases:
#    0  Download data (POPE, COCO, AMBER)
#    1  Component-level activation patching        → Table 1, 2
#    2  Head-level patching (optional)             → concentration analysis
#    3  Conditional Pathway Analysis (CPA)           → Table 3
#    4  Logit lens validation                      → Table 4
#    5  Cross-architecture comparison              → Table 5
#    6  Intervention on POPE-adversarial           → Table 6
#    7  LLaVA top-k ablation                       → Table 7
#    8  Cross-benchmark (POPE splits + AMBER)      → Table 8
#    9  Summarize all results
#
#  Usage:
#    bash reproduce.sh                    # everything
#    bash reproduce.sh --phase 1          # single phase
#    bash reproduce.sh --model qwen       # single model
#    bash reproduce.sh --phase 6 --model llama  # single phase + model
#    bash reproduce.sh --phase 0          # download data only
#    bash reproduce.sh --dry-run          # show what would run
#
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.

# ── Configuration ────────────────────────────────────────────────

POPE_PATH="${POPE_PATH:-data/pope/}"
IMAGE_DIR="${IMAGE_DIR:-data/coco/val2014/}"
AMBER_PATH="${AMBER_PATH:-data/amber/}"
AMBER_IMAGE_DIR="${AMBER_IMAGE_DIR:-data/amber/image/}"
RESULTS="${RESULTS:-results/}"

# Sample counts (paper defaults)
N_PATCH=1000         # Phase 1: patching
N_CPA=400            # Phase 3: CPA
N_LENS=200           # Phase 4: logit lens
N_INTERVENTION=500   # Phase 6-8: intervention eval

# ── Models ───────────────────────────────────────────────────────
# 5 VLMs: 3 backbone families, 4 fusion strategies, 2 scales

declare -A MODELS=(
    [qwen]="Qwen/Qwen3-VL-8B-Instruct"
    [llava]="llava-hf/llava-v1.6-mistral-7b-hf"
    [internvl8b]="OpenGVLab/InternVL3-8B-hf"
    [llama]="meta-llama/Llama-3.2-11B-Vision-Instruct"
    [internvl14b]="OpenGVLab/InternVL3-14B-hf"
)

# Ordered for consistent iteration (bash associative arrays are unordered)
MODEL_ORDER=(qwen llava internvl8b llama internvl14b)

# Optimal scale factors per model (from Table 6)
declare -A OPTIMAL_SCALES=(
    [qwen]="0.25"
    [llava]="0.5"
    [internvl8b]="0.5"
    [llama]="0.5"
    [internvl14b]="0.5"
)

# LLaVA top-k setting
LLAVA_TOPK=10

# ── Argument parsing ─────────────────────────────────────────────

RUN_PHASE=""
ONLY_MODEL=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)    RUN_PHASE="$2"; shift 2 ;;
        --model)    ONLY_MODEL="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=true; shift ;;
        -h|--help)
            echo "Usage: bash reproduce.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --phase N     Run only phase N (0, 1, ..., 6, 6b, 6c, 7, 8, 9)"
            echo "  --model KEY   Run only one model (qwen|llava|internvl8b|llama|internvl14b)"
            echo "  --dry-run     Show commands without executing"
            echo "  -h, --help    Show this help"
            echo ""
            echo "Environment variables:"
            echo "  ITI_ALL_MODELS=1  Phase 6c runs ITI on all 5 models"
            echo "                    (default: Llama only, ~1.5h)"
            echo ""
            echo "Phases:"
            echo "  0   Download data"
            echo "  1   Activation patching (Tables 1-2)       ~4-18h/model"
            echo "  2   Head-level patching (optional)          ~12-24h/model"
            echo "  3   Conditional Pathway Analysis (Table 3)    ~2-3h/model"
            echo "  4   Logit lens validation (Table 4)         ~1h/model"
            echo "  5   Cross-architecture comparison (Table 5) <1min"
            echo "  6   Intervention on POPE-adv (Table 6)      ~2h/model"
            echo "  6b  Directional suppression baseline        ~2h/model"
            echo "  6c  Literal ITI baseline (Llama only)       ~1.5h"
            echo "  6d  Grounding-pathway suppression (specificity)  ~2h/model"
            echo "  7   LLaVA top-k ablation (Table 7)          ~3h"
            echo "  8   Cross-benchmark POPE+AMBER (Table 8)    ~3h/model"
            echo "  9   Summarize all results"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1 (use --help for usage)"
            exit 1
            ;;
    esac
done

# ── Helpers ──────────────────────────────────────────────────────

should_run() { [[ -z "$RUN_PHASE" || "$RUN_PHASE" == "$1" ]]; }
should_run_model() { [[ -z "$ONLY_MODEL" || "$ONLY_MODEL" == "$1" ]]; }

circuit_path() {
    # Returns the circuit.json path for a model key
    local key="$1"
    echo "${RESULTS}/${key}_component/circuit.json"
}

run_cmd() {
    # Execute or print command depending on --dry-run
    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY-RUN] $*"
    else
        "$@"
    fi
}

phase_header() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  PHASE $1"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "═══════════════════════════════════════════════════════════════"
}

check_circuit() {
    local key="$1"
    local cpath
    cpath=$(circuit_path "$key")
    if [[ ! -f "$cpath" ]]; then
        echo "  ✗ ${key}: circuit.json not found at ${cpath}"
        echo "    Run Phase 1 first: bash reproduce.sh --phase 1 --model ${key}"
        return 1
    fi
    return 0
}


# ═══════════════════════════════════════════════════════════════════
#  PHASE 0: Download Data
# ═══════════════════════════════════════════════════════════════════

if should_run 0; then
    phase_header "0: Download data"

    echo "  Checking data directories..."

    # POPE
    if [[ ! -f "${POPE_PATH}/coco_pope_adversarial.json" ]]; then
        echo "  → Downloading POPE..."
        mkdir -p data/pope
        run_cmd git clone https://github.com/AoiDragon/POPE.git data/pope_repo
        run_cmd cp data/pope_repo/output/coco/*.json data/pope/
        echo "  ✓ POPE ready"
    else
        echo "  ✓ POPE already exists"
    fi

    # COCO val2014
    if [[ ! -d "${IMAGE_DIR}" ]]; then
        echo "  → Downloading COCO val2014 images..."
        mkdir -p data/coco
        run_cmd wget -q http://images.cocodataset.org/zips/val2014.zip -O data/coco/val2014.zip
        run_cmd unzip -q data/coco/val2014.zip -d data/coco/
        run_cmd rm data/coco/val2014.zip
        echo "  ✓ COCO val2014 ready"
    else
        echo "  ✓ COCO val2014 already exists"
    fi

    # COCO annotations (for CHAIR evaluation)
    if [[ ! -f "data/coco/annotations/instances_val2014.json" ]]; then
        echo "  → Downloading COCO annotations..."
        mkdir -p data/coco
        run_cmd wget -q http://images.cocodataset.org/annotations/annotations_trainval2014.zip \
            -O data/coco/annotations.zip
        run_cmd unzip -q data/coco/annotations.zip -d data/coco/
        run_cmd rm data/coco/annotations.zip
        echo "  ✓ COCO annotations ready"
    else
        echo "  ✓ COCO annotations already exist"
    fi

    # AMBER
    if [[ ! -d "${AMBER_PATH}" ]]; then
        echo "  → Downloading AMBER annotations..."
        run_cmd git clone https://github.com/junyangwang0410/AMBER.git "${AMBER_PATH}"
        echo "  ✓ AMBER annotations ready"
    else
        echo "  ✓ AMBER annotations already exist"
    fi

    if [[ ! -d "${AMBER_IMAGE_DIR}" ]]; then
        echo "  → Downloading AMBER images (requires gdown)..."
        pip install -q gdown 2>/dev/null || true
        run_cmd gdown 1MaCHgtupcZUjf007anNl4_MV0o4DjXvl -O "${AMBER_PATH}/amber_images.zip"
        run_cmd unzip -q "${AMBER_PATH}/amber_images.zip" -d "${AMBER_PATH}/"
        echo "  ✓ AMBER images ready"
    else
        echo "  ✓ AMBER images already exist"
    fi

    echo ""
    echo "  Data summary:"
    echo "    POPE:  $(ls ${POPE_PATH}/*.json 2>/dev/null | wc -l) files"
    echo "    COCO:  $(ls ${IMAGE_DIR}/*.jpg 2>/dev/null | wc -l) images"
    echo "    AMBER: $(ls ${AMBER_IMAGE_DIR}/*.jpg 2>/dev/null | wc -l) images"
fi


# ═══════════════════════════════════════════════════════════════════
#  PHASE 1: Component-level activation patching (Tables 1, 2)
# ═══════════════════════════════════════════════════════════════════

if should_run 1; then
    phase_header "1: Component-level activation patching → Tables 1, 2"

    for key in "${MODEL_ORDER[@]}"; do
        should_run_model "$key" || continue
        model="${MODELS[$key]}"
        out="${RESULTS}/${key}_component/"
        echo "  → ${key}: ${model}"

        run_cmd python experiments/run_patching.py \
            --model "$model" \
            --pope-path "$POPE_PATH" --image-dir "$IMAGE_DIR" \
            --split adversarial --n-samples $N_PATCH \
            --granularity component \
            --output "$out"
    done
fi


# ═══════════════════════════════════════════════════════════════════
#  PHASE 2: Head-level patching (optional, for concentration analysis)
# ═══════════════════════════════════════════════════════════════════

if should_run 2; then
    phase_header "2: Head-level patching (optional)"

    for key in "${MODEL_ORDER[@]}"; do
        should_run_model "$key" || continue
        model="${MODELS[$key]}"
        out="${RESULTS}/${key}_head/"
        echo "  → ${key}: ${model}"

        run_cmd python experiments/run_patching.py \
            --model "$model" \
            --pope-path "$POPE_PATH" --image-dir "$IMAGE_DIR" \
            --split adversarial --n-samples $N_PATCH \
            --granularity head \
            --output "$out"
    done
fi


# ═══════════════════════════════════════════════════════════════════
#  PHASE 3: Conditional Pathway Analysis (Table 3)
# ═══════════════════════════════════════════════════════════════════

if should_run 3; then
    phase_header "3: Conditional Pathway Analysis → Table 3"

    for key in "${MODEL_ORDER[@]}"; do
        should_run_model "$key" || continue
        check_circuit "$key" || continue
        model="${MODELS[$key]}"
        circuit=$(circuit_path "$key")
        out="${RESULTS}/${key}_pathway/"
        echo "  → ${key}: ${model}"

        run_cmd python experiments/run_pathway_patching.py \
            --model "$model" \
            --circuit "$circuit" \
            --pope-path "$POPE_PATH" --image-dir "$IMAGE_DIR" \
            --n-samples $N_CPA \
            --output "$out"
    done
fi


# ═══════════════════════════════════════════════════════════════════
#  PHASE 4: Logit lens validation (Table 4)
# ═══════════════════════════════════════════════════════════════════

if should_run 4; then
    phase_header "4: Logit lens validation → Table 4"

    for key in "${MODEL_ORDER[@]}"; do
        should_run_model "$key" || continue
        check_circuit "$key" || continue
        model="${MODELS[$key]}"
        circuit=$(circuit_path "$key")
        out="${RESULTS}/${key}_logit_lens/"
        echo "  → ${key}: ${model}"

        run_cmd python experiments/run_logit_lens.py \
            --model "$model" \
            --circuit "$circuit" \
            --pope-path "$POPE_PATH" --image-dir "$IMAGE_DIR" \
            --n-samples $N_LENS \
            --output "$out"
    done
fi


# ═══════════════════════════════════════════════════════════════════
#  PHASE 5: Cross-architecture comparison (Table 5)
# ═══════════════════════════════════════════════════════════════════

if should_run 5; then
    phase_header "5: Cross-architecture comparison → Table 5"

    circuits=()
    for key in "${MODEL_ORDER[@]}"; do
        c=$(circuit_path "$key")
        # Also check head-level circuits
        h="${RESULTS}/${key}_head/circuit.json"
        if [[ -f "$h" ]]; then
            circuits+=("$h")
        elif [[ -f "$c" ]]; then
            circuits+=("$c")
        fi
    done

    echo "  Found ${#circuits[@]} circuit files"

    if [[ ${#circuits[@]} -ge 2 ]]; then
        run_cmd python experiments/reanalyze_crossarch.py \
            --circuits "${circuits[@]}" \
            --output "${RESULTS}/crossarch/"
    else
        echo "  ✗ Need at least 2 circuit files. Run Phase 1 first."
    fi
fi


# ═══════════════════════════════════════════════════════════════════
#  PHASE 6: Intervention on POPE-adversarial (Table 6)
# ═══════════════════════════════════════════════════════════════════

if should_run 6; then
    phase_header "6: Intervention on POPE-adversarial → Table 6"

    for key in "${MODEL_ORDER[@]}"; do
        should_run_model "$key" || continue
        check_circuit "$key" || continue
        model="${MODELS[$key]}"
        circuit=$(circuit_path "$key")
        scale="${OPTIMAL_SCALES[$key]}"
        out="${RESULTS}/${key}_intervention/"
        echo "  → ${key}: scale=${scale}"

        if [[ "$key" == "llava" ]]; then
            # LLaVA uses top-k ablation
            run_cmd python experiments/run_intervention.py \
                --model "$model" \
                --circuit "$circuit" \
                --pope-path "$POPE_PATH" --image-dir "$IMAGE_DIR" \
                --split adversarial --n-samples $N_INTERVENTION \
                --scale "$scale" \
                --topk-ablation $LLAVA_TOPK \
                --output "$out"
        else
            run_cmd python experiments/run_intervention.py \
                --model "$model" \
                --circuit "$circuit" \
                --pope-path "$POPE_PATH" --image-dir "$IMAGE_DIR" \
                --split adversarial --n-samples $N_INTERVENTION \
                --scales 0.0 0.25 0.5 0.75 \
                --output "$out"
        fi
    done
fi


# ═══════════════════════════════════════════════════════════════════
#  PHASE 6b: Directional suppression baseline (Section 5.5)
#           Our mean-difference directional projection — the ITI-family
#           baseline reported as "not effective" in the paper.
# ═══════════════════════════════════════════════════════════════════

if should_run 6b; then
    phase_header "6b: Directional suppression baseline → Section 5.5"

    for key in "${MODEL_ORDER[@]}"; do
        should_run_model "$key" || continue
        check_circuit "$key" || continue
        model="${MODELS[$key]}"
        circuit=$(circuit_path "$key")
        out="${RESULTS}/${key}_directional/"
        echo "  → ${key}: directional suppression"

        run_cmd python experiments/run_intervention_v2.py \
            --model "$model" \
            --circuit "$circuit" \
            --mode full \
            --pope-path "$POPE_PATH" --image-dir "$IMAGE_DIR" \
            --split adversarial \
            --n-samples $N_INTERVENTION --n-val 100 \
            --skip-optimal-scaling \
            --output "$out"
    done
fi


# ═══════════════════════════════════════════════════════════════════
#  PHASE 6c: Literal ITI baseline (Li et al. 2024)
#           Probe-based head selection + σ-scaled additive shift.
#           Stronger than 6b; used as a sanity check that more
#           sophisticated single-direction steering also fails.
#
#  By default, runs only on Llama-3.2-Vision (most concentrated
#  hallucination pathway = friendliest target for ITI).  To run on
#  all five models, set ITI_ALL_MODELS=1 before invoking the script.
# ═══════════════════════════════════════════════════════════════════

if should_run 6c; then
    phase_header "6c: Literal ITI baseline → Section 5.5 / Appendix"

    # Default: Llama only.  Override with ITI_ALL_MODELS=1.
    if [[ "${ITI_ALL_MODELS:-0}" == "1" ]]; then
        ITI_MODELS=("${MODEL_ORDER[@]}")
    else
        ITI_MODELS=(llama)
        echo "  (running ITI on Llama only; set ITI_ALL_MODELS=1 to run all)"
    fi

    for key in "${ITI_MODELS[@]}"; do
        should_run_model "$key" || continue
        model="${MODELS[$key]}"
        out="${RESULTS}/${key}_iti/"
        echo "  → ${key}: ITI probe training + evaluation"

        run_cmd python experiments/run_iti_baseline.py \
            --model "$model" \
            --mode full \
            --pope-path "$POPE_PATH" --image-dir "$IMAGE_DIR" \
            --split adversarial \
            --n-extract 1000 \
            --n-samples $N_INTERVENTION --n-val 100 \
            --k-values 12 24 48 96 \
            --alpha-values -20 -15 -10 -5 5 10 15 20 \
            --disjoint-splits \
            --output "$out"
    done
fi

# ═══════════════════════════════════════════════════════════════════
#  PHASE 6d: Grounding-pathway suppression (specificity control)
#           Suppresses d>0 components instead of d<0. Predicts:
#           accuracy drops sharply, halluc rate does NOT decrease.
#           Together with Phase 6, establishes double dissociation.
# ═══════════════════════════════════════════════════════════════════

if should_run 6d; then
    phase_header "6d: Grounding-pathway suppression (specificity control)"

    for key in "${MODEL_ORDER[@]}"; do
        should_run_model "$key" || continue
        check_circuit "$key" || continue
        model="${MODELS[$key]}"
        circuit=$(circuit_path "$key")
        out="${RESULTS}/${key}_grounding_suppress/"
        echo "  → ${key}: grounding pathway (d>0) suppression"

        run_cmd python experiments/run_intervention.py \
            --model "$model" \
            --circuit "$circuit" \
            --pathway visual \
            --pope-path "$POPE_PATH" --image-dir "$IMAGE_DIR" \
            --split adversarial --n-samples $N_INTERVENTION \
            --scales 0.0 0.25 0.5 0.75 \
            --output "$out"
    done
fi


# ═══════════════════════════════════════════════════════════════════
#  PHASE 6e: Layer/type-matched random control (specificity control)
#           For each circuit component (L, ctype), samples a replacement
#           (L', ctype) from layers L' != L not in the circuit. Tests
#           "is it specifically these components, or any same-type
#           components at different depths?"
# ═══════════════════════════════════════════════════════════════════
 
if should_run 6e; then
    phase_header "6e: Layer/type-matched random control"
 
    for key in "${MODEL_ORDER[@]}"; do
        should_run_model "$key" || continue
        check_circuit "$key" || continue
        model="${MODELS[$key]}"
        circuit=$(circuit_path "$key")
        scale="${OPTIMAL_SCALES[$key]}"
        out="${RESULTS}/${key}_typematched_ctrl/"
        echo "  → ${key}: type-matched control at scale=${scale}"
 
        run_cmd python experiments/run_intervention.py \
            --model "$model" \
            --circuit "$circuit" \
            --pathway halluc \
            --pope-path "$POPE_PATH" --image-dir "$IMAGE_DIR" \
            --split adversarial --n-samples $N_INTERVENTION \
            --scale "$scale" \
            --random-mode type-matched \
            --n-random-seeds 5 \
            --output "$out"
    done
fi
 


# ═══════════════════════════════════════════════════════════════════
#  PHASE 7: LLaVA top-k ablation (Table 7)
# ═══════════════════════════════════════════════════════════════════

if should_run 7; then
    phase_header "7: LLaVA top-k ablation → Table 7"

    key="llava"
    if should_run_model "$key" && check_circuit "$key"; then
        model="${MODELS[$key]}"
        circuit=$(circuit_path "$key")
        out="${RESULTS}/${key}_topk/"
        echo "  → LLaVA top-k ablation: k ∈ {3, 5, 8, 10, 15, 20, 30}"

        run_cmd python experiments/run_intervention.py \
            --model "$model" \
            --circuit "$circuit" \
            --pope-path "$POPE_PATH" --image-dir "$IMAGE_DIR" \
            --split adversarial --n-samples $N_INTERVENTION \
            --scale 0.5 \
            --topk-ablation 3 5 8 10 15 20 30 \
            --output "$out"
    fi
fi


# ═══════════════════════════════════════════════════════════════════
#  PHASE 8: Cross-benchmark evaluation — POPE splits + AMBER (Table 8)
# ═══════════════════════════════════════════════════════════════════

if should_run 8; then
    phase_header "8: Cross-benchmark → Table 8"

    for key in "${MODEL_ORDER[@]}"; do
        should_run_model "$key" || continue
        check_circuit "$key" || continue
        model="${MODELS[$key]}"
        circuit=$(circuit_path "$key")
        scale="${OPTIMAL_SCALES[$key]}"

        echo ""
        echo "  ── Model: ${key} (scale=${scale}) ──"

        # 8a: POPE popular + random
        for split in popular random; do
            out="${RESULTS}/multi_benchmark/${key}/pope_${split}/"
            echo "  → POPE-${split}"

            if [[ "$key" == "llava" ]]; then
                run_cmd python experiments/run_intervention.py \
                    --model "$model" --circuit "$circuit" \
                    --pope-path "$POPE_PATH" --image-dir "$IMAGE_DIR" \
                    --split "$split" --n-samples $N_INTERVENTION \
                    --scale "$scale" --topk-ablation $LLAVA_TOPK \
                    --output "$out"
            else
                run_cmd python experiments/run_intervention.py \
                    --model "$model" --circuit "$circuit" \
                    --pope-path "$POPE_PATH" --image-dir "$IMAGE_DIR" \
                    --split "$split" --n-samples $N_INTERVENTION \
                    --scales $scale \
                    --output "$out"
            fi
        done

        # 8b: POPE adversarial (for the multi-bench table, reuse Phase 6 if exists)
        out="${RESULTS}/multi_benchmark/${key}/pope_adversarial/"
        if [[ ! -f "${out}/intervention_results.json" ]]; then
            echo "  → POPE-adversarial"
            if [[ "$key" == "llava" ]]; then
                run_cmd python experiments/run_intervention.py \
                    --model "$model" --circuit "$circuit" \
                    --pope-path "$POPE_PATH" --image-dir "$IMAGE_DIR" \
                    --split adversarial --n-samples $N_INTERVENTION \
                    --scale "$scale" --topk-ablation $LLAVA_TOPK \
                    --output "$out"
            else
                run_cmd python experiments/run_intervention.py \
                    --model "$model" --circuit "$circuit" \
                    --pope-path "$POPE_PATH" --image-dir "$IMAGE_DIR" \
                    --split adversarial --n-samples $N_INTERVENTION \
                    --scales $scale \
                    --output "$out"
            fi
        else
            echo "  ✓ POPE-adversarial already exists, skipping"
        fi

        # 8c: AMBER — all types combined
        if [[ -d "$AMBER_PATH" ]]; then
            out="${RESULTS}/multi_benchmark/${key}/amber_all/"
            echo "  → AMBER (all types)"
            if [[ "$key" == "llava" ]]; then
                run_cmd python experiments/run_intervention.py \
                    --model "$model" --circuit "$circuit" \
                    --benchmarks amber \
                    --amber-path "$AMBER_PATH" --amber-image-dir "$AMBER_IMAGE_DIR" \
                    --image-dir "$IMAGE_DIR" \
                    --n-samples $N_INTERVENTION --scale "$scale" \
                    --topk-ablation $LLAVA_TOPK \
                    --output "$out"
            else
                run_cmd python experiments/run_intervention.py \
                    --model "$model" --circuit "$circuit" \
                    --benchmarks amber \
                    --amber-path "$AMBER_PATH" --amber-image-dir "$AMBER_IMAGE_DIR" \
                    --image-dir "$IMAGE_DIR" \
                    --n-samples $N_INTERVENTION --scales $scale \
                    --output "$out"
            fi

            # 8d: AMBER — per type
            for atype in existence attribute relation; do
                out="${RESULTS}/multi_benchmark/${key}/amber_${atype}/"
                echo "  → AMBER (${atype})"
                if [[ "$key" == "llava" ]]; then
                    run_cmd python experiments/run_intervention.py \
                        --model "$model" --circuit "$circuit" \
                        --benchmarks amber \
                        --amber-path "$AMBER_PATH" --amber-image-dir "$AMBER_IMAGE_DIR" \
                        --amber-types "$atype" \
                        --image-dir "$IMAGE_DIR" \
                        --n-samples $N_INTERVENTION --scale "$scale" \
                        --topk-ablation $LLAVA_TOPK \
                        --output "$out"
                else
                    run_cmd python experiments/run_intervention.py \
                        --model "$model" --circuit "$circuit" \
                        --benchmarks amber \
                        --amber-path "$AMBER_PATH" --amber-image-dir "$AMBER_IMAGE_DIR" \
                        --amber-types "$atype" \
                        --image-dir "$IMAGE_DIR" \
                        --n-samples $N_INTERVENTION --scales $scale \
                        --output "$out"
                fi
            done
        else
            echo "  ✗ AMBER not found at ${AMBER_PATH}. Run Phase 0 first."
        fi
    done

    # Summarize multi-benchmark results
    echo ""
    echo "  Generating summary table..."
    run_cmd python experiments/summarize_multi_benchmark.py \
        --results-dir "${RESULTS}/multi_benchmark/" \
        --output "${RESULTS}/multi_benchmark/summary_table.json"
fi


# ═══════════════════════════════════════════════════════════════════
#  PHASE 9: Summarize all results
# ═══════════════════════════════════════════════════════════════════

if should_run 9; then
    phase_header "9: Summarize all results"
    run_cmd python summarize_all.py --results "${RESULTS}"
fi


# ═══════════════════════════════════════════════════════════════════
#  DONE
# ═══════════════════════════════════════════════════════════════════

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  COMPLETE — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Results in: ${RESULTS}"
echo ""
echo "  To summarize: python summarize_all.py --results ${RESULTS}"
echo "═══════════════════════════════════════════════════════════════"