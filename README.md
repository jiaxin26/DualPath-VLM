# Dual-Pathway Circuits of Object Hallucination in Vision-Language Models

Code for the paper:

> **Dual-Pathway Circuits of Object Hallucination in Vision-Language Models**

We use activation patching and a new diagnostic, **Conditional Pathway Analysis (CPA)**, to identify the components inside five VLMs that drive visual hallucination. We find a shared *dual-pathway* organization—a visual grounding pathway and a hallucination pathway—and use it to design a training-free intervention that reduces hallucination by 40–76% on POPE with ≤2 pp accuracy cost.

---

## Highlights

- **Conditional Pathway Analysis (CPA)** — a diagnostic that compares the joint causal effect of restoring all components in a pathway with the sum of their individual effects, decomposing the joint effect into individual contributions plus an interaction term that reveals redundancy vs. synergy.
- **Universal dual-pathway organization** — found across 5 VLMs spanning 3 LLM backbone families and two scales (7B–14B).
- **Targeted intervention** — 40–76% relative hallucination reduction on POPE-adversarial at ≤2 pp accuracy cost, with no retraining or learned probes.
- **Cross-benchmark generalization** — partial transfer to AMBER's relational hallucination (36–47% reduction in three models), revealing that distinct hallucination types engage partially distinct circuits.

---

## Repository structure

```
├── patching/                       # Core patching framework + model adapters
│   ├── core.py                     # Base patcher, CPA, statistics (BH correction)
│   ├── qwen_patcher.py             # Qwen3-VL adapter
│   ├── llava_patcher.py            # LLaVA-v1.6 adapter
│   ├── internvl_patcher.py         # InternVL3 adapter
│   └── llama_patcher.py            # Llama-3.2-Vision adapter
├── analysis/
│   └── circuits.py                 # Circuit extraction & cross-architecture comparison
├── experiments/
│   ├── run_patching.py             # Phase 1, 2: activation patching (component / head)
│   ├── run_pathway_patching.py     # Phase 3 : CPA
│   ├── run_logit_lens.py           # Phase 4 : logit lens
│   ├── reanalyze_crossarch.py      # Phase 5 : cross-architecture comparison
│   ├── run_intervention.py         # Phases 6, 6d, 6e, 7, 8 (uniform scaling, controls,
│   │                               #                          top-k, cross-benchmark)
│   ├── run_intervention_v2.py      # Phase 6b: mean-difference (directional) baseline
│   ├── run_iti_baseline.py         # Phase 6c: probe-based ITI baseline
│   ├── benchmarks.py               # POPE / AMBER loaders
│   └── summarize_multi_benchmark.py
├── plot_dual_pathway.py            # Figure 2: dual-pathway across 5 models
├── plot_cpa_figure.py              # Figure 3: CPA interaction
├── plot_pareto.py                  # Figure 4: intervention Pareto frontier
├── plot_cross_benchmark.py         # Figure 5: cross-benchmark heatmap
├── reproduce.sh                    # End-to-end reproduction pipeline
├── summarize_all.py                # Aggregate all results into a summary
└── data/                           # Datasets (downloaded separately, see below)
```

> **Figure 1** in the paper is a conceptual schematic of the activation-patching pipeline and is not generated from the experimental results; the four plot scripts above produce Figures 2–5.

---

## Setup

### 1. Install dependencies

After extracting the supplementary archive, enter the `code/` directory:

```bash
cd code/

# Create environment
conda create -n hallu python=3.10 -y
conda activate hallu

# Install requirements
pip install -r requirements.txt
```

You will also need to log in to Hugging Face to download gated models like `meta-llama/Llama-3.2-11B-Vision-Instruct`:

```bash
huggingface-cli login
```

### 2. Download data

We provide a one-line download script. **All datasets are pulled from their official sources** — we do not redistribute them.

```bash
bash reproduce.sh --phase 0
```

This downloads:

| Dataset | Source | Size | Used for |
|---------|--------|-----:|----------|
| POPE annotations | [AoiDragon/POPE](https://github.com/AoiDragon/POPE) | <1 MB | Patching, intervention |
| COCO val2014 images | [COCO](https://cocodataset.org) | ~6 GB | Image inputs for POPE |
| COCO annotations | [COCO](https://cocodataset.org) | ~250 MB | (optional) CHAIR eval |
| AMBER annotations | [junyangwang0410/AMBER](https://github.com/junyangwang0410/AMBER) | <5 MB | Cross-benchmark eval |
| AMBER images | [Google Drive](https://drive.google.com/file/d/1MaCHgtupcZUjf007anNl4_MV0o4DjXvl/view) | ~1.5 GB | Cross-benchmark eval |

If you prefer to download manually, place files like this:

```
data/
├── pope/
│   ├── coco_pope_adversarial.json
│   ├── coco_pope_popular.json
│   └── coco_pope_random.json
├── coco/
│   └── val2014/                    # ~40k .jpg images
└── amber/
    ├── data/query/query_discriminative.json
    └── image/                      # ~1k AMBER_*.jpg images
```

### 3. Hardware

All experiments were run on **NVIDIA H200 144GB** GPUs. A single A100 80GB is also sufficient for all models.

---

## Quick start

Reproduce the entire paper end-to-end:

```bash
bash reproduce.sh
```

This runs Phases 0–9 (~80–120 GPU-hours total). To run a single phase or a single model:

```bash
# Run only Phase 1 (activation patching)
bash reproduce.sh --phase 1

# Run only one model end-to-end
bash reproduce.sh --model qwen

# Single phase + single model
bash reproduce.sh --phase 6 --model llama

# Sub-phases of Phase 6 (interventions and baselines)
bash reproduce.sh --phase 6b              # mean-difference (directional) baseline
bash reproduce.sh --phase 6c              # ITI baseline (Llama only by default;
                                          # set ITI_ALL_MODELS=1 to run on all 5)
bash reproduce.sh --phase 6d              # grounding-pathway specificity control
bash reproduce.sh --phase 6e              # layer/type-matched random control

# Print commands without executing (sanity check)
bash reproduce.sh --dry-run
```

After everything finishes, summarize all results:

```bash
python summarize_all.py
```

---

## License

MIT License (see `LICENSE`). POPE/COCO/AMBER datasets retain their original licenses (Apache 2.0 / CC BY 4.0 / Apache 2.0); the download script fetches them from their official sources.

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{liu2026dualpathwaycircuitsobjecthallucination,
      title={Dual-Pathway Circuits of Object Hallucination in Vision-Language Models}, 
      author={Jiaxin Liu and Ding Zhong and Yue Wang and Zhidong Yang and Zhaolu Kang and Guangyuan Dong and Qishi Zhan and Pengcheng Fang and Aofan Liu},
      year={2026},
      eprint={2605.13156},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.13156}, 
}
```