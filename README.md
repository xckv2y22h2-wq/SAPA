# SAPA — Code Bundle for ACM CCS 2026 Submission

Code bundle accompanying the paper *"SAPA: Semantic-Aware Perturbation Attacks on Vision-Language Models"*. This README maps each experiment reported in the paper to the program files that produced it.

## Directory layout

This bundle contains **only the training and attacking code** required to reproduce the paper's victim-model training and adversarial-attack experiments. Result aggregation, statistical analysis, and ablation studies are not included.

```
code/
├── new_visual_prompt.py             # Adversarial training (single-GPU): TeCoA / PMG-AFT / TRADES / Joint
├── new_finetuning_ddp.py            # Multi-GPU DDP training (same methods, faster)
├── visual_prompt.py                 # Earlier baseline trainer (kept for reference)
│
├── attacks.py                       # PGD, C&W, AA, targeted_pgd, targeted_cw implementations (PGD/C&W family)
├── run_attack.py                    # Single-attack runner for PGD/C&W/targeted variants
├── run_attack_comparison.py         # Sweep orchestrator for PGD/C&W/targeted attacks
│
├── run_semantic_attack_comparison.py  # SAPA sweep orchestrator (the paper's actual SAPA pipeline)
├── run_semantic_attack_config.py      # SAPA single-attack runner — uses sapa/ classes (Phase 1-4)
│
├── adv_clip_loss.py                 # Adversarial loss formulations (used by both training and attacks)
├── utils.py                         # Common utilities (logging, seeding, dataset loaders)
│
├── sapa/                            # SAPA attack components (Phase 1–4)
│   ├── __init__.py
│   ├── semantic_wordnet_anchor.py            # Phase 1: WordNet semantic anchor selection
│   ├── semantic_clip_anchor.py               # Phase 1: CLIP-filtered anchor refinement
│   ├── semantic_feature_perturbation.py      # Phase 2: final-layer alignment
│   ├── semantic_feature_perturbation_multilayer.py  # multi-layer variant
│   ├── semantic_feature_perturbation_adaptive.py    # adaptive variant
│   ├── adversarial_text_generation.py        # Phase 3 helpers
│   └── llava_text_adaption.py                # Optional Phase 3: LLaVA test-time adaptation
│
├── config/                          # YAML experiment configurations (18 files)
│   ├── config.yaml                  # Default attack-comparison config
│   ├── pgd.yaml, cw.yaml, sapa.yaml # Attack-specific configs
│   ├── full_comparison.yaml         # Comprehensive matrix
│   ├── targeted_baseline.yaml       # Targeted PGD/C&W baseline configs
│   ├── compare_finalonly.yaml       # Final-layer-only comparison
│   ├── compare_multilayer.yaml      # Multi-layer comparison
│   └── ...
│
├── replace/                         # Model loading + dataset implementations
│   ├── model_loader.py              # Unified loader for CLIP / TeCoA / FARE / TRADES / PMG-AFT / MobileCLIP
│   ├── clip.py, model.py            # Modified CLIP modules
│   └── tv_datasets/                 # 14 dataset implementations (Caltech, ImageNet, OxfordPets, …)
│
├── modified_clip/                   # OpenAI CLIP fork with compatibility wrappers
├── models/                          # Model utilities, prompters, weight averaging (used by training)
├── data_utils/                      # Dataset adaptive semantic targeting helpers (used by attacks)
│
├── requirements.txt                 # Python dependencies
├── requirement_final.txt            # Final-locked dependencies
└── imagenet_classes_names.txt       # ImageNet-1K class label list
```

## Training the robust-CLIP victim models (prerequisite)

The paper evaluates SAPA against four adversarially trained CLIP variants (TeCoA, PMG-AFT, TRADES, plus the standard CLIP baseline; FARE is loaded from its public release). Pretrained checkpoints are too large to bundle (~1.6 GB each), so this section explains how to **train them from scratch** using the included scripts. **All experiments below depend on these checkpoints existing in `./models/`.**

### Training scripts

| File | Purpose |
|---|---|
| `new_visual_prompt.py` | Visual-prompt + adversarial fine-tuning (single-GPU). Used for TeCoA / PMG-AFT / TRADES. |
| `new_finetuning_ddp.py` | Multi-GPU DDP version of the above (faster but requires ≥2 GPUs). |
| `visual_prompt.py` | Earlier baseline trainer (kept for reference). |
| `models/swa.py` | Stochastic Weight Averaging utility used at the end of training. |

### Defaults (matching paper's evaluated checkpoints)

- Backbone: **CLIP ViT-B/32**
- Train dataset: **ImageNet-1K** training split
- Batch size: **256**
- Optimizer: **SGD**, momentum 0.9, weight decay 0
- Learning rate: **40** (visual-prompt setup; very large because we tune the additive prompt token, not weights)
- Epochs: **10**
- Adversarial inner-loop: **2-step** PGD-AT with `train_eps`/`train_stepsize` fixed
- Default ε for training: **1/255** (paper-evaluated checkpoint), with ε=2..4/255 variants used in robustness ablations

### Train each robust-CLIP variant

All commands assume `./datasets/ImageNet/` exists and you have one H100 (80 GB).

#### TeCoA (Mao et al., ICLR 2023) — text-guided contrastive AT
```bash
python new_visual_prompt.py \
    --epochs 10 --train_eps 1 \
    --vpt_method TeCoA \
    --exp_name TeCoA_ViT-B_eps1
```
Output: `./checkpoints/TeCoA_ViT-B_eps1/model_best.pth.tar` (load as `TeCoA_ViT-B_model_best.pth.tar`).

#### PMG-AFT (Wang et al., CVPR 2024) — pretrained-model-guided AT
```bash
python new_visual_prompt.py \
    --epochs 10 --train_eps 1 \
    --vpt_method PMG \
    --W_Pred_Align_Ori 1.0 --W_Pred_Align 1.0 \
    --exp_name PMG_ViT-B_eps1
```
Output: `./checkpoints/PMG_ViT-B_eps1/model_best.pth.tar` (load as `PMG_model_best.pth.tar`).

#### TRADES (Zhang et al., ICML 2019) — natural CE + β·KL(adv, ori)
```bash
python new_visual_prompt.py \
    --epochs 10 --train_eps 1 \
    --vpt_method TRADES \
    --W_Pred_Align_Ori 9.0 \
    --exp_name TRADES_ViT-B_eps1
```
Output: `./checkpoints/TRADES_ViT-B_eps1/model_best.pth.tar` (load as `TRADES_model_best.pth.tar`).

#### Joint adversarial image + text prompt tuning (paper's Adv\_Img+Adv\_Text variant)
```bash
python new_visual_prompt.py \
    --epochs 10 --train_eps 1 \
    --vpt_method ImgText_PGD \
    --W_Pred_Align_Ori 1.0 \
    --adv_prompt_gen True \
    --text_perb_stepsize 1e-4 \
    --exp_name ImgText_PGD_ViT-B_eps1
```

### Multi-epsilon variants (robustness ablations)

For ε ∈ {2, 3, 4} / 255, change `--train_eps`, `--train_stepsize`, and `--test_eps` accordingly:
```bash
# TeCoA at ε=2/255
python new_visual_prompt.py --epochs 10 --train_eps 2 --train_stepsize 2 --test_eps 2 \
    --vpt_method TeCoA --exp_name TeCoA_ViT-B_eps2

# PMG at ε=4/255
python new_visual_prompt.py --epochs 10 --train_eps 4 --train_stepsize 4 --test_eps 4 \
    --vpt_method PMG --W_Pred_Align_Ori 1.0 --W_Pred_Align 1.0 --exp_name PMG_ViT-B_eps4
```

### FARE (Schlarmann et al., ICML 2024)
FARE is **not** trained by this codebase. It uses an unsupervised feature-alignment objective and the authors release pretrained checkpoints publicly. The paper's ViT-B/32 evaluation uses the FARE4 (ε=4/255) ViT-B/32 checkpoint:

| Variant | HuggingFace repo |
|---|---|
| **FARE4 ViT-B/32** (paper's checkpoint) | https://huggingface.co/chs20/FARE4-ViT-B-32-laion2B-s34B-b79K |


To use the FARE4 ViT-B/32 checkpoint with this codebase:
```bash

git lfs install
git clone https://huggingface.co/chs20/FARE4-ViT-B-32-laion2B-s34B-b79K
mv FARE4-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.bin ./models/FARE_ViT-B_model_best.pth.tar

```

After placement, `replace/model_loader.py::load_FARE` will pick it up automatically.

### After training: verify the checkpoint
```bash
python run_attack.py \
    --model_path ./checkpoints/TeCoA_ViT-B_eps1/model_best.pth.tar \
    --model_type TeCoA --arch ViT-B/32 \
    --attack pgd --epsilon 0.0157 \
    --dataset OxfordPets --data_path ./datasets/ \
    --output_dir ./eval_results
```
Compare clean and adversarial accuracy against the paper's reported numbers (Table 14 in appendix) to confirm successful training.

### Training compute & time

| Model | Train time on 1×H100 | Train time on 2×H100 (DDP) |
|---|---|---|
| TeCoA, ε=1/255 | ~12-16 hours | ~7-9 hours |
| PMG-AFT, ε=1/255 | ~14-18 hours | ~8-10 hours |
| TRADES, ε=1/255 | ~12-16 hours | ~7-9 hours |
| Each multi-ε variant | similar (the inner-loop step count is fixed) | similar |

Single-GPU is sufficient with 80 GB VRAM at batch size 256. For DDP, use:
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
    new_finetuning_ddp.py --epochs 10 --train_eps 1 --vpt_method TeCoA ...
```

### Evaluation-only mode

Once a checkpoint is trained or downloaded, evaluate clean and PGD accuracy with:
```bash
python new_visual_prompt.py --evaluate --eval_type full --test_eps 1 \
    --resume ./checkpoints/TeCoA_ViT-B_eps1/model_best.pth.tar
```

## Mapping experiments → code

The paper's main 600-configuration attack sweep splits into **two pipelines** that share the same model/dataset/ε grid:

| Pipeline | Attack(s) | Orchestrator | Single-attack runner | Core attack code |
|---|---|---|---|---|
| **PGD/C&W family** | PGD, C&W | `run_attack_comparison.py` | `run_attack.py` | `attacks.py` (function-style) |
| **SAPA family** (the paper's contribution) | SAPA | `run_semantic_attack_comparison.py` | `run_semantic_attack_config.py` | `sapa/` (class-style: `SemanticFeatureSpacePerturbation`, Phase 1–4) |
| **Targeted baselines** | targeted_pgd, targeted_cw | `run_attack_comparison.py` | `run_attack.py` | `attacks.py` |


### PGD / C&W sweep (subset of Table 3)

5 models × 10 datasets × 4 ε for PGD and C&W:
```bash
python run_attack_comparison.py --config config/config.yaml --attacks pgd CW
```

- `run_attack_comparison.py` orchestrates the sweep (driver, batches, caching, summary export).
- `attacks.py::attack_pgd`, `attacks.py::attack_CW` implement the PGD and C&W attacks.
- `replace/model_loader.py` loads each of the 5 models (CLIP, TeCoA, FARE, TRADES, PMG-AFT).
- `replace/tv_datasets/dataset_ops.py` loads each of the 10 datasets.

### SAPA sweep (the paper's main contribution — Table 3 SAPA row)

5 models × 10 datasets × 4 ε for SAPA, using the proper Phase-1-to-4 implementation:
```bash
python run_semantic_attack_comparison.py --config config/sapa.yaml
```

- `run_semantic_attack_comparison.py` orchestrates the SAPA sweep, dispatching subprocess calls to `run_semantic_attack_config.py` per (model, dataset, ε).
- `run_semantic_attack_config.py` is the per-configuration single-attack runner.
- `sapa/semantic_wordnet_anchor.py` (Phase 1) selects WordNet semantic anchors.
- `sapa/semantic_clip_anchor.py` (Phase 1) refines anchors via CLIP filtering.
- `sapa/semantic_feature_perturbation.py` (Phase 2) implements the final-layer alignment as `SemanticFeatureSpacePerturbation` class.
- `sapa/llava_text_adaption.py` (optional Phase 3) provides LLaVA-based test-time text adaptation.
- `data_utils/dataset_adaptive_semantic_target.py` provides per-dataset target adaptation.

### Targeted baselines (Table 11)

5 models × 3 datasets (OxfordPets, ImageNet, Caltech-101) × 4 ε for `targeted_pgd` and `targeted_cw`:
```bash
python run_attack_comparison.py --config config/targeted_baseline.yaml \
       --attacks targeted_pgd targeted_cw
```

- `attacks.py::attack_pgd_targeted_semantic`, `attacks.py::attack_CW_targeted_semantic` implement the targeted variants using the CLIP Semantic Anchor target selection (`sapa/semantic_wordnet_anchor.py`).

## Setup

```bash
pip install -r requirements.txt
# Additional: NLTK WordNet
python -c "import nltk; nltk.download('wordnet')"
# For MobileCLIP:
pip install open_clip_torch
```

Models are expected at `./models/` (CLIP downloads automatically; TeCoA/FARE/TRADES/PMG-AFT/AUDIENCE checkpoints load via `replace/model_loader.py` — see in-file paths).

Datasets are expected at `./datasets/` (set via `data.path` in YAML configs).

## Environment

The experiments reported in the paper were executed on the following hardware:

| Component | Specification |
|---|---|
| **GPU** | 1 × NVIDIA H100 (80 GB VRAM) |
| **CPU** | Intel Core i9-14900K |
| **RAM** | 32 GB |
| **OS** | Linux (Ubuntu-class) |
| **CUDA** | ≥ 12.1 (driver compatible with PyTorch 2.7) |
| **PyTorch** | 2.7.0 (see `requirements.txt`) |

Single-GPU execution; no multi-GPU / DDP setup is required for any experiment in the paper.

## Reproduction notes

- All randomness is seeded.
- **Training the four victim models** (TeCoA / PMG-AFT / TRADES / Joint Adv\_Img+Text at ε=1/255) — ~12–18 hours each on 1×H100; ~7–10 hours on 2×H100 with DDP.
- **Full attack sweep (Table 3)** — 5 models × 10 datasets × 4 ε × 3 attacks = 600 configurations: ~**5–8 days** of wall-clock time on the environment above. SAPA is the dominant cost (~2.5× the per-config runtime of PGD/C\&W due to GEO's joint optimization of image perturbation and target embedding).
- **Targeted baseline sweep (Targeted PGD/C\&W on 5 models × 3 datasets × 4 ε)** — adds ~1–2 days.

For practical re-runs, narrowing the sweep via `--models`, `--datasets`, `--attacks`, `--epsilons` flags on `run_attack_comparison.py` lets reviewers verify a representative subset (e.g., one robust model on three datasets at one ε) in 2–4 hours.

## What is *not* included in this bundle

- **Pretrained model checkpoints** (TeCoA, FARE, TRADES, PMG-AFT) — these are publicly distributed by their original authors under their own licenses. Train them locally using the commands in the *Training* section, or download from the original repos (FARE: see Schlarmann et al. links above).
- **Datasets** — all 10 are public benchmarks (Caltech-101/256, ImageNet, OxfordPets, StanfordCars, FGVCAircraft, Food-101, DTD, EuroSAT, SUN397). Standard download paths are referenced in `replace/tv_datasets/`.
- **Result files** (text logs, XLSX summaries) — regenerated by running the scripts above.

## License

This research code is released under the same license as the paper's anonymous artifact submission. After acceptance, code will be released under MIT/Apache-2.0 with attribution.
