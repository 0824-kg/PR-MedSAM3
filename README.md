# PR-MedSAM3

## Overview

PR-MedSAM3 is a prompt-robust framework for text-guided medical image segmentation. It addresses prompt-induced variability: different textual expressions that refer to the same annotated target may lead to different segmentation masks.

The framework combines:

- **Concept-Prompt Calibration Adapter (CPCA-v2)** for concept-level prompt calibration;
- **Group-Guided Prompt-Stable Mask Adaptation (GPSMA)** for mask-level stabilization; and
- **group-consistency training** to encourage consistent predictions across same-target prompt expressions.

The study evaluates prompt robustness using canonical-prompt Dice together with:

- **Worst-Prompt Dice (WPD)**, which summarizes the least-favorable Dice score across evaluated same-target prompt expressions; and
- **Prompt Robustness Gap (PRG)**, which summarizes the corresponding best-to-worst variation.

WPD and PRG should be interpreted jointly. Same-target prompt robustness and mismatched-concept prompt specificity are treated as separate aspects of model behavior.

---

## Repository Structure

The main files and directories used by the PR-MedSAM3 workflow are:

```text
PR-MedSAM3/
├── configs/                       # Training configurations and prompt inventories
├── models/                        # CPCA-v2, GPSMA, and PR-MedSAM3 wrapper modules
├── sam3/                          # SAM3 source used by the project
├── sam3_lora/                     # Supporting LoRA implementation
├── sam3_lora_configs/             # Additional LoRA configurations
├── scripts/                       # Dataset-specific and analysis scripts
├── tools/                         # Dataset preparation and verification utilities
├── prompt_utils.py                # Prompt loading, normalization, and sampling
├── convert_binary_mask_dataset_to_coco.py
├── train_medsam3_prompt_robust.py # Main PR-MedSAM3 training entry point
├── train_sam3_lora_native.py      # Canonical-prompt LoRA baseline trainer
├── validate_sam3_lora.py          # Standard validation/inference
├── validate_prompt_robustness.py  # Same-target prompt-robustness evaluation
├── requirements.txt
└── README.md
```

The CPCA-v2 and GPSMA implementations are located in:

```text
models/prompt_robust_modules.py
```

Their integration with the segmentation model is implemented in:

```text
models/medsam3_prompt_robust_wrapper.py
```

The runtime seen-prompt inventory is stored in:

```text
configs/prompts.yaml
```

---

## Requirements

Install the Python dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

The recorded environment used for the manuscript experiments was:

- Python 3.10.0
- PyTorch 2.12.0+cu130
- CUDA 13.0
- NVIDIA GeForce RTX 4090

These values describe the recorded experimental environment and are not intended to define the only compatible software or hardware configuration.

The main PR-MedSAM3 training and frozen-checkpoint evaluation workflows require CUDA-compatible GPU execution.

---

## Third-Party Datasets

The original medical images and annotations are **not redistributed in this repository**. Users should obtain the datasets from their original public sources and comply with the corresponding licenses and terms of use.

### Breast Ultrasound Images (BUSI)

Original dataset publication:

https://doi.org/10.1016/j.dib.2019.104863

The original BUSI dataset contains 780 image records. The study retained 647 image records with nonempty foreground masks and used the following fixed partition:

- 452 training images
- 97 validation images
- 98 test images

The study partition was performed at the image-record level. Patient identifiers were not available, so patient-level separation could not be verified.

Some BUSI images have multiple associated masks. The repository preparation code retains nonempty associated masks as annotations for the corresponding image, and evaluation combines foreground annotations into a binary reference mask.

### Kvasir-Sessile

Kvasir-Sessile is a publicly available subset associated with Kvasir-SEG.

Official dataset resource:

https://datasets.simula.no/kvasir-seg/

Official Kvasir-Sessile download:

https://datasets.simula.no/downloads/kvasir-sessile.zip

The study used 196 image-mask pairs with the following fixed partition:

- 118 training pairs
- 39 validation pairs
- 39 test pairs

### ISIC2018

Official ISIC Challenge data page:

https://challenge.isic-archive.com/data/

ISIC 2018 Challenge page:

https://challenge.isic-archive.com/landing/2018/

The study used the lesion-segmentation data with the following fixed partition:

- 2,594 training images
- 100 validation images
- 1,000 test images

---

## Dataset Preparation

The main loaders use COCO-style split directories:

```text
DATASET_ROOT/
├── train/
│   ├── images/
│   └── _annotations.coco.json
├── valid/
│   ├── images/
│   └── _annotations.coco.json
└── test/
    ├── images/
    └── _annotations.coco.json
```

### BUSI

The BUSI preparation utility is:

```bash
python tools/prepare_busi_coco.py \
  --dataset-root datasets/BUSI \
  --output-root prepared_datasets/BUSI \
  --seed 42
```

### Kvasir-Sessile

For an already prepared fixed split layout:

```bash
python convert_binary_mask_dataset_to_coco.py \
  --dataset-root datasets/sessile-Kvasir \
  --output-root prepared_datasets/sessile-Kvasir \
  --category-name "sessile polyp" \
  --copy-images
```

### ISIC2018

For an already prepared fixed split layout:

```bash
python convert_binary_mask_dataset_to_coco.py \
  --dataset-root datasets/ISIC2018 \
  --output-root prepared_datasets/ISIC2018 \
  --category-name "skin lesion" \
  --copy-images
```

The Kvasir-Sessile and ISIC2018 conversion command assumes that the intended train/validation/test split has already been arranged before conversion.

During PR-MedSAM3 training, images are loaded as RGB images, resized to 1008 × 1008, converted to floating point, and normalized. Segmentation masks are resized with nearest-neighbor interpolation and binarized before training.

---

## Prompt Inventories

The canonical target expressions are:

| Dataset | Canonical prompt |
|---|---|
| BUSI | `breast tumor` |
| Kvasir-Sessile | `sessile polyp` |
| ISIC2018 | `skin lesion` |

The primary same-target prompt categories are:

- synonym expressions;
- lexical variations; and
- challenging expressions.

The full seen-prompt inventory is stored in:

```text
configs/prompts.yaml
```

Prompt loading, normalization, deduplication, and sampling are implemented in:

```text
prompt_utils.py
```

For grouped training, the default group size is:

```text
K = 2
```

with:

- slot 0: canonical prompt;
- slot 1: one alternative expression sampled from the pooled noncanonical prompt inventory.

Sampling is performed from the pooled alternative inventory rather than uniformly across prompt categories.

The repository also contains the fixed inventories used for held-out same-target and prompt-specificity analyses.

---

## Training

The main PR-MedSAM3 training entry point is:

```text
train_medsam3_prompt_robust.py
```

A BUSI training example is:

```bash
python train_medsam3_prompt_robust.py \
  --config configs/prompt_robust_busi_cpca_v2_gpsma.yaml \
  --robust_method cpca_v2_gpsma \
  --data_dir prepared_datasets/BUSI \
  --output_dir work_dir/BUSI/prompt_robust/cpca_v2_gpsma \
  --prompts_config configs/prompts.yaml \
  --prompt_dataset_key busi \
  --device 0
```

Kvasir-Sessile:

```bash
python train_medsam3_prompt_robust.py \
  --config configs/prompt_robust_kvasir_cpca_v2_gpsma.yaml \
  --robust_method cpca_v2_gpsma \
  --data_dir prepared_datasets/sessile-Kvasir \
  --output_dir work_dir/sessile-Kvasir/prompt_robust/cpca_v2_gpsma \
  --prompts_config configs/prompts.yaml \
  --prompt_dataset_key sessile_kvasir \
  --device 0
```

ISIC2018:

```bash
python train_medsam3_prompt_robust.py \
  --config configs/prompt_robust_isic2018_cpca_v2_gpsma.yaml \
  --robust_method cpca_v2_gpsma \
  --data_dir prepared_datasets/ISIC2018 \
  --output_dir work_dir/ISIC2018/prompt_robust/cpca_v2_gpsma \
  --prompts_config configs/prompts.yaml \
  --prompt_dataset_key isic2018 \
  --device 0
```

The reported configuration uses:

| Setting | Value |
|---|---:|
| Vision encoder | frozen |
| LoRA rank | 16 |
| LoRA scaling factor | 32 |
| LoRA dropout | 0.1 |
| Optimizer | AdamW |
| Epochs | 20 |
| Batch size | 1 |
| Gradient accumulation | 8 |
| Weight decay | 0.01 |
| Learning rate | 1e-4 |
| Group-consistency weight | 0.1 |
| Group size | K = 2 |
| Recorded training seed | 42 |

The required base SAM3 and MedSAM3 initialization weights are not redistributed in this repository and should be obtained from their upstream sources.

---

## Standard Validation / Inference

Standard LoRA validation is performed with:

```text
validate_sam3_lora.py
```

Example:

```bash
python validate_sam3_lora.py \
  --config configs/busi_ft_canon.yaml \
  --weights checkpoints/best_lora_weights.pt \
  --val_data_dir prepared_datasets/BUSI/test \
  --prompt-mode canonical \
  --prompts-config configs/prompts.yaml \
  --prompt-dataset-key busi \
  --save-per-image-csv \
  --per-image-csv-path results/busi_canonical_validation.csv
```

The evaluation script reports segmentation metrics including Dice and IoU together with the additional metrics implemented by the validation workflow.

---

## Same-Target Prompt-Robustness Evaluation

Prompt robustness is evaluated with:

```text
validate_prompt_robustness.py
```

Example:

```bash
python validate_prompt_robustness.py \
  --config configs/prompt_robust_busi_cpca_v2_gpsma.yaml \
  --checkpoint work_dir/BUSI/prompt_robust/cpca_v2_gpsma/model_best.pth \
  --data_dir prepared_datasets/BUSI/test \
  --dataset_key busi \
  --prompts_config configs/prompts.yaml \
  --prompt_group all \
  --save_per_image_csv \
  --per_image_csv_path results/busi_seen_per_prompt.csv \
  --save_robustness_csv \
  --robustness_csv_path results/busi_seen_robustness.csv \
  --method cpca_v2_gpsma
```

### Worst-Prompt Dice (WPD)

For image `i` and evaluated same-target prompt set `k`:

```text
WPD = mean_i(min_k Dice(i, k))
```

WPD therefore summarizes the least-favorable segmentation performance across the evaluated same-target prompt expressions.

### Prompt Robustness Gap (PRG)

```text
PRG = mean_i(max_k Dice(i, k) - min_k Dice(i, k))
```

PRG summarizes the image-level best-to-worst variation across the evaluated same-target expressions.

WPD and PRG should be interpreted jointly because a small PRG can also occur when all prompt-conditioned predictions perform poorly.

---

## Held-Out Same-Target Prompt Evaluation

Held-out prompt evaluation uses frozen model checkpoints and does not involve retraining.

For PR-MedSAM3, held-out same-target expressions are evaluated using the canonical prompt together with the current held-out alternative in a two-prompt group:

```text
K = 2
```

The held-out prompt inventory is stored in:

```text
medical_physics_revision/no_retraining_experiments/configs/heldout_prompts.yaml
```

The corresponding evaluation implementation is:

```text
medical_physics_revision/no_retraining_experiments/scripts/03_heldout_prompts.py
```

This experiment evaluates only the predefined held-out same-target expression inventory and should not be interpreted as evidence of robustness to arbitrary free-form or clinical language.

---

## Prompt-Specificity Proxy

Prompt-specificity evaluation uses frozen checkpoints and evaluates:

- valid same-target prompts;
- cross-concept prompts; and
- irrelevant or nonspecific prompts.

Each prompt is evaluated independently using single-prompt inference:

```text
K = 1
```

For mismatched prompts, the correct canonical prompt is not supplied simultaneously.

The prompt inventory is stored in:

```text
medical_physics_revision/no_retraining_experiments/configs/specificity_prompts.yaml
```

The corresponding evaluation implementation is:

```text
medical_physics_revision/no_retraining_experiments/scripts/04_prompt_specificity.py
```

This analysis is intended as a **prompt-specificity proxy** and should not be interpreted as a clinical false-positive specificity test.

---

## Reproducing the Reported Experiments

A typical reproduction workflow is:

1. Obtain BUSI, Kvasir-Sessile, and ISIC2018 from their original sources.
2. Prepare the fixed train/validation/test partitions used by the study.
3. Convert the datasets to the COCO-style layout required by the repository.
4. Install the required Python dependencies.
5. Obtain the required SAM3 and MedSAM3 pretrained weights.
6. Configure dataset and checkpoint paths.
7. Train the selected model variant.
8. Run canonical-prompt evaluation on the fixed test partition.
9. Run same-target prompt-robustness evaluation.
10. Run the frozen-checkpoint held-out and prompt-specificity analyses when required.

The reported experiments use the following fixed partitions:

- **BUSI:** 452 train / 97 validation / 98 test
- **Kvasir-Sessile:** 118 train / 39 validation / 39 test
- **ISIC2018:** 2,594 train / 100 validation / 1,000 test

The primary and additional evaluations reported in the manuscript are performed on the fixed test partitions. Test-set metrics are not used for model optimization or checkpoint selection.

The main training configuration records one training seed, 42. The study is not presented as a multi-seed experiment.

Held-out same-target prompt evaluation and prompt-specificity analysis use frozen checkpoints and do not involve retraining.

---

## Pretrained Models and Upstream Code

This project builds on the following open-source resources:

- SAM3: https://github.com/facebookresearch/sam3
- MedSAM3: https://github.com/Joey-S-Liu/MedSAM3
- SAM3_LoRA: https://github.com/Sompote/SAM3_LoRA

The corresponding pretrained weight files are not redistributed in this repository. Users should obtain them from their original sources and comply with the corresponding licenses and terms of use.

---

## Code Availability

The source code is publicly available at:

https://github.com/0824-kg/PR-MedSAM3

The version corresponding to the PeerJ submission has been archived at Zenodo:

https://doi.org/10.5281/zenodo.22673947

---

## Citation

If you use this repository, please cite the corresponding PR-MedSAM3 manuscript:

> **PR-MedSAM3: Reliability-Aware Concept and Mask Stabilization for Prompt-Robust Text-Guided Medical Image Segmentation**

Full bibliographic information and the final article DOI will be added after publication.

---

## License

No standalone license file is currently included in this repository. Users must also comply with the licenses and terms of the upstream SAM3, MedSAM3, SAM3_LoRA, and third-party dataset resources.
