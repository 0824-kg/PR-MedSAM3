#!/usr/bin/env python3
"""
Prepare the raw BUSI dataset as COCO annotations for MedSAM3.

Expected raw layout:
  BUSI/
    benign/
      benign (1).png
      benign (1)_mask.png
    malignant/
      malignant (1).png
      malignant (1)_mask.png
    normal/
      normal (1).png
      normal (1)_mask.png

Output layout:
  BUSI/
    train/images/
    train/_annotations.coco.json
    valid/images/
    valid/_annotations.coco.json
    test/images/
    test/_annotations.coco.json
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from convert_binary_mask_dataset_to_coco import IMAGE_EXTENSIONS, build_annotation


CATEGORY = {
    "id": 1,
    "name": "breast tumor",
    "supercategory": "breast tumor",
}
DEFAULT_DATASET_ROOT = Path("datasets/BUSI")
DEFAULT_OUTPUT_ROOT = Path("/root/shared-nvme/prepared_datasets/BUSI")
FOREGROUND_CLASSES = ("benign", "malignant")
OPTIONAL_CLASSES = ("normal",)
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class Sample:
    image_path: Path
    mask_paths: Tuple[Path, ...]
    source_class: str


@dataclass(frozen=True)
class PrepareStats:
    dataset_root: Path
    output_root: Path
    split_counts: Dict[str, int]
    split_annotation_counts: Dict[str, int]
    skipped_normal: bool
    skipped_normal_images: int
    included_normal_images: int
    category_name: str


def mask_base_stem(mask_stem: str) -> str:
    marker = "_mask"
    lower = mask_stem.lower()
    marker_index = lower.find(marker)
    if marker_index == -1:
        return mask_stem
    return mask_stem[:marker_index]


def is_mask_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and "_mask" in path.stem.lower()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and "_mask" not in path.stem.lower()


def build_class_mask_index(class_dir: Path) -> Dict[str, List[Path]]:
    mask_index: Dict[str, List[Path]] = {}
    for mask_path in sorted(class_dir.iterdir()):
        if not is_mask_file(mask_path):
            continue
        mask_index.setdefault(mask_base_stem(mask_path.stem), []).append(mask_path)
    return mask_index


def load_binary_mask(mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as mask_image:
        return np.array(mask_image.convert("L")) > 0


def valid_foreground_masks(mask_paths: Iterable[Path]) -> Tuple[Path, ...]:
    valid_masks = []
    for mask_path in mask_paths:
        mask = load_binary_mask(mask_path)
        if bool(mask.any()):
            valid_masks.append(mask_path)
    return tuple(valid_masks)


def collect_samples(dataset_root: Path) -> Tuple[List[Sample], int, int]:
    samples: List[Sample] = []
    skipped_normal_images = 0
    included_normal_images = 0

    for class_name in (*FOREGROUND_CLASSES, *OPTIONAL_CLASSES):
        class_dir = dataset_root / class_name
        if not class_dir.is_dir():
            continue

        mask_index = build_class_mask_index(class_dir)
        for image_path in sorted(path for path in class_dir.iterdir() if is_image_file(path)):
            candidate_masks = mask_index.get(image_path.stem, [])
            if not candidate_masks:
                if class_name == "normal":
                    skipped_normal_images += 1
                continue

            valid_masks = valid_foreground_masks(candidate_masks)
            if not valid_masks:
                if class_name == "normal":
                    skipped_normal_images += 1
                continue

            if class_name == "normal":
                included_normal_images += 1

            samples.append(
                Sample(
                    image_path=image_path,
                    mask_paths=valid_masks,
                    source_class=class_name,
                )
            )

    return samples, skipped_normal_images, included_normal_images


def split_samples(
    samples: Sequence[Sample],
    seed: int,
    train_ratio: float,
    valid_ratio: float,
) -> Dict[str, List[Sample]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * train_ratio)
    valid_count = int(total * valid_ratio)
    test_count = total - train_count - valid_count

    return {
        "train": shuffled[:train_count],
        "valid": shuffled[train_count : train_count + valid_count],
        "test": shuffled[train_count + valid_count : train_count + valid_count + test_count],
    }


def unique_destination_name(image_path: Path, used_names: set[str]) -> str:
    destination_name = image_path.name
    if destination_name not in used_names:
        used_names.add(destination_name)
        return destination_name

    stem = image_path.stem
    suffix = image_path.suffix
    counter = 1
    while True:
        destination_name = f"{stem}_{counter}{suffix}"
        if destination_name not in used_names:
            used_names.add(destination_name)
            return destination_name
        counter += 1


def convert_split(split_name: str, samples: Sequence[Sample], output_root: Path) -> Tuple[int, int]:
    split_dir = output_root / split_name
    image_dir = split_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    used_names: set[str] = set()
    ann_id = 1

    for image_id, sample in enumerate(samples, start=1):
        destination_name = unique_destination_name(sample.image_path, used_names)
        shutil.copy2(sample.image_path, image_dir / destination_name)

        with Image.open(sample.image_path) as image:
            width, height = image.size

        images.append(
            {
                "id": image_id,
                "file_name": f"images/{destination_name}",
                "width": width,
                "height": height,
            }
        )

        for mask_path in sample.mask_paths:
            mask = load_binary_mask(mask_path)
            annotation = build_annotation(mask, image_id=image_id, ann_id=ann_id, category_id=1)
            if annotation is None:
                continue
            annotations.append(annotation)
            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [CATEGORY],
    }

    with open(split_dir / "_annotations.coco.json", "w", encoding="utf-8") as handle:
        json.dump(coco, handle)

    return len(images), len(annotations)


def prepare_busi_coco(
    dataset_root: Path,
    output_root: Path,
    seed: int = 42,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    clean_output: bool = True,
) -> PrepareStats:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"BUSI dataset root not found: {dataset_root}")

    samples, skipped_normal_images, included_normal_images = collect_samples(dataset_root)
    if not samples:
        raise RuntimeError(f"No BUSI images with valid foreground masks found under {dataset_root}")

    split_map = split_samples(
        samples=samples,
        seed=seed,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    if clean_output:
        for split_name in SPLITS:
            split_dir = output_root / split_name
            if split_dir.exists():
                shutil.rmtree(split_dir)

    split_counts: Dict[str, int] = {}
    split_annotation_counts: Dict[str, int] = {}
    for split_name in SPLITS:
        image_count, annotation_count = convert_split(split_name, split_map[split_name], output_root)
        split_counts[split_name] = image_count
        split_annotation_counts[split_name] = annotation_count

    return PrepareStats(
        dataset_root=dataset_root,
        output_root=output_root,
        split_counts=split_counts,
        split_annotation_counts=split_annotation_counts,
        skipped_normal=skipped_normal_images > 0,
        skipped_normal_images=skipped_normal_images,
        included_normal_images=included_normal_images,
        category_name=CATEGORY["name"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare raw BUSI data as COCO for MedSAM3.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Raw BUSI dataset root. Default: {DEFAULT_DATASET_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"COCO output root. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random split seed. Default: 42")
    parser.add_argument(
        "--no-clean-output",
        action="store_true",
        help="Do not remove existing train/valid/test directories under the output root before writing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = prepare_busi_coco(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        seed=args.seed,
        clean_output=not args.no_clean_output,
    )

    print(f"Raw BUSI path: {stats.dataset_root}")
    print(f"Output COCO path: {stats.output_root}")
    print(f"Category name: {stats.category_name}")
    print(
        "Normal class: "
        f"skipped {stats.skipped_normal_images}, included {stats.included_normal_images}"
    )
    for split_name in SPLITS:
        print(
            f"{split_name}: {stats.split_counts[split_name]} images, "
            f"{stats.split_annotation_counts[split_name]} annotations"
        )


if __name__ == "__main__":
    main()
