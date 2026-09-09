#!/usr/bin/env python3
"""
Convert binary mask segmentation datasets into the COCO layout expected by
train_sam3_lora_native.py.

Input layout examples supported:
  dataset_root/
    Train/image/*.jpg
    Train/masks/*_segmentation.png
    Val/image/*.jpg
    Test/image/*.jpg

  dataset_root/
    train/images/*.jpg
    train/masks/*.jpg
    val/images/*.jpg

Output layout:
  output_root/
    train/
      images/*.jpg
      _annotations.coco.json
    valid/
      images/*.jpg
      _annotations.coco.json
    test/
      images/*.jpg
      _annotations.coco.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "tr": "train",
    "val": "valid",
    "valid": "valid",
    "validation": "valid",
    "dev": "valid",
    "test": "test",
    "testing": "test",
    "te": "test",
}
IMAGE_DIR_CANDIDATES = ("images", "image", "imgs", "img")
MASK_DIR_CANDIDATES = ("masks", "mask", "labels", "label", "ground_truth")
MASK_SUFFIX_CANDIDATES = (
    "",
    "_segmentation",
    "_mask",
    "_lesion",
    "_polyp",
)


@dataclass
class SplitPaths:
    split_name: str
    image_dir: Path
    mask_dir: Path


def normalize_name(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def infer_category_name(dataset_root: Path) -> str:
    root_name = dataset_root.name.lower()
    if "isic" in root_name:
        return "skin lesion"
    if "kvasir" in root_name:
        return "sessile polyp"
    return "object"


def detect_split_dirs(dataset_root: Path) -> List[SplitPaths]:
    split_paths: List[SplitPaths] = []

    for child in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        normalized = normalize_name(child.name)
        canonical_split = None
        for alias, mapped in SPLIT_ALIASES.items():
            if normalized == normalize_name(alias):
                canonical_split = mapped
                break

        if canonical_split is None:
            continue

        image_dir = first_existing_dir(child, IMAGE_DIR_CANDIDATES)
        mask_dir = first_existing_dir(child, MASK_DIR_CANDIDATES)
        if image_dir is None or mask_dir is None:
            raise FileNotFoundError(
                f"Could not find image/mask directories under {child}. "
                f"Expected one of {IMAGE_DIR_CANDIDATES} and {MASK_DIR_CANDIDATES}."
            )

        split_paths.append(
            SplitPaths(split_name=canonical_split, image_dir=image_dir, mask_dir=mask_dir)
        )

    if not split_paths:
        raise FileNotFoundError(f"No supported split directories found under {dataset_root}")

    return split_paths


def first_existing_dir(root: Path, candidates: Iterable[str]) -> Optional[Path]:
    existing = {normalize_name(path.name): path for path in root.iterdir() if path.is_dir()}
    for candidate in candidates:
        match = existing.get(normalize_name(candidate))
        if match is not None:
            return match
    return None


def build_mask_index(mask_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for mask_path in sorted(mask_dir.iterdir()):
        if not mask_path.is_file() or mask_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = normalize_mask_stem(mask_path.stem)
        index[key] = mask_path
    return index


def normalize_mask_stem(stem: str) -> str:
    normalized = stem
    for suffix in MASK_SUFFIX_CANDIDATES[1:]:
        if normalized.lower().endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def find_mask_for_image(image_path: Path, mask_index: Dict[str, Path]) -> Optional[Path]:
    image_stem = image_path.stem
    direct_candidates = [image_stem]
    for suffix in MASK_SUFFIX_CANDIDATES[1:]:
        direct_candidates.append(f"{image_stem}{suffix}")

    for candidate in direct_candidates:
        key = normalize_mask_stem(candidate)
        if key in mask_index:
            return mask_index[key]
    return None


def encode_uncompressed_rle(mask: np.ndarray) -> Dict[str, object]:
    flat = mask.astype(np.uint8).T.flatten()
    counts: List[int] = []
    prev = 0
    run_length = 0

    for pixel in flat:
        if int(pixel) == prev:
            run_length += 1
        else:
            counts.append(run_length)
            run_length = 1
            prev = int(pixel)
    counts.append(run_length)

    return {
        "size": [int(mask.shape[0]), int(mask.shape[1])],
        "counts": counts,
    }


def build_annotation(mask: np.ndarray, image_id: int, ann_id: int, category_id: int) -> Optional[Dict[str, object]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    area = int(mask.sum())

    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x_min, y_min, width, height],
        "area": area,
        "iscrowd": 0,
        "segmentation": encode_uncompressed_rle(mask),
    }


def convert_split(
    split_paths: SplitPaths,
    output_root: Path,
    category_name: str,
    copy_images: bool,
) -> Tuple[int, int]:
    output_split_dir = output_root / split_paths.split_name
    output_split_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir = output_split_dir / "images"
    if copy_images:
        output_images_dir.mkdir(parents=True, exist_ok=True)

    mask_index = build_mask_index(split_paths.mask_dir)
    images = []
    annotations = []
    image_id = 1
    ann_id = 1

    for image_path in sorted(split_paths.image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        mask_path = find_mask_for_image(image_path, mask_index)
        if mask_path is None:
            continue

        if copy_images:
            dst_name = image_path.name
            dst_path = output_images_dir / dst_name
            shutil.copy2(image_path, dst_path)
            file_name = f"images/{dst_name}"
        else:
            # Store the original absolute image path so the training loader can
            # resolve it directly without copying large image folders around.
            file_name = str(image_path.resolve()).replace("\\", "/")

        with Image.open(image_path) as image:
            width, height = image.size

        with Image.open(mask_path) as mask_image:
            mask = np.array(mask_image.convert("L")) > 0

        annotation = build_annotation(mask, image_id=image_id, ann_id=ann_id, category_id=1)
        if annotation is None:
            continue

        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )
        annotations.append(annotation)

        image_id += 1
        ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": category_name,
                "supercategory": category_name,
            }
        ],
    }

    with open(output_split_dir / "_annotations.coco.json", "w", encoding="utf-8") as handle:
        json.dump(coco, handle)

    return len(images), len(annotations)


def convert_dataset(
    dataset_root: Path,
    output_root: Path,
    category_name: Optional[str] = None,
    copy_images: bool = False,
) -> None:
    category_name = category_name or infer_category_name(dataset_root)
    split_dirs = detect_split_dirs(dataset_root)

    print(f"Converting dataset: {dataset_root}")
    print(f"Category name: {category_name}")
    print(f"Output root: {output_root}")

    for split_paths in split_dirs:
        num_images, num_annotations = convert_split(
            split_paths,
            output_root,
            category_name,
            copy_images=copy_images,
        )
        print(
            f"  {split_paths.split_name}: {num_images} images, {num_annotations} annotations"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert binary-mask datasets into COCO format for MedSAM3 training"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the dataset root, e.g. datasets/ISIC2017",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Where to write the converted dataset",
    )
    parser.add_argument(
        "--category-name",
        type=str,
        default=None,
        help="Single foreground category name. Auto-inferred if omitted.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into the converted dataset. Default is to reference original images by relative path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_dataset(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        category_name=args.category_name,
        copy_images=args.copy_images,
    )


if __name__ == "__main__":
    main()
