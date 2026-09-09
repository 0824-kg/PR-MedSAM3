#!/usr/bin/env python3
"""Validate recovered COCO-format datasets for PR-MedSAM3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rle_area(segmentation: Dict) -> int:
    counts = segmentation.get("counts")
    if not isinstance(counts, list):
        return 0
    # COCO RLE counts alternate background/foreground runs.
    return int(sum(counts[1::2]))


def segmentation_area(segmentation) -> int:
    if isinstance(segmentation, dict):
        return rle_area(segmentation)
    if isinstance(segmentation, list):
        area = 0.0
        for poly in segmentation:
            if not isinstance(poly, list) or len(poly) < 6:
                continue
            xs = poly[0::2]
            ys = poly[1::2]
            shoelace = 0.0
            for idx in range(len(xs)):
                j = (idx + 1) % len(xs)
                shoelace += xs[idx] * ys[j] - xs[j] * ys[idx]
            area += abs(shoelace) / 2.0
        return int(area)
    return 0


def validate_split(dataset_root: Path, split: str) -> Tuple[List[str], List[str], List[float], int]:
    ann_path = dataset_root / split / "_annotations.coco.json"
    errors: List[str] = []
    warnings: List[str] = []
    areas: List[float] = []

    if not ann_path.exists():
        errors.append(f"Missing annotation file: {ann_path}")
        return errors, warnings, areas, 0

    data = load_json(ann_path)
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])
    if not categories:
        errors.append(f"{split}: no categories found")

    image_by_id = {}
    file_names: List[str] = []
    for img in images:
        img_id = img.get("id")
        file_name = img.get("file_name")
        width = img.get("width")
        height = img.get("height")
        if img_id is None:
            errors.append(f"{split}: image without id")
            continue
        if not file_name:
            errors.append(f"{split}: image {img_id} without file_name")
            continue
        image_path = dataset_root / split / file_name
        if not image_path.exists():
            errors.append(f"{split}: referenced image missing: {image_path}")
        if not width or not height:
            errors.append(f"{split}: image {file_name} missing width/height")
        image_by_id[img_id] = img
        file_names.append(Path(str(file_name)).name)

    for ann in annotations:
        ann_id = ann.get("id")
        image_id = ann.get("image_id")
        if image_id not in image_by_id:
            errors.append(f"{split}: annotation {ann_id} has invalid image_id={image_id}")
        bbox = ann.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append(f"{split}: annotation {ann_id} invalid bbox")
        else:
            _, _, w, h = bbox
            if w <= 0 or h <= 0:
                errors.append(f"{split}: annotation {ann_id} non-positive bbox={bbox}")
        segmentation = ann.get("segmentation")
        if segmentation is None:
            errors.append(f"{split}: annotation {ann_id} missing segmentation")
        else:
            seg_area = segmentation_area(segmentation)
            ann_area = float(ann.get("area", 0) or 0)
            area_value = ann_area if ann_area > 0 else float(seg_area)
            if area_value <= 0:
                errors.append(f"{split}: annotation {ann_id} has empty mask/area")
            else:
                areas.append(area_value)
        if ann.get("category_id") is None:
            errors.append(f"{split}: annotation {ann_id} missing category_id")

    images_without_ann = set(image_by_id) - {ann.get("image_id") for ann in annotations}
    if images_without_ann:
        warnings.append(f"{split}: {len(images_without_ann)} images have no annotation")

    print(f"[{split}] images={len(images)} annotations={len(annotations)} categories={[c.get('name') for c in categories]}")
    return errors, warnings, areas, len(annotations)


def write_split_summary(dataset_root: Path, dataset_name: str) -> None:
    import csv

    rows = []
    for split in ("train", "valid"):
        ann_path = dataset_root / split / "_annotations.coco.json"
        if not ann_path.exists():
            continue
        data = load_json(ann_path)
        ann_counts: Dict[int, int] = {}
        for ann in data.get("annotations", []):
            ann_counts[int(ann["image_id"])] = ann_counts.get(int(ann["image_id"]), 0) + 1
        for img in data.get("images", []):
            rows.append(
                {
                    "dataset": dataset_name,
                    "split": split,
                    "image_id": img.get("id"),
                    "file_name": img.get("file_name"),
                    "width": img.get("width"),
                    "height": img.get("height"),
                    "num_annotations": ann_counts.get(int(img.get("id")), 0),
                }
            )
    if rows:
        with open(dataset_root / "split_summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def read_split_file_names(dataset_root: Path, split: str) -> set[str]:
    ann_path = dataset_root / split / "_annotations.coco.json"
    if not ann_path.exists():
        return set()
    data = load_json(ann_path)
    return {Path(str(img.get("file_name", ""))).name for img in data.get("images", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify recovered PR-MedSAM3 COCO dataset")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    all_errors: List[str] = []
    all_warnings: List[str] = []
    all_areas: List[float] = []
    total_annotations = 0

    for split in ("train", "valid"):
        errors, warnings, areas, num_annotations = validate_split(dataset_root, split)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
        all_areas.extend(areas)
        total_annotations += num_annotations

    train_files = read_split_file_names(dataset_root, "train")
    valid_files = read_split_file_names(dataset_root, "valid")
    overlap = train_files & valid_files
    if overlap:
        all_errors.append(f"train/valid overlap contains {len(overlap)} file(s): {sorted(overlap)[:5]}")

    if all_areas:
        print(
            "[Area] foreground area: "
            f"min={min(all_areas):.1f} mean={mean(all_areas):.1f} max={max(all_areas):.1f}"
        )
    print(f"[Total] annotations={total_annotations}")

    for warning in all_warnings:
        print(f"[Warning] {warning}")
    if all_errors:
        for error in all_errors:
            print(f"[Error] {error}")
        raise SystemExit(1)

    write_split_summary(dataset_root, args.dataset_name)
    print("[OK] COCO dataset validation passed.")


if __name__ == "__main__":
    main()
