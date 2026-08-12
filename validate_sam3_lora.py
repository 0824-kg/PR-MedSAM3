#!/usr/bin/env python3
"""
Validation script for SAM3 LoRA model
Loads saved weights and runs validation with detailed debugging
"""

import os
import argparse
import yaml
import json
import csv
import re
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
import numpy as np
from PIL import Image as PILImage
import contextlib
import matplotlib.pyplot as plt

# SAM3 Imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.model_misc import SAM3Output
from sam3.train.loss.loss_fns import IABCEMdetr, Boxes, Masks, CORE_LOSS_KEY
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import Datapoint, Image, Object, FindQueryLoaded, InferenceMetadata
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights, count_parameters
from models.medsam3_prompt_robust_wrapper import MedSAM3PromptRobustWrapper

from torchvision.transforms import v2

# Import evaluation modules
from sam3.eval.cgf1_eval import CGF1Evaluator, COCOCustom
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_utils
from sam3.train.masks_ops import rle_encode
from prompt_utils import PromptManager

# Import SAM3's NMS
from sam3.perflib.nms import nms_masks

BOX_PROMPT_MODES = ("text_only", "text_plus_loose_box", "text_plus_tight_box")


def decode_coco_segmentation(segmentation, height, width):
    """Decode polygon, compressed RLE, or uncompressed RLE segmentation."""
    if isinstance(segmentation, dict):
        counts = segmentation.get("counts")
        if isinstance(counts, list):
            segmentation = mask_utils.frPyObjects(segmentation, height, width)
        return mask_utils.decode(segmentation)

    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
        return mask_utils.decode(rle)

    raise TypeError(f"Unknown segmentation format: {type(segmentation)}")


def resolve_config_relative_path(config_path, target_path):
    """Resolve a config-relative path with stable handling for sibling config files."""
    if target_path is None:
        return None

    target = Path(target_path).expanduser()
    if target.is_absolute():
        return target.resolve()

    config_dir = Path(config_path).resolve().parent
    cwd = Path.cwd().resolve()

    candidates = []
    if len(target.parts) == 1:
        candidates.append((config_dir / target).resolve())
    else:
        if target.parts[0] == config_dir.name:
            candidates.append((config_dir.parent / target).resolve())
        candidates.append((config_dir / target).resolve())
    candidates.append((cwd / target).resolve())

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        candidate_key = str(candidate)
        if candidate_key not in seen:
            seen.add(candidate_key)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    if len(target.parts) == 1:
        return config_dir / target
    if target.parts[0] == config_dir.name:
        return config_dir.parent / target
    return config_dir / target


def box_cxcywh_to_xyxy(boxes):
    """Convert normalized [cx, cy, w, h] boxes to normalized [x1, y1, x2, y2]."""
    if boxes.numel() == 0:
        return boxes.clone()
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1)


def box_xyxy_to_cxcywh(boxes):
    """Convert normalized [x1, y1, x2, y2] boxes to normalized [cx, cy, w, h]."""
    if boxes.numel() == 0:
        return boxes.clone()
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1), dim=-1)


def compute_tight_box_from_gt_boxes(gt_boxes):
    """Build one normalized tight box from one or more normalized GT [cx, cy, w, h] boxes."""
    if gt_boxes is None or gt_boxes.numel() == 0:
        return None
    gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes.view(-1, 4))
    x1y1 = gt_boxes_xyxy[:, :2].min(dim=0).values
    x2y2 = gt_boxes_xyxy[:, 2:].max(dim=0).values
    tight_xyxy = torch.cat((x1y1, x2y2), dim=0).clamp(0.0, 1.0)
    return box_xyxy_to_cxcywh(tight_xyxy)


def expand_box_to_loose(box_cxcywh, expand_ratio=0.25):
    """Expand a normalized tight box by 25% on each side and clamp to image bounds."""
    if box_cxcywh is None:
        return None
    box_xyxy = box_cxcywh_to_xyxy(box_cxcywh.view(1, 4))[0]
    x1, y1, x2, y2 = box_xyxy.tolist()
    width = x2 - x1
    height = y2 - y1
    loose_xyxy = torch.tensor(
        [
            x1 - expand_ratio * width,
            y1 - expand_ratio * height,
            x2 + expand_ratio * width,
            y2 + expand_ratio * height,
        ],
        dtype=box_cxcywh.dtype,
        device=box_cxcywh.device,
    ).clamp(0.0, 1.0)
    return box_xyxy_to_cxcywh(loose_xyxy)


def build_input_box_prompts(find_targets, box_mode):
    """Create query-level box prompts for one validation box mode."""
    num_queries = int(find_targets.num_boxes.shape[0])
    device = find_targets.num_boxes.device
    box_dtype = find_targets.boxes_padded.dtype

    if box_mode == "text_only":
        return (
            torch.zeros((0, num_queries, 4), dtype=box_dtype, device=device),
            torch.ones((num_queries, 0), dtype=torch.bool, device=device),
            torch.zeros((0, num_queries), dtype=torch.long, device=device),
        )

    input_boxes = torch.zeros((1, num_queries, 4), dtype=box_dtype, device=device)
    input_boxes_mask = torch.ones((num_queries, 1), dtype=torch.bool, device=device)
    input_boxes_label = torch.zeros((1, num_queries), dtype=torch.long, device=device)
    num_boxes_list = find_targets.num_boxes.detach().cpu().tolist()

    for query_idx, num_gt_boxes in enumerate(num_boxes_list):
        if int(num_gt_boxes) <= 0:
            continue
        tight_box = compute_tight_box_from_gt_boxes(
            find_targets.boxes_padded[query_idx, : int(num_gt_boxes)]
        )
        if tight_box is None:
            continue
        prompt_box = tight_box
        if box_mode == "text_plus_loose_box":
            prompt_box = expand_box_to_loose(tight_box, expand_ratio=0.25)
        elif box_mode != "text_plus_tight_box":
            raise ValueError(f"Unsupported box_mode: {box_mode}")

        input_boxes[0, query_idx] = prompt_box
        input_boxes_mask[query_idx, 0] = False
        input_boxes_label[0, query_idx] = 1

    return input_boxes, input_boxes_mask, input_boxes_label


@contextlib.contextmanager
def temporary_box_prompt_mode(input_batch, box_mode):
    """Temporarily override validation query box prompts for one forward pass."""
    stage_input = input_batch.find_inputs[0]
    original_input_boxes = stage_input.input_boxes
    original_input_boxes_mask = stage_input.input_boxes_mask
    original_input_boxes_label = stage_input.input_boxes_label

    new_boxes, new_mask, new_label = build_input_box_prompts(input_batch.find_targets[0], box_mode)
    stage_input.input_boxes = new_boxes
    stage_input.input_boxes_mask = new_mask
    stage_input.input_boxes_label = new_label

    try:
        yield
    finally:
        stage_input.input_boxes = original_input_boxes
        stage_input.input_boxes_mask = original_input_boxes_mask
        stage_input.input_boxes_label = original_input_boxes_label

class COCOSegmentDataset(Dataset):
    """Dataset class for COCO format segmentation data"""
    def __init__(
        self,
        data_dir,
        split="train",
        prompt_manager=None,
        prompt_mode="canonical",
        multi_prompt_group=None,
        lexical_variations_by_canonical=None,
        challenging_by_canonical=None,
    ):
        """
        Args:
            data_dir: Root directory containing train/valid/test folders
            split: One of 'train', 'valid', 'test'
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir / split
        self.prompt_manager = prompt_manager
        self.prompt_mode = prompt_mode
        self.multi_prompt_group = multi_prompt_group
        self.lexical_variations_by_canonical = lexical_variations_by_canonical or {}
        self.challenging_by_canonical = challenging_by_canonical or {}
        self.warned_missing_lexical_categories = set()
        self.warned_missing_challenging_categories = set()

        # Load COCO annotations
        ann_file = self.split_dir / "_annotations.coco.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")

        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)

        # Build index: image_id -> image info
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.image_ids = sorted(list(self.images.keys()))

        # Build index: image_id -> list of annotations
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        # Load categories
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        print(f"Loaded COCO dataset: {split} split")
        print(f"  Images: {len(self.image_ids)}")
        print(f"  Annotations: {len(self.coco_data['annotations'])}")
        print(f"  Categories: {self.categories}")

        self.resolution = 1008
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.image_ids)

    def _resolve_query_text(self, category_name):
        return self._resolve_query_texts(category_name)[0]

    def _resolve_query_texts(self, category_name):
        canonical_prompt = category_name.lower()
        if self.prompt_manager is not None:
            canonical_prompt = self.prompt_manager.get_canonical_prompt(category_name)

        if self.multi_prompt_group == "canonical":
            return [canonical_prompt]

        if self.multi_prompt_group == "synonyms":
            if self.prompt_manager is None:
                print(
                    "[Prompt Warning] multi_prompt_group=synonyms requested but prompts config "
                    "is unavailable. Falling back to canonical queries."
                )
                return [canonical_prompt]

            synonym_prompts = self.prompt_manager.synonyms_by_canonical.get(canonical_prompt, [])
            if not synonym_prompts:
                self.prompt_manager.warn_missing_category(category_name)
                return [canonical_prompt]
            return list(synonym_prompts)

        if self.multi_prompt_group == "lexical_variations":
            lexical_prompts = self.lexical_variations_by_canonical.get(canonical_prompt, [])
            if not lexical_prompts:
                if canonical_prompt not in self.warned_missing_lexical_categories:
                    self.warned_missing_lexical_categories.add(canonical_prompt)
                    print(
                        f"[Prompt Warning] Missing lexical_variations config for category "
                        f"'{canonical_prompt}'. Falling back to canonical prompt only."
                    )
                return [canonical_prompt]
            return list(lexical_prompts)

        if self.multi_prompt_group == "challenging":
            challenging_prompts = self.challenging_by_canonical.get(canonical_prompt, [])
            if not challenging_prompts:
                if canonical_prompt not in self.warned_missing_challenging_categories:
                    self.warned_missing_challenging_categories.add(canonical_prompt)
                    print(
                        f"[Prompt Warning] Missing challenging config for category "
                        f"'{canonical_prompt}'. Falling back to canonical prompt only."
                    )
                return [canonical_prompt]
            return list(challenging_prompts)

        if self.prompt_manager is None:
            return [canonical_prompt]

        if self.prompt_mode == "synonym":
            return [self.prompt_manager.sample_synonym_prompt(category_name, allow_fallback=True)]
        return [canonical_prompt]

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        # Load image
        img_path = self.split_dir / img_info['file_name']
        pil_image = PILImage.open(img_path).convert("RGB")
        orig_w, orig_h = pil_image.size

        # Resize image
        pil_image = pil_image.resize((self.resolution, self.resolution), PILImage.BILINEAR)

        # Transform to tensor
        image_tensor = self.transform(pil_image)

        # Get annotations for this image
        annotations = self.img_to_anns.get(img_id, [])

        objects = []
        object_class_names = []
        object_category_ids = []

        # Scale factors
        scale_w = self.resolution / orig_w
        scale_h = self.resolution / orig_h

        for i, ann in enumerate(annotations):
            # Get bbox - format is [x, y, width, height] in COCO format
            bbox_coco = ann.get("bbox", None)
            if bbox_coco is None:
                continue

            # Get class name from category_id
            category_id = ann.get("category_id", 0)
            class_name = self.categories.get(category_id, "object")
            object_class_names.append(class_name)
            object_category_ids.append(category_id)

            # Convert from COCO [x, y, w, h] to normalized [cx, cy, w, h] (CxCyWH)
            # This matches train_sam3_lora_native.py and SAM3's internal convention.
            x, y, w, h = bbox_coco
            cx = x + w / 2.0
            cy = y + h / 2.0

            # Scale to resolution and normalize to [0, 1]
            box_tensor = torch.tensor([
                cx * scale_w / self.resolution,
                cy * scale_h / self.resolution,
                w * scale_w / self.resolution,
                h * scale_h / self.resolution,
            ], dtype=torch.float32)

            # Handle segmentation mask (polygon or RLE format)
            segment = None
            segmentation = ann.get("segmentation", None)

            if segmentation:
                try:
                    mask_np = decode_coco_segmentation(segmentation, orig_h, orig_w)

                    # Resize mask to model resolution
                    mask_t = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
                    mask_t = torch.nn.functional.interpolate(
                        mask_t,
                        size=(self.resolution, self.resolution),
                        mode="nearest"
                    )
                    segment = mask_t.squeeze() > 0.5  # [1008, 1008] boolean tensor

                except Exception as e:
                    print(f"Warning: Error processing mask for image {img_id}, ann {i}: {e}")
                    segment = None

            obj = Object(
                bbox=box_tensor,
                area=(box_tensor[2] * box_tensor[3]).item(),
                object_id=i,
                segment=segment
            )
            objects.append(obj)

        image_obj = Image(
            data=image_tensor,
            objects=objects,
            size=(self.resolution, self.resolution)
        )

        # Construct queries per category. In multi-prompt mode, each category may emit
        # multiple text queries that all point to the same GT objects.
        class_to_query_info = defaultdict(
            lambda: {"object_ids": [], "category_id": 0, "category_name": ""}
        )
        for obj, class_name, category_id in zip(objects, object_class_names, object_category_ids):
            class_key = class_name.lower()
            class_to_query_info[class_key]["object_ids"].append(obj.object_id)
            class_to_query_info[class_key]["category_id"] = category_id
            class_to_query_info[class_key]["category_name"] = class_name

        # Create one query per category
        queries = []
        if len(class_to_query_info) > 0:
            for query_info in class_to_query_info.values():
                obj_ids = query_info["object_ids"]
                category_id = query_info["category_id"]
                category_name = query_info["category_name"]

                for query_text in self._resolve_query_texts(category_name):
                    query = FindQueryLoaded(
                        query_text=query_text,
                        image_id=0,
                        object_ids_output=obj_ids,
                        is_exhaustive=True,
                        query_processing_order=0,
                        inference_metadata=InferenceMetadata(
                            coco_image_id=img_id,
                            original_image_id=img_id,
                            original_category_id=category_id,
                            original_size=(orig_h, orig_w),
                            object_id=-1,
                            frame_index=-1
                        )
                    )
                    queries.append(query)
        else:
            # No annotations: create a single generic query
            query = FindQueryLoaded(
                query_text="object",
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=img_id,
                    original_image_id=img_id,
                    original_category_id=0,
                    original_size=(orig_h, orig_w),
                    object_id=-1,
                    frame_index=-1
                )
            )
            queries.append(query)

        return Datapoint(
            find_queries=queries,
            images=[image_obj],
            raw_images=[pil_image]
        )


def merge_overlapping_masks(binary_masks, scores, boxes, iou_threshold=0.15):
    """
    Merge overlapping masks that likely represent the same object (e.g., crack segments).

    This is more aggressive than NMS - it MERGES masks instead of suppressing them.
    Useful for cracks where model splits one crack into many segments.

    Args:
        binary_masks: Binary masks [N, H, W]
        scores: Confidence scores [N]
        boxes: Bounding boxes [N, 4]
        iou_threshold: IoU threshold for merging (default: 0.15, lower = more aggressive)

    Returns:
        Tuple of (merged_masks, merged_scores, merged_boxes)
    """
    if len(binary_masks) == 0:
        return binary_masks, scores, boxes

    # Sort by score (highest first)
    sorted_indices = torch.argsort(scores, descending=True)
    binary_masks = binary_masks[sorted_indices]
    scores = scores[sorted_indices]
    boxes = boxes[sorted_indices]

    merged_masks = []
    merged_scores = []
    merged_boxes = []
    used = torch.zeros(len(binary_masks), dtype=torch.bool)

    for i in range(len(binary_masks)):
        if used[i]:
            continue

        current_mask = binary_masks[i].clone()
        current_score = scores[i].item()
        current_box = boxes[i]
        used[i] = True

        # Find overlapping masks and merge them
        for j in range(i + 1, len(binary_masks)):
            if used[j]:
                continue

            # Compute IoU
            intersection = (current_mask & binary_masks[j]).sum().item()
            union = (current_mask | binary_masks[j]).sum().item()
            iou = intersection / union if union > 0 else 0

            # If overlaps significantly, merge it
            if iou > iou_threshold:
                current_mask = current_mask | binary_masks[j]
                current_score = max(current_score, scores[j].item())
                used[j] = True

        merged_masks.append(current_mask)
        merged_scores.append(current_score)
        merged_boxes.append(current_box)

    if len(merged_masks) > 0:
        merged_masks = torch.stack(merged_masks)
        merged_scores = torch.tensor(merged_scores, device=scores.device)
        merged_boxes = torch.stack(merged_boxes)
    else:
        merged_masks = binary_masks[:0]
        merged_scores = scores[:0]
        merged_boxes = boxes[:0]

    return merged_masks, merged_scores, merged_boxes


def apply_sam3_nms(pred_logits, pred_masks, pred_boxes, prob_threshold=0.3, nms_iou_threshold=0.7, max_detections=100):
    """
    Apply SAM3's standard NMS pipeline to filter predictions.

    Args:
        pred_logits: [N, 1] logits
        pred_masks: [N, H, W] mask logits
        pred_boxes: [N, 4] boxes in normalized [cx, cy, w, h] format
        prob_threshold: Score threshold for filtering (default: 0.3, SAM3 uses 0.5)
        nms_iou_threshold: IoU threshold for NMS (default: 0.7, SAM3 uses 0.5-0.7)
        max_detections: Maximum detections to keep (default: 100)

    Returns:
        Tuple of (filtered_masks, filtered_scores, filtered_boxes)
    """
    if len(pred_logits) == 0:
        return pred_masks[:0], pred_logits[:0].squeeze(-1), pred_boxes[:0]

    # Convert logits to probabilities
    pred_probs = torch.sigmoid(pred_logits).squeeze(-1)  # [N]

    # Convert mask logits to binary masks (sigmoid + threshold)
    pred_masks_sigmoid = torch.sigmoid(pred_masks)  # [N, H, W]
    pred_masks_binary = pred_masks_sigmoid > 0.5  # [N, H, W]

    # Apply SAM3's NMS
    # nms_masks expects: pred_probs [N], pred_masks [N, H, W], prob_threshold, iou_threshold
    # Returns: keep mask [N] of booleans
    keep_mask = nms_masks(
        pred_probs=pred_probs,
        pred_masks=pred_masks_binary.float(),  # NMS expects float masks
        prob_threshold=prob_threshold,
        iou_threshold=nms_iou_threshold
    )

    # Filter predictions
    filtered_masks = pred_masks_sigmoid[keep_mask]  # Keep sigmoid masks for later
    filtered_scores = pred_probs[keep_mask]
    filtered_boxes = pred_boxes[keep_mask]

    # Top-K selection by score
    if max_detections > 0 and len(filtered_scores) > max_detections:
        top_k_scores, top_k_indices = torch.topk(filtered_scores, k=max_detections, largest=True)
        filtered_masks = filtered_masks[top_k_indices]
        filtered_scores = top_k_scores
        filtered_boxes = filtered_boxes[top_k_indices]

    return filtered_masks, filtered_scores, filtered_boxes


def filter_predictions_for_binary_metrics(pred_logits, pred_masks, pred_boxes,
                                          prob_threshold=0.3, nms_iou_threshold=0.7, max_detections=100,
                                          merge_cracks=False, merge_iou_threshold=0.15):
    """
    Filter a single-query prediction using the same logic as COCO conversion,
    then return binary masks for query-level Dice/IoU computation.
    """
    if len(pred_logits) == 0:
        return pred_masks[:0]

    if merge_cracks:
        pred_probs = torch.sigmoid(pred_logits).squeeze(-1)
        valid_mask = pred_probs > prob_threshold

        filtered_masks = pred_masks[valid_mask]
        filtered_scores = pred_probs[valid_mask]
        filtered_boxes = pred_boxes[valid_mask]

        if len(filtered_masks) == 0:
            return pred_masks[:0]

        pred_masks_sigmoid = torch.sigmoid(filtered_masks)
        pred_masks_binary = pred_masks_sigmoid > 0.5

        merged_masks, merged_scores, _ = merge_overlapping_masks(
            pred_masks_binary.cpu(),
            filtered_scores.cpu(),
            filtered_boxes.cpu(),
            iou_threshold=merge_iou_threshold
        )

        if max_detections > 0 and len(merged_scores) > max_detections:
            _, top_k_indices = torch.topk(merged_scores, k=max_detections, largest=True)
            merged_masks = merged_masks[top_k_indices]

        return merged_masks.bool()

    filtered_masks, _, _ = apply_sam3_nms(
        pred_logits=pred_logits,
        pred_masks=pred_masks,
        pred_boxes=pred_boxes,
        prob_threshold=prob_threshold,
        nms_iou_threshold=nms_iou_threshold,
        max_detections=max_detections
    )
    return (filtered_masks > 0.5).cpu()


def convert_predictions_to_coco_format(predictions_list, image_ids, resolution=288,
                                       prob_threshold=0.3, nms_iou_threshold=0.7, max_detections=100,
                                       merge_cracks=False, merge_iou_threshold=0.15):
    """
    Convert model predictions to COCO format using SAM3's NMS pipeline.

    Args:
        predictions_list: List of predictions per image
        image_ids: List of image IDs
        resolution: Resolution for box scaling (default: 288)
        prob_threshold: Score threshold (default: 0.3, SAM3 uses 0.5)
        nms_iou_threshold: NMS IoU threshold (default: 0.7)
        max_detections: Max detections per image (default: 100)
        merge_cracks: If True, merge overlapping segments instead of NMS suppression (default: False)
        merge_iou_threshold: IoU threshold for merging (default: 0.15, lower = more aggressive)
    """
    coco_predictions = []
    pred_id = 0

    if merge_cracks:
        print(f"\n[INFO] Converting {len(predictions_list)} predictions to COCO format...")
        print(f"[INFO] Using CRACK MERGING mode: prob_threshold={prob_threshold}, merge_iou={merge_iou_threshold}, max_dets={max_detections}")
        print(f"[INFO] This will MERGE overlapping crack segments instead of suppressing them")
    else:
        print(f"\n[INFO] Converting {len(predictions_list)} predictions to COCO format...")
        print(f"[INFO] Using SAM3 NMS: prob_threshold={prob_threshold}, nms_iou={nms_iou_threshold}, max_dets={max_detections}")

    for img_id, preds in tqdm(zip(image_ids, predictions_list), total=len(predictions_list), desc="Converting predictions"):
        if preds is None or len(preds.get('pred_logits', [])) == 0:
            continue

        logits = preds['pred_logits']  # [N, 1]
        boxes = preds['pred_boxes']    # [N, 4]
        masks = preds['pred_masks']    # [N, H, W]

        if merge_cracks:
            # Step 1: Filter by score threshold
            pred_probs = torch.sigmoid(logits).squeeze(-1)  # [N]
            valid_mask = pred_probs > prob_threshold

            filtered_masks = masks[valid_mask]
            filtered_scores = pred_probs[valid_mask]
            filtered_boxes = boxes[valid_mask]

            if len(filtered_masks) > 0:
                # Step 2: Convert masks to binary
                pred_masks_sigmoid = torch.sigmoid(filtered_masks)
                pred_masks_binary = (pred_masks_sigmoid > 0.5)

                # Step 3: MERGE overlapping crack segments
                merged_masks, merged_scores, merged_boxes = merge_overlapping_masks(
                    pred_masks_binary.cpu(),
                    filtered_scores.cpu(),
                    filtered_boxes.cpu(),
                    iou_threshold=merge_iou_threshold
                )

                # Step 4: Top-K selection by score
                if max_detections > 0 and len(merged_scores) > max_detections:
                    top_k_scores, top_k_indices = torch.topk(merged_scores, k=max_detections, largest=True)
                    merged_masks = merged_masks[top_k_indices]
                    merged_scores = top_k_scores
                    merged_boxes = merged_boxes[top_k_indices]

                # Return merged results (already binary)
                filtered_masks = merged_masks.float()  # Already binary, just convert to float
                filtered_scores = merged_scores
                filtered_boxes = merged_boxes
            else:
                filtered_masks = torch.tensor([])
                filtered_scores = torch.tensor([])
                filtered_boxes = torch.tensor([])
        else:
            # Apply SAM3's NMS pipeline (standard suppression)
            filtered_masks, filtered_scores, filtered_boxes = apply_sam3_nms(
                pred_logits=logits,
                pred_masks=masks,
                pred_boxes=boxes,
                prob_threshold=prob_threshold,
                nms_iou_threshold=nms_iou_threshold,
                max_detections=max_detections
            )

        if len(filtered_masks) > 0:
            # Convert filtered masks to binary for RLE encoding
            binary_masks = (filtered_masks > 0.5).cpu()
            rles = rle_encode(binary_masks)

            for idx, (rle, score, box) in enumerate(zip(rles, filtered_scores.cpu().tolist(), filtered_boxes.cpu().tolist())):
                cx, cy, w, h = box
                x = (cx - w/2) * resolution
                y = (cy - h/2) * resolution
                w = w * resolution
                h = h * resolution

                pred_dict = {
                    'image_id': int(img_id),
                    'category_id': 1,
                    'segmentation': rle,
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'score': float(score),
                    'id': pred_id
                }

                coco_predictions.append(pred_dict)
                pred_id += 1

    return coco_predictions


def create_coco_gt_from_dataset(dataset, image_ids=None, mask_resolution=288):
    """
    Create COCO ground truth dictionary from dataset.

    OPTIMIZATION: Downsample GT masks to 288×288 to match prediction resolution.
    """
    print(f"\n[INFO] Creating COCO ground truth (downsampling to {mask_resolution}×{mask_resolution})...")

    coco_gt = {
        'info': {
            'description': 'SAM3 LoRA Validation Dataset',
            'version': '1.0',
            'year': 2024
        },
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'object'}]
    }

    ann_id = 0
    indices = range(len(dataset)) if image_ids is None else image_ids

    for idx in tqdm(list(indices), desc="Creating GT"):
        coco_gt['images'].append({
            'id': int(idx),
            'width': mask_resolution,
            'height': mask_resolution,
            'is_instance_exhaustive': True
        })

        datapoint = dataset[idx]

        for obj in datapoint.images[0].objects:
            # obj.bbox is normalized [cx, cy, w, h]; convert to COCO [x, y, w, h]
            cx, cy, bw, bh = (obj.bbox * mask_resolution).tolist()
            x, y, w, h = cx - bw / 2, cy - bh / 2, bw, bh

            ann = {
                'id': ann_id,
                'image_id': int(idx),
                'category_id': 1,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'ignore': 0
            }

            if obj.segment is not None:
                # Downsample mask from 1008×1008 to mask_resolution×mask_resolution
                mask_tensor = obj.segment.unsqueeze(0).unsqueeze(0).float()
                downsampled_mask = torch.nn.functional.interpolate(
                    mask_tensor,
                    size=(mask_resolution, mask_resolution),
                    mode='bilinear',
                    align_corners=False
                ) > 0.5

                mask_np = downsampled_mask.squeeze().cpu().numpy().astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_np))
                rle['counts'] = rle['counts'].decode('utf-8')
                ann['segmentation'] = rle

            coco_gt['annotations'].append(ann)
            ann_id += 1

    print(f"[INFO] Created {len(coco_gt['images'])} images, {len(coco_gt['annotations'])} annotations")

    return coco_gt


def _build_union_mask(annotations, image_height, image_width):
    """Merge all instance masks for one image into a single binary mask."""
    union_mask = np.zeros((image_height, image_width), dtype=bool)

    for ann in annotations:
        segmentation = ann.get("segmentation")
        if segmentation is None:
            continue

        mask = mask_utils.decode(segmentation)
        if mask.ndim == 3:
            mask = np.any(mask, axis=2)
        union_mask |= mask.astype(bool)

    return union_mask


def compute_binary_mask_metrics(coco_gt_dict, coco_predictions):
    """
    Compute image-level binary segmentation Dice/IoU by merging all masks per image.

    This is more appropriate than cgF1 for single-class medical segmentation datasets
    such as ISIC, where image-level negative samples may be absent.
    """
    gt_images = {img["id"]: img for img in coco_gt_dict["images"]}

    gt_by_image = {}
    for ann in coco_gt_dict["annotations"]:
        gt_by_image.setdefault(ann["image_id"], []).append(ann)

    pred_by_image = {}
    for ann in coco_predictions:
        pred_by_image.setdefault(ann["image_id"], []).append(ann)

    dice_scores = []
    iou_scores = []

    total_intersection = 0
    total_gt = 0
    total_pred = 0
    total_union = 0

    for image_id, image_info in gt_images.items():
        height = int(image_info["height"])
        width = int(image_info["width"])

        gt_mask = _build_union_mask(gt_by_image.get(image_id, []), height, width)
        pred_mask = _build_union_mask(pred_by_image.get(image_id, []), height, width)

        intersection = int(np.logical_and(gt_mask, pred_mask).sum())
        union = int(np.logical_or(gt_mask, pred_mask).sum())
        gt_area = int(gt_mask.sum())
        pred_area = int(pred_mask.sum())

        if gt_area == 0 and pred_area == 0:
            dice = 1.0
            iou = 1.0
        else:
            dice = (2.0 * intersection) / max(gt_area + pred_area, 1)
            iou = intersection / max(union, 1)

        dice_scores.append(dice)
        iou_scores.append(iou)

        total_intersection += intersection
        total_gt += gt_area
        total_pred += pred_area
        total_union += union

    global_dice = (
        (2.0 * total_intersection) / max(total_gt + total_pred, 1)
        if (total_gt > 0 or total_pred > 0)
        else 1.0
    )
    global_iou = (
        total_intersection / max(total_union, 1)
        if total_union > 0
        else 1.0
    )

    return {
        "mean_dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "mean_iou": float(np.mean(iou_scores)) if iou_scores else 0.0,
        "global_dice": float(global_dice),
        "global_iou": float(global_iou),
        "num_images": len(gt_images),
    }


def compute_binary_scores_from_masks(gt_mask, pred_mask):
    """Compute Dice/IoU for a pair of binary masks."""
    intersection = int(np.logical_and(gt_mask, pred_mask).sum())
    union = int(np.logical_or(gt_mask, pred_mask).sum())
    gt_area = int(gt_mask.sum())
    pred_area = int(pred_mask.sum())

    if gt_area == 0 and pred_area == 0:
        return 1.0, 1.0

    dice = (2.0 * intersection) / max(gt_area + pred_area, 1)
    iou = intersection / max(union, 1)
    return float(dice), float(iou)


def build_union_mask_from_tensor_masks(mask_tensor, target_size=(288, 288)):
    """Build a single binary union mask from an instance-mask tensor."""
    if mask_tensor is None or len(mask_tensor) == 0:
        return np.zeros(target_size, dtype=bool)

    masks = mask_tensor.float()
    if tuple(masks.shape[-2:]) != tuple(target_size):
        masks = torch.nn.functional.interpolate(
            masks.unsqueeze(1),
            size=target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(1)

    union_mask = (masks > 0.5).any(dim=0).cpu().numpy().astype(bool)
    return union_mask


def save_per_image_metrics_csv(records, csv_path):
    """Save per-query image-level binary metrics to CSV."""
    fieldnames = [
        "image_id",
        "category_id",
        "canonical_prompt",
        "prompt_group",
        "prompt_mode",
        "query_text",
        "dice",
        "iou",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "image_id": record["image_id"],
                "category_id": record["category_id"],
                "canonical_prompt": record["canonical_prompt"],
                "prompt_group": record["prompt_group"],
                "prompt_mode": record["prompt_mode"],
                "query_text": record["query_text"],
                "dice": f"{record['dice']:.6f}",
                "iou": f"{record['iou']:.6f}",
            })


def aggregate_robustness_records(records):
    """Aggregate prompt-wise records into per-image robustness metrics."""
    grouped_records = defaultdict(list)
    for record in records:
        group_key = (
            record["image_id"],
            record["category_id"],
            record["canonical_prompt"],
            record["prompt_group"],
        )
        grouped_records[group_key].append(record)

    robustness_records = []
    for (image_id, category_id, canonical_prompt, prompt_group), group_records in grouped_records.items():
        sorted_by_dice = sorted(group_records, key=lambda item: item["dice"])
        worst_record = sorted_by_dice[0]
        best_record = sorted_by_dice[-1]

        robustness_records.append({
            "image_id": image_id,
            "category_id": category_id,
            "canonical_prompt": canonical_prompt,
            "prompt_group": prompt_group,
            "num_prompts": len(group_records),
            "wpd": float(worst_record["dice"]),
            "best_prompt_dice": float(best_record["dice"]),
            "prg": float(best_record["dice"] - worst_record["dice"]),
            "worst_query_text": worst_record["query_text"],
            "best_query_text": best_record["query_text"],
        })

    robustness_records.sort(
        key=lambda item: (item["image_id"], item["category_id"], item["prompt_group"])
    )
    return robustness_records


def save_robustness_metrics_csv(records, csv_path):
    """Save per-image robustness metrics to CSV."""
    fieldnames = [
        "image_id",
        "category_id",
        "canonical_prompt",
        "prompt_group",
        "num_prompts",
        "wpd",
        "best_prompt_dice",
        "prg",
        "worst_query_text",
        "best_query_text",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "image_id": record["image_id"],
                "category_id": record["category_id"],
                "canonical_prompt": record["canonical_prompt"],
                "prompt_group": record["prompt_group"],
                "num_prompts": record["num_prompts"],
                "wpd": f"{record['wpd']:.6f}",
                "best_prompt_dice": f"{record['best_prompt_dice']:.6f}",
                "prg": f"{record['prg']:.6f}",
                "worst_query_text": record["worst_query_text"],
                "best_query_text": record["best_query_text"],
            })


def aggregate_box_dependency_records(records):
    """Aggregate per-mode box dependency runs into per-image BRG records."""
    grouped_records = defaultdict(dict)
    mode_to_field = {
        "text_only": "text_only_dice",
        "text_plus_loose_box": "loose_box_dice",
        "text_plus_tight_box": "tight_box_dice",
    }

    for record in records:
        group_key = (
            record["image_id"],
            record["category_id"],
            record["canonical_prompt"],
            record["prompt_group"],
            record["prompt_mode"],
            record["query_text"],
        )
        grouped_records[group_key][record["box_mode"]] = record["dice"]

    aggregated_records = []
    for (
        image_id,
        category_id,
        canonical_prompt,
        prompt_group,
        prompt_mode,
        query_text,
    ), mode_scores in grouped_records.items():
        aggregated_record = {
            "image_id": image_id,
            "category_id": category_id,
            "canonical_prompt": canonical_prompt,
            "prompt_group": prompt_group,
            "prompt_mode": prompt_mode,
            "query_text": query_text,
            "text_only_dice": None,
            "loose_box_dice": None,
            "tight_box_dice": None,
            "brg": None,
        }
        for box_mode, dice in mode_scores.items():
            field_name = mode_to_field.get(box_mode)
            if field_name is not None:
                aggregated_record[field_name] = float(dice)
        if (
            aggregated_record["text_only_dice"] is not None
            and aggregated_record["tight_box_dice"] is not None
        ):
            aggregated_record["brg"] = (
                aggregated_record["tight_box_dice"] - aggregated_record["text_only_dice"]
            )
        aggregated_records.append(aggregated_record)

    return sorted(
        aggregated_records,
        key=lambda item: (
            item["image_id"],
            item["category_id"],
            item["prompt_group"],
            item["query_text"],
        ),
    )


def save_box_dependency_csv(records, csv_path):
    """Save per-image box dependency metrics and BRG to CSV."""
    fieldnames = [
        "image_id",
        "category_id",
        "canonical_prompt",
        "prompt_group",
        "prompt_mode",
        "query_text",
        "text_only_dice",
        "loose_box_dice",
        "tight_box_dice",
        "brg",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "image_id": record["image_id"],
                "category_id": record["category_id"],
                "canonical_prompt": record["canonical_prompt"],
                "prompt_group": record["prompt_group"],
                "prompt_mode": record["prompt_mode"],
                "query_text": record["query_text"],
                "text_only_dice": "" if record["text_only_dice"] is None else f"{record['text_only_dice']:.6f}",
                "loose_box_dice": "" if record["loose_box_dice"] is None else f"{record['loose_box_dice']:.6f}",
                "tight_box_dice": "" if record["tight_box_dice"] is None else f"{record['tight_box_dice']:.6f}",
                "brg": "" if record["brg"] is None else f"{record['brg']:.6f}",
            })


def load_robustness_metrics_csv(csv_path):
    """Load robustness CSV for visualization ranking."""
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "image_id": int(row["image_id"]),
                "category_id": int(row["category_id"]),
                "canonical_prompt": row["canonical_prompt"],
                "prompt_group": row["prompt_group"],
                "num_prompts": int(row["num_prompts"]),
                "wpd": float(row["wpd"]),
                "best_prompt_dice": float(row["best_prompt_dice"]),
                "prg": float(row["prg"]),
                "worst_query_text": row["worst_query_text"],
                "best_query_text": row["best_query_text"],
            })
    return records


def resize_binary_mask(mask_np, target_size):
    """Resize a binary mask to (width, height) using nearest-neighbor."""
    mask_img = PILImage.fromarray(mask_np.astype(np.uint8) * 255)
    return np.array(mask_img.resize(target_size, PILImage.NEAREST)) > 0


def overlay_mask_on_image(image_np, mask_np, color, alpha=0.45):
    """Overlay a binary mask on an RGB image."""
    overlay = image_np.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32)
    overlay[mask_np] = (1.0 - alpha) * overlay[mask_np] + alpha * color_arr
    return overlay.clip(0, 255).astype(np.uint8)


def sanitize_filename(text):
    """Create a filesystem-safe filename fragment."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return cleaned or "sample"


def load_selected_prompt_sensitivity_cases(csv_path, dataset_name=None):
    """Load selected Figure 1 cases and return rows for the current dataset."""
    if not csv_path:
        return []
    path = Path(csv_path)
    if not path.exists():
        print(f"[PredMask] Warning: selected_cases_csv not found: {path}")
        return []
    normalized_dataset = str(dataset_name or "").lower().replace("_", "-")
    if normalized_dataset == "kvasir":
        normalized_dataset = "sessile-kvasir"
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row_dataset = str(row.get("dataset", "")).lower().replace("_", "-")
            if normalized_dataset and row_dataset and row_dataset != normalized_dataset:
                continue
            rows.append(row)
    return rows


def save_binary_mask_png(mask_np, path, target_size):
    mask_resized = resize_binary_mask(mask_np, target_size)
    PILImage.fromarray(mask_resized.astype(np.uint8) * 255).save(path)


def export_selected_prompt_sensitivity_masks(
    dataset,
    per_image_records,
    selected_cases_csv,
    output_dir,
    dataset_name,
    prompt_group,
):
    """Export original image, GT, and selected prompt prediction masks for Figure 1."""
    if not selected_cases_csv or not output_dir:
        return
    prompt_group = prompt_group or "canonical"
    role_by_group = {
        "canonical": (("canonical_prompt",), "pred_canonical.png"),
        "synonyms": (("selected_synonym_prompt", "synonym_prompt", "worst_synonym_prompt"), "pred_synonym.png"),
        "challenging": (("selected_challenging_prompt", "challenging_prompt", "worst_challenging_prompt"), "pred_challenging.png"),
    }
    if prompt_group not in role_by_group:
        return

    selected_rows = load_selected_prompt_sensitivity_cases(selected_cases_csv, dataset_name)
    if not selected_rows:
        print(f"[PredMask] No selected cases for dataset={dataset_name}.")
        return
    selected_rows = sorted(
        selected_rows,
        key=lambda row: int(row.get("selected_rank", row.get("rank", 1)) or 1),
    )[:1]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_fields, pred_filename = role_by_group[prompt_group]
    records_by_key = {
        (str(record["image_id"]), str(record["query_text"])): record
        for record in per_image_records
    }

    exported = 0
    for row in selected_rows:
        image_id = str(row.get("image_id", ""))
        prompt = next((row.get(field, "") for field in prompt_fields if row.get(field, "")), "")
        if not image_id or not prompt:
            continue
        record = records_by_key.get((image_id, prompt))
        if record is None:
            print(
                f"[PredMask] Warning: selected prediction not found: "
                f"dataset={dataset_name} image_id={image_id} group={prompt_group} prompt={prompt}"
            )
            continue

        dataset_index = record["dataset_index"]
        img_id = dataset.image_ids[dataset_index]
        img_info = dataset.images[img_id]
        image_path = dataset.split_dir / img_info["file_name"]
        original_image = PILImage.open(image_path).convert("RGB")
        target_size = (original_image.width, original_image.height)

        image_out = output_dir / "image.png"
        gt_out = output_dir / "gt.png"
        original_image.save(image_out)
        save_binary_mask_png(record["gt_union_mask"], gt_out, target_size)
        save_binary_mask_png(record["pred_union_mask"], output_dir / pred_filename, target_size)
        metadata_path = output_dir / f"metadata_{prompt_group}.csv"
        with metadata_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["dataset", "image_id", "prompt_group", "query_text", "dice", "iou", "file"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "dataset": dataset_name,
                    "image_id": image_id,
                    "prompt_group": prompt_group,
                    "query_text": prompt,
                    "dice": f"{record['dice']:.6f}",
                    "iou": f"{record['iou']:.6f}",
                    "file": pred_filename,
                }
            )
        exported += 1

    print(f"[PredMask] Exported {exported} selected {prompt_group} mask(s) to: {output_dir.resolve()}")


def export_topk_visualizations(dataset, per_image_records, robustness_records, output_dir, topk):
    """Export top-k robustness visualizations using current validation results."""
    if not per_image_records or not robustness_records:
        print("[Vis] No records available for visualization export.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records_by_group = defaultdict(list)
    for record in per_image_records:
        group_key = (
            record["image_id"],
            record["category_id"],
            record["canonical_prompt"],
            record["prompt_group"],
        )
        records_by_group[group_key].append(record)

    sorted_robustness = sorted(
        robustness_records,
        key=lambda item: (-item["prg"], item["image_id"], item["category_id"]),
    )

    color_cycle = [
        (255, 99, 71),
        (65, 105, 225),
        (255, 165, 0),
        (138, 43, 226),
        (46, 139, 87),
        (220, 20, 60),
    ]

    exported = 0
    for rank, robustness_record in enumerate(sorted_robustness[:topk], start=1):
        group_key = (
            robustness_record["image_id"],
            robustness_record["category_id"],
            robustness_record["canonical_prompt"],
            robustness_record["prompt_group"],
        )
        prompt_records = records_by_group.get(group_key, [])
        if not prompt_records:
            print(f"[Vis] Warning: no prompt-wise records found for robustness sample {group_key}.")
            continue

        prompt_records = sorted(prompt_records, key=lambda item: item["query_text"])
        dataset_index = prompt_records[0]["dataset_index"]
        img_id = dataset.image_ids[dataset_index]
        img_info = dataset.images[img_id]
        image_path = dataset.split_dir / img_info["file_name"]
        original_image = PILImage.open(image_path).convert("RGB")
        original_np = np.array(original_image)
        target_size = (original_image.width, original_image.height)

        gt_mask = resize_binary_mask(prompt_records[0]["gt_union_mask"], target_size)
        gt_overlay = overlay_mask_on_image(original_np, gt_mask, color=(34, 139, 34), alpha=0.45)

        num_panels = 2 + len(prompt_records)
        fig, axes = plt.subplots(1, num_panels, figsize=(4 * num_panels, 4.5))
        if num_panels == 1:
            axes = [axes]

        axes[0].imshow(original_np)
        axes[0].set_title(f"Image\nimage_id={robustness_record['image_id']}")
        axes[0].axis("off")

        axes[1].imshow(gt_overlay)
        axes[1].set_title(
            f"GT\ncategory={robustness_record['canonical_prompt']}"
        )
        axes[1].axis("off")

        for idx, record in enumerate(prompt_records):
            pred_mask = resize_binary_mask(record["pred_union_mask"], target_size)
            pred_overlay = overlay_mask_on_image(
                original_np,
                pred_mask,
                color=color_cycle[idx % len(color_cycle)],
                alpha=0.45,
            )
            ax = axes[idx + 2]
            ax.imshow(pred_overlay)
            ax.set_title(
                f"{record['query_text']}\nDice={record['dice']:.4f} IoU={record['iou']:.4f}"
            )
            ax.axis("off")

        fig.suptitle(
            f"Rank {rank} | image_id={robustness_record['image_id']} | "
            f"group={robustness_record['prompt_group']} | "
            f"WPD={robustness_record['wpd']:.4f} | "
            f"Best={robustness_record['best_prompt_dice']:.4f} | "
            f"PRG={robustness_record['prg']:.4f}",
            fontsize=12,
        )
        fig.tight_layout()

        filename = (
            f"rank_{rank:03d}_img_{robustness_record['image_id']}"
            f"_cat_{robustness_record['category_id']}"
            f"_{sanitize_filename(robustness_record['prompt_group'])}"
            f"_{sanitize_filename(robustness_record['canonical_prompt'])}.png"
        )
        fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        exported += 1

    print(f"[Vis] Exported {exported} visualization(s) to: {output_dir.resolve()}")


def _select_prompt_records_for_paper(prompt_records):
    """Pick best / middle / worst prompt records by Dice."""
    sorted_records = sorted(prompt_records, key=lambda item: item["dice"])
    worst_record = sorted_records[0]
    best_record = sorted_records[-1]
    middle_record = sorted_records[len(sorted_records) // 2]
    return best_record, middle_record, worst_record


def _select_paper_figure_samples(robustness_records, topk, image_ids=None):
    """Select robustness rows for the paper figure."""
    if image_ids:
        selected = []
        remaining = list(robustness_records)
        for image_id in image_ids:
            match = next((record for record in remaining if record["image_id"] == image_id), None)
            if match is not None:
                selected.append(match)
                remaining.remove(match)
            else:
                print(f"[Paper Figure] Warning: image_id={image_id} not found in robustness records.")
        return selected

    sorted_records = sorted(
        robustness_records,
        key=lambda item: (-item["prg"], item["image_id"], item["category_id"]),
    )
    return sorted_records[:topk]


def compose_paper_figure(
    dataset,
    per_image_records,
    robustness_records,
    output_path,
    topk,
    image_ids=None,
):
    """Compose a paper-style multi-row figure with fixed 5-column layout."""
    if not per_image_records or not robustness_records:
        print("[Paper Figure] No records available for paper figure export.")
        return

    selected_samples = _select_paper_figure_samples(
        robustness_records=robustness_records,
        topk=topk,
        image_ids=image_ids,
    )
    if not selected_samples:
        print("[Paper Figure] No samples selected for paper figure export.")
        return

    records_by_group = defaultdict(list)
    for record in per_image_records:
        group_key = (
            record["image_id"],
            record["category_id"],
            record["canonical_prompt"],
            record["prompt_group"],
        )
        records_by_group[group_key].append(record)

    fig, axes = plt.subplots(
        nrows=len(selected_samples),
        ncols=5,
        figsize=(20, 4.6 * len(selected_samples)),
        dpi=300,
    )
    if len(selected_samples) == 1:
        axes = np.expand_dims(axes, axis=0)

    column_titles = ["Input", "GT", "Best", "Middle", "Worst"]
    for col_idx, title in enumerate(column_titles):
        axes[0, col_idx].set_title(title, fontsize=14, fontweight="bold")

    colors = {
        "gt": (34, 139, 34),
        "best": (65, 105, 225),
        "middle": (255, 165, 0),
        "worst": (220, 20, 60),
    }

    for row_idx, sample in enumerate(selected_samples):
        group_key = (
            sample["image_id"],
            sample["category_id"],
            sample["canonical_prompt"],
            sample["prompt_group"],
        )
        prompt_records = records_by_group.get(group_key, [])
        if not prompt_records:
            print(f"[Paper Figure] Warning: no prompt records found for sample {group_key}.")
            continue

        best_record, middle_record, worst_record = _select_prompt_records_for_paper(prompt_records)
        representative_record = prompt_records[0]
        dataset_index = representative_record["dataset_index"]
        img_id = dataset.image_ids[dataset_index]
        img_info = dataset.images[img_id]
        image_path = dataset.split_dir / img_info["file_name"]

        original_image = PILImage.open(image_path).convert("RGB")
        original_np = np.array(original_image)
        target_size = (original_image.width, original_image.height)

        gt_mask = resize_binary_mask(representative_record["gt_union_mask"], target_size)
        gt_overlay = overlay_mask_on_image(original_np, gt_mask, color=colors["gt"], alpha=0.45)

        best_mask = resize_binary_mask(best_record["pred_union_mask"], target_size)
        middle_mask = resize_binary_mask(middle_record["pred_union_mask"], target_size)
        worst_mask = resize_binary_mask(worst_record["pred_union_mask"], target_size)

        best_overlay = overlay_mask_on_image(original_np, best_mask, color=colors["best"], alpha=0.45)
        middle_overlay = overlay_mask_on_image(original_np, middle_mask, color=colors["middle"], alpha=0.45)
        worst_overlay = overlay_mask_on_image(original_np, worst_mask, color=colors["worst"], alpha=0.45)

        row_axes = axes[row_idx]
        row_axes[0].imshow(original_np)
        row_axes[1].imshow(gt_overlay)
        row_axes[2].imshow(best_overlay)
        row_axes[3].imshow(middle_overlay)
        row_axes[4].imshow(worst_overlay)

        for ax in row_axes:
            ax.axis("off")

        row_axes[2].text(
            0.02, 0.02,
            f"{best_record['query_text']}\nDice={best_record['dice']:.4f}",
            transform=row_axes[2].transAxes,
            fontsize=10,
            color="white",
            bbox=dict(facecolor="black", alpha=0.65, pad=3),
            va="bottom",
        )
        row_axes[3].text(
            0.02, 0.02,
            f"{middle_record['query_text']}\nDice={middle_record['dice']:.4f}",
            transform=row_axes[3].transAxes,
            fontsize=10,
            color="white",
            bbox=dict(facecolor="black", alpha=0.65, pad=3),
            va="bottom",
        )
        row_axes[4].text(
            0.02, 0.02,
            f"{worst_record['query_text']}\nDice={worst_record['dice']:.4f}",
            transform=row_axes[4].transAxes,
            fontsize=10,
            color="white",
            bbox=dict(facecolor="black", alpha=0.65, pad=3),
            va="bottom",
        )

        row_axes[0].text(
            0.0, 1.14,
            f"Sample {row_idx + 1} (ID {sample['image_id']}) | PRG={sample['prg']:.4f} | WPD={sample['wpd']:.4f}",
            transform=row_axes[0].transAxes,
            fontsize=11.5,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(
        left=0.02,
        right=0.995,
        top=0.97,
        bottom=0.03,
        wspace=0.03,
        hspace=0.18,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Paper Figure] Saved paper figure to: {output_path.resolve()}")


def convert_predictions_to_coco_format_original_res(predictions_list, image_ids, dataset, model_resolution=288, score_threshold=0.0, merge_overlaps=True, iou_threshold=0.3, debug=False):
    """
    Convert model predictions to COCO format at ORIGINAL image resolution.

    This matches the inference approach (infer_sam.py) where:
    1. Masks are upsampled from 288x288 to original image size
    2. Boxes are scaled to original image size
    3. Evaluation happens at original resolution

    Args:
        predictions_list: List of predictions per image
        image_ids: List of image IDs (indices into dataset)
        dataset: Dataset to get original image sizes
        model_resolution: Model output resolution (default: 288)
        score_threshold: Confidence threshold
        merge_overlaps: Whether to merge overlapping predictions
        iou_threshold: IoU threshold for merging
        debug: Print debug info
    """
    coco_predictions = []
    pred_id = 0

    if debug:
        print(f"\n[DEBUG] Converting {len(predictions_list)} predictions to COCO format (ORIGINAL RESOLUTION)...")
        if merge_overlaps:
            print(f"[DEBUG] Overlapping segment merging ENABLED (IoU threshold={iou_threshold})")

    for img_id, preds in zip(image_ids, predictions_list):
        if preds is None or len(preds.get('pred_logits', [])) == 0:
            continue

        # Get original image size from dataset
        datapoint = dataset[img_id]
        orig_h, orig_w = datapoint.find_queries[0].inference_metadata.original_size

        logits = preds['pred_logits']
        boxes = preds['pred_boxes']
        masks = preds['pred_masks']  # [N, 288, 288]

        scores = torch.sigmoid(logits).squeeze(-1)

        # Filter by score threshold
        valid_mask = scores > score_threshold
        num_before = len(scores)
        scores = scores[valid_mask]
        boxes = boxes[valid_mask]
        masks = masks[valid_mask]

        if debug and img_id == image_ids[0]:
            print(f"[DEBUG] Image {img_id}: {num_before} queries -> {len(scores)} after filtering (threshold={score_threshold})")
            if len(scores) > 0:
                print(f"[DEBUG]   Original size: {orig_w}x{orig_h}")
                print(f"[DEBUG]   Filtered scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")

        if len(masks) == 0:
            continue

        # Upsample masks from 288x288 to original resolution (like infer_sam.py)
        masks_sigmoid = torch.sigmoid(masks)  # [N, 288, 288]
        masks_upsampled = torch.nn.functional.interpolate(
            masks_sigmoid.unsqueeze(1).float(),  # [N, 1, 288, 288]
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [N, orig_h, orig_w]

        binary_masks = (masks_upsampled > 0.5).cpu()

        # Merge overlapping predictions
        if merge_overlaps and len(binary_masks) > 0:
            num_before_merge = len(binary_masks)
            binary_masks, scores, boxes = merge_overlapping_masks(
                binary_masks, scores.cpu(), boxes.cpu(), iou_threshold=iou_threshold
            )
            if debug and img_id == image_ids[0]:
                print(f"[DEBUG]   Merged {num_before_merge} predictions -> {len(binary_masks)} (IoU threshold={iou_threshold})")

        if len(binary_masks) > 0:
            mask_areas = binary_masks.flatten(1).sum(1)

            if debug and img_id == image_ids[0]:
                print(f"[DEBUG]   Upsampled mask shape: {binary_masks.shape}")
                print(f"[DEBUG]   Mask areas: min={mask_areas.min():.0f}, max={mask_areas.max():.0f}, mean={mask_areas.float().mean():.0f}")

            rles = rle_encode(binary_masks)

            for idx, (rle, score, box) in enumerate(zip(rles, scores.cpu().tolist(), boxes.cpu().tolist())):
                # Convert box from normalized [cx, cy, w, h] to original image coordinates
                cx, cy, w_norm, h_norm = box
                x = (cx - w_norm/2) * orig_w
                y = (cy - h_norm/2) * orig_h
                w = w_norm * orig_w
                h = h_norm * orig_h

                # Clamp coordinates to image bounds
                x = max(0, min(x, orig_w))
                y = max(0, min(y, orig_h))
                w = max(0, min(w, orig_w - x))
                h = max(0, min(h, orig_h - y))

                # Skip if box is too small after clamping
                if w < 1 or h < 1:
                    continue

                pred_dict = {
                    'image_id': int(img_id),
                    'category_id': 1,
                    'segmentation': rle,
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'score': float(score),
                    'id': pred_id
                }

                if debug and img_id == image_ids[0] and idx == 0:
                    print(f"[DEBUG]   First prediction bbox (at {orig_w}x{orig_h}): {pred_dict['bbox']}")

                coco_predictions.append(pred_dict)
                pred_id += 1

    return coco_predictions


def create_coco_gt_from_dataset_original_res(dataset, image_ids=None, debug=False):
    """
    Create COCO ground truth dictionary from dataset at ORIGINAL resolution.

    This matches the inference approach (infer_sam.py) where GT is kept
    at original image size for evaluation.

    Args:
        dataset: Dataset with images and annotations
        image_ids: List of image IDs to include (None = all)
        debug: Print debug info
    """
    if debug:
        print(f"\n[DEBUG] Creating COCO ground truth (ORIGINAL RESOLUTION)...")

    coco_gt = {
        'info': {
            'description': 'SAM3 LoRA Validation Dataset',
            'version': '1.0',
            'year': 2024
        },
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'object'}]
    }

    ann_id = 0
    indices = range(len(dataset)) if image_ids is None else image_ids

    for idx in indices:
        datapoint = dataset[idx]

        # Get original image size
        orig_h, orig_w = datapoint.find_queries[0].inference_metadata.original_size

        coco_gt['images'].append({
            'id': int(idx),
            'width': orig_w,
            'height': orig_h,
            'is_instance_exhaustive': True
        })

        for obj in datapoint.images[0].objects:
            # obj.bbox is normalized [cx, cy, w, h]; convert to COCO [x, y, w, h]
            cx, cy, bw, bh = obj.bbox.tolist()
            w = bw * orig_w
            h = bh * orig_h
            x = cx * orig_w - w / 2
            y = cy * orig_h - h / 2

            ann = {
                'id': ann_id,
                'image_id': int(idx),
                'category_id': 1,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'ignore': 0
            }

            if obj.segment is not None:
                # Upsample mask from 1008x1008 to original size
                mask_tensor = obj.segment.unsqueeze(0).unsqueeze(0).float()
                upsampled_mask = torch.nn.functional.interpolate(
                    mask_tensor,
                    size=(orig_h, orig_w),
                    mode='bilinear',
                    align_corners=False
                ) > 0.5

                mask_np = upsampled_mask.squeeze().cpu().numpy().astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_np))
                rle['counts'] = rle['counts'].decode('utf-8')
                ann['segmentation'] = rle

            coco_gt['annotations'].append(ann)
            ann_id += 1

    if debug:
        print(f"[DEBUG] Created {len(coco_gt['images'])} images, {len(coco_gt['annotations'])} annotations")
        if len(coco_gt['annotations']) > 0:
            sample_gt = coco_gt['annotations'][0]
            sample_img = coco_gt['images'][0]
            print(f"[DEBUG] Sample GT: image_id={sample_gt['image_id']}, bbox={sample_gt['bbox']}, image_size={sample_img['width']}x{sample_img['height']}")

    return coco_gt


def move_to_device(obj, device):
    """Recursively move objects to device"""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(x, device) for x in obj)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif hasattr(obj, "__dataclass_fields__"):
        for field in obj.__dataclass_fields__:
            val = getattr(obj, field)
            setattr(obj, field, move_to_device(val, device))
        return obj
    return obj


def validate(config_path, weights_path, val_data_dir, num_samples=None,
             prob_threshold=0.3, nms_iou=0.7, merge_cracks=False, merge_iou=0.15,
             use_base_model=False, prompt_mode="canonical", prompts_config=None,
             prompt_dataset_key=None, save_per_image_csv=False, per_image_csv_path=None,
             multi_prompt_group=None, save_robustness_csv=False, robustness_csv_path=None,
             save_visualizations=False, vis_output_dir=None, vis_topk=10,
             robustness_csv_input=None, compose_paper_figure_flag=False,
             paper_figure_output=None, paper_figure_topk=10, paper_figure_image_ids=None,
             box_mode="text_only", run_box_dependency_eval=False,
             save_box_csv=False, box_csv_path=None, debug_box_shapes=False,
             robust_checkpoint_path=None, save_pred_masks=False, pred_mask_dir=None,
             selected_cases_csv=None):
    """Run validation with full metrics (mAP, cgF1) and SAM3 NMS

    Args:
        config_path: Path to config file (for LoRA settings only). Not required if use_base_model=True.
        weights_path: Path to LoRA weights. Not required if use_base_model=True.
        val_data_dir: Direct path to validation data directory containing _annotations.coco.json
                      (e.g., /workspace/data2/valid)
        num_samples: Optional limit for number of samples (for debugging)
        use_base_model: If True, use original SAM3 model without LoRA (default: False)

    Example (with LoRA):
        validate(
            config_path="configs/full_lora_config.yaml",
            weights_path="outputs/sam3_lora_full/best_lora_weights.pt",
            val_data_dir="/workspace/data2/valid"
        )

    Example (base SAM3 model):
        validate(
            config_path=None,
            weights_path=None,
            val_data_dir="/workspace/data2/valid",
            use_base_model=True
        )
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = None
    config_path_resolved = None
    if config_path is not None:
        config_path_resolved = Path(config_path).resolve()
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    model_cfg = (config or {}).get("model", {})
    checkpoint_path = model_cfg.get("checkpoint_path")
    load_from_hf = model_cfg.get("load_from_hf", checkpoint_path is None)

    # Build model
    print("\nBuilding SAM3 model...")
    if checkpoint_path:
        print(f"Using local SAM3 checkpoint: {checkpoint_path}")
    elif not load_from_hf:
        raise ValueError(
            "Validation config disables HF download but no model.checkpoint_path was provided."
        )
    model = build_sam3_image_model(
        device=device.type,
        compile=False,
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_hf,
        bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        eval_mode=False
    )

    # Load config for batch_size and other settings
    if use_base_model:
        # Use original SAM3 model without LoRA
        print("Using original SAM3 model (no LoRA)")
        stats = count_parameters(model)
        print(f"Total params: {stats['total_parameters']:,}")
        # Use default batch_size for base model
        batch_size = 1
    else:
        # Apply LoRA and load weights
        if config_path is None or (weights_path is None and robust_checkpoint_path is None):
            raise ValueError("config_path and weights_path or robust_checkpoint_path are required when use_base_model=False")

        # Apply LoRA
        print("Applying LoRA configuration...")
        lora_cfg = config["lora"]
        lora_config = LoRAConfig(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
            apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
            apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
            apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
        )
        model = apply_lora_to_model(model, lora_config)

        if robust_checkpoint_path is not None:
            print(f"\nLoading prompt-robust checkpoint from {robust_checkpoint_path}...")
            ckpt = torch.load(robust_checkpoint_path, map_location="cpu")
            method = ckpt.get("method") or (ckpt.get("config", {}).get("robust", {}) or {}).get("robust_method", "")
            use_cpca = method in {"cpca", "cpca_gpsma", "full", "cpca_v2", "cpca_v2_gpsma", "cpca_v2_gpsma_worst"}
            use_gpsma = method in {"gpsma", "gpsma_worst", "cpca_gpsma", "full", "cpca_v2_gpsma", "cpca_v2_gpsma_worst"}
            cpca_variant = "v2" if method in {"cpca_v2", "cpca_v2_gpsma", "cpca_v2_gpsma_worst"} else "v1"
            model = MedSAM3PromptRobustWrapper(
                model,
                use_cpca=use_cpca,
                use_gpsma=use_gpsma,
                cpca_variant=cpca_variant,
                text_embed_dim=None,
                mask_channels=1,
            )
            state = ckpt.get("model", ckpt)
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(
                f"Loaded robust state: missing={len(missing)} unexpected={len(unexpected)} "
                f"use_cpca={use_cpca} use_gpsma={use_gpsma}"
            )
        else:
            # Load weights
            print(f"\nLoading LoRA weights from {weights_path}...")
            load_lora_weights(model, weights_path)

        stats = count_parameters(model)
        print(f"Trainable params: {stats['trainable_parameters']:,} ({stats['trainable_percentage']:.2f}%)")

        # Get batch_size from config
        batch_size = config["training"]["batch_size"]

    model.to(device)
    model.eval()

    prompts_cfg = (config or {}).get("prompts", {})
    resolved_prompt_mode = prompt_mode
    resolved_multi_prompt_group = multi_prompt_group
    prompt_manager = None
    lexical_variations_by_canonical = {}
    challenging_by_canonical = {}
    dataset_key = prompt_dataset_key or prompts_cfg.get("dataset_key")
    prompt_config_path = prompts_config or prompts_cfg.get("config_path")

    if prompt_config_path is not None:
        if config_path_resolved is not None:
            prompt_config_path = resolve_config_relative_path(config_path_resolved, prompt_config_path)
        else:
            prompt_config_path = Path(prompt_config_path).expanduser().resolve()

    if dataset_key and prompt_config_path is not None and Path(prompt_config_path).exists():
        prompt_manager = PromptManager(
            prompt_config_path=prompt_config_path,
            dataset_key=dataset_key,
            warning_fn=print,
        )

        with open(prompt_config_path, "r", encoding="utf-8") as f:
            prompt_config_raw = yaml.safe_load(f) or {}
        dataset_prompt_cfg = (prompt_config_raw.get("datasets") or {}).get(dataset_key, {})
        raw_lexical_variations = dataset_prompt_cfg.get("lexical_variations", {})
        for canonical_prompt, lexical_list in raw_lexical_variations.items():
            canonical_normalized = prompt_manager.normalize(canonical_prompt)
            cleaned_lexical = []
            for lexical_prompt in lexical_list or []:
                lexical_normalized = prompt_manager.normalize(lexical_prompt)
                if lexical_normalized and lexical_normalized not in cleaned_lexical:
                    cleaned_lexical.append(lexical_normalized)
            lexical_variations_by_canonical[canonical_normalized] = cleaned_lexical
        raw_challenging = dataset_prompt_cfg.get("challenging", {})
        for canonical_prompt, challenging_list in raw_challenging.items():
            canonical_normalized = prompt_manager.normalize(canonical_prompt)
            cleaned_challenging = []
            for challenging_prompt in challenging_list or []:
                challenging_normalized = prompt_manager.normalize(challenging_prompt)
                if (
                    challenging_normalized
                    and challenging_normalized not in cleaned_challenging
                ):
                    cleaned_challenging.append(challenging_normalized)
            challenging_by_canonical[canonical_normalized] = cleaned_challenging
    else:
        if prompt_mode == "synonym":
            print(
                "[Prompt Warning] prompt_mode=synonym requested but prompts config is unavailable. "
                "Falling back to canonical queries."
            )
            resolved_prompt_mode = "canonical"
        if multi_prompt_group == "synonyms":
            print(
                "[Prompt Warning] multi_prompt_group=synonyms requested but prompts config is unavailable. "
                "Falling back to canonical query group."
            )
            resolved_multi_prompt_group = "canonical"
        if multi_prompt_group == "lexical_variations":
            print(
                "[Prompt Warning] multi_prompt_group=lexical_variations requested but prompts config is unavailable. "
                "Falling back to canonical query group."
            )
            resolved_multi_prompt_group = "canonical"
        if multi_prompt_group == "challenging":
            print(
                "[Prompt Warning] multi_prompt_group=challenging requested but prompts config is unavailable. "
                "Falling back to canonical query group."
            )
            resolved_multi_prompt_group = "canonical"

    # Load validation data directly from the specified directory
    print(f"\nLoading validation data from {val_data_dir}...")

    # Load COCO annotations directly
    ann_file = Path(val_data_dir) / "_annotations.coco.json"
    if not ann_file.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")

    # Create a simple dataset class that loads from the directory directly
    class DirectCOCODataset(COCOSegmentDataset):
        def __init__(
            self,
            data_dir,
            prompt_manager=None,
            prompt_mode="canonical",
            multi_prompt_group=None,
            lexical_variations_by_canonical=None,
            challenging_by_canonical=None,
        ):
            self.data_dir = Path(data_dir)
            self.split_dir = self.data_dir
            self.prompt_manager = prompt_manager
            self.prompt_mode = prompt_mode
            self.multi_prompt_group = multi_prompt_group
            self.lexical_variations_by_canonical = lexical_variations_by_canonical or {}
            self.challenging_by_canonical = challenging_by_canonical or {}
            self.warned_missing_lexical_categories = set()
            self.warned_missing_challenging_categories = set()

            # Load COCO annotations
            ann_file = self.split_dir / "_annotations.coco.json"
            if not ann_file.exists():
                raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")

            with open(ann_file, 'r') as f:
                self.coco_data = json.load(f)

            # Build index: image_id -> image info
            self.images = {img['id']: img for img in self.coco_data['images']}
            self.image_ids = sorted(list(self.images.keys()))

            # Build index: image_id -> list of annotations
            self.img_to_anns = {}
            for ann in self.coco_data['annotations']:
                img_id = ann['image_id']
                if img_id not in self.img_to_anns:
                    self.img_to_anns[img_id] = []
                self.img_to_anns[img_id].append(ann)

            # Load categories
            self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
            print(f"Loaded COCO dataset from {data_dir}")
            print(f"  Images: {len(self.image_ids)}")
            print(f"  Annotations: {len(self.coco_data['annotations'])}")
            print(f"  Categories: {self.categories}")

            self.resolution = 1008
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    val_ds = DirectCOCODataset(
        val_data_dir,
        prompt_manager=prompt_manager,
        prompt_mode=resolved_prompt_mode,
        multi_prompt_group=resolved_multi_prompt_group,
        lexical_variations_by_canonical=lexical_variations_by_canonical,
        challenging_by_canonical=challenging_by_canonical,
    )

    print(f"Prompt mode: {resolved_prompt_mode}")
    if resolved_multi_prompt_group is not None:
        print(f"Multi prompt group: {resolved_multi_prompt_group}")
    if dataset_key:
        print(f"Prompt dataset_key: {dataset_key}")
    else:
        print("Prompt dataset_key: not set")

    prompt_examples = []
    category_names = list(val_ds.categories.values())
    if category_names:
        example_categories = category_names[: min(3, len(category_names))]
        for category_name in example_categories:
            query_texts = val_ds._resolve_query_texts(category_name)
            prompt_examples.append((category_name.lower(), query_texts))

    for canonical_prompt, query_texts in prompt_examples:
        print(f"Prompt query sample: {canonical_prompt} -> {', '.join(query_texts[:3])}")

    if resolved_multi_prompt_group == "lexical_variations":
        print("Lexical variations expansion:")
        print(f"  dataset_key: {dataset_key if dataset_key else 'not set'}")
        lexical_categories = sorted(set(prompt_manager.get_canonical_prompt(name) for name in category_names)) if prompt_manager is not None else sorted(set(name.lower() for name in category_names))
        for canonical_prompt in lexical_categories:
            lexical_prompts = val_ds._resolve_query_texts(canonical_prompt)
            print(f"  {canonical_prompt}: {lexical_prompts}")
    if resolved_multi_prompt_group == "challenging":
        print("Challenging prompt expansion:")
        print(f"  dataset_key: {dataset_key if dataset_key else 'not set'}")
        challenging_categories = sorted(set(prompt_manager.get_canonical_prompt(name) for name in category_names)) if prompt_manager is not None else sorted(set(name.lower() for name in category_names))
        for canonical_prompt in challenging_categories:
            challenging_prompts = val_ds._resolve_query_texts(canonical_prompt)
            print(f"  {canonical_prompt}: {challenging_prompts}")

    if num_samples:
        print(f"\n[INFO] Limiting validation to {num_samples} samples for debugging")

    def collate_fn(batch):
        return collate_fn_api(batch, dict_key="input", with_seg_masks=True)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,  # Enable parallel data loading
        pin_memory=True  # Faster GPU transfer
    )

    # Create matcher for loss computation
    matcher = BinaryHungarianMatcherV2(
        cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal=True
    )

    # Run validation
    print("\n" + "="*80)
    print("RUNNING VALIDATION")
    print("="*80)

    all_predictions = []
    all_image_ids = []
    val_losses = []
    per_image_records = []
    box_dependency_mode_records = []
    save_box_csv = save_box_csv or box_csv_path is not None
    run_box_dependency_eval = (
        run_box_dependency_eval or save_box_csv or box_csv_path is not None
    )
    box_modes_to_run = list(BOX_PROMPT_MODES) if run_box_dependency_eval else [box_mode]
    printed_box_shape_debug = set()

    print(f"Box mode: {box_mode}")
    if run_box_dependency_eval:
        print("Box dependency eval modes: text_only, text_plus_loose_box, text_plus_tight_box")

    # Use automatic mixed precision for faster inference
    use_amp = device.type == 'cuda'

    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(tqdm(val_loader, desc="Validation")):
            if num_samples and batch_idx * batch_size >= num_samples:
                break

            input_batch = batch_dict["input"]
            input_batch = move_to_device(input_batch, device)

            for current_box_mode in box_modes_to_run:
                with temporary_box_prompt_mode(input_batch, current_box_mode):
                    if debug_box_shapes and current_box_mode not in printed_box_shape_debug:
                        stage_input = input_batch.find_inputs[0]
                        print(
                            f"[Box Debug] mode={current_box_mode} "
                            f"input_boxes.shape={tuple(stage_input.input_boxes.shape)} "
                            f"input_boxes_mask.shape={tuple(stage_input.input_boxes_mask.shape)} "
                            f"input_boxes_label.shape={tuple(stage_input.input_boxes_label.shape)}"
                        )
                        printed_box_shape_debug.add(current_box_mode)
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            outputs_list = model(input_batch)
                    else:
                        outputs_list = model(input_batch)

                with SAM3Output.iteration_mode(
                    outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
                ) as outputs_iter:
                    final_stage = list(outputs_iter)[-1]
                    final_outputs = final_stage[-1]

                    num_queries_actual = final_outputs['pred_logits'].shape[0]
                    stage_inputs = input_batch.find_inputs[0]
                    stage_targets = input_batch.find_targets[0]
                    stage_metadata = input_batch.find_metadatas[0]

                    num_boxes_list = stage_targets.num_boxes.detach().cpu().tolist()
                    target_segments = None
                    if stage_targets.segments is not None:
                        target_segments = stage_targets.segments.detach().cpu()

                    text_ids = stage_inputs.text_ids.detach().cpu().tolist()
                    local_img_ids = stage_inputs.img_ids.detach().cpu().tolist()
                    coco_image_ids = stage_metadata.coco_image_id.detach().cpu().tolist()
                    original_category_ids = stage_metadata.original_category_id.detach().cpu().tolist()
                    segment_offset = 0

                    for i in range(num_queries_actual):
                        dataset_image_index = batch_idx * batch_size + int(local_img_ids[i])
                        if current_box_mode == box_mode:
                            all_image_ids.append(dataset_image_index)
                            all_predictions.append({
                                'pred_logits': final_outputs['pred_logits'][i].detach().cpu(),
                                'pred_boxes': final_outputs['pred_boxes'][i].detach().cpu(),
                                'pred_masks': final_outputs['pred_masks'][i].detach().cpu()
                            })

                        query_text = input_batch.find_text_batch[text_ids[i]]
                        category_id = int(original_category_ids[i])
                        category_name = val_ds.categories.get(category_id, "object")
                        if prompt_manager is not None:
                            canonical_prompt = prompt_manager.get_canonical_prompt(category_name)
                        else:
                            canonical_prompt = category_name.lower()
                        query_num_boxes = int(num_boxes_list[i])
                        if target_segments is not None and query_num_boxes > 0:
                            gt_masks = target_segments[segment_offset:segment_offset + query_num_boxes]
                        else:
                            gt_masks = None
                        segment_offset += query_num_boxes

                        gt_union_mask = build_union_mask_from_tensor_masks(
                            gt_masks,
                            target_size=(288, 288),
                        )
                        pred_binary_masks = filter_predictions_for_binary_metrics(
                            pred_logits=final_outputs['pred_logits'][i].detach().cpu(),
                            pred_masks=final_outputs['pred_masks'][i].detach().cpu(),
                            pred_boxes=final_outputs['pred_boxes'][i].detach().cpu(),
                            prob_threshold=prob_threshold,
                            nms_iou_threshold=nms_iou,
                            max_detections=100,
                            merge_cracks=merge_cracks,
                            merge_iou_threshold=merge_iou,
                        )
                        pred_union_mask = build_union_mask_from_tensor_masks(
                            pred_binary_masks,
                            target_size=(288, 288),
                        )
                        dice, iou = compute_binary_scores_from_masks(gt_union_mask, pred_union_mask)

                        if current_box_mode == box_mode:
                            per_image_records.append({
                                "image_id": int(coco_image_ids[i]),
                                "category_id": category_id,
                                "canonical_prompt": canonical_prompt,
                                "prompt_group": resolved_multi_prompt_group or resolved_prompt_mode,
                                "prompt_mode": resolved_prompt_mode,
                                "query_text": query_text,
                                "dice": dice,
                                "iou": iou,
                                "dataset_index": dataset_image_index,
                                "gt_union_mask": gt_union_mask,
                                "pred_union_mask": pred_union_mask,
                            })

                        if run_box_dependency_eval:
                            box_dependency_mode_records.append({
                                "image_id": int(coco_image_ids[i]),
                                "category_id": category_id,
                                "canonical_prompt": canonical_prompt,
                                "prompt_group": resolved_multi_prompt_group or resolved_prompt_mode,
                                "prompt_mode": resolved_prompt_mode,
                                "query_text": query_text,
                                "box_mode": current_box_mode,
                                "dice": dice,
                            })

    print(f"\nCollected predictions for {len(all_predictions)} query evaluations")

    # Compute metrics
    print("\n" + "="*80)
    print("COMPUTING METRICS")
    print("="*80)

    # Create COCO ground truth (downsampled to 288×288 - fast!)
    print(f"\n[INFO] Creating ground truth from validation dataset...")
    gt_image_ids = sorted(set(all_image_ids))
    coco_gt_dict = create_coco_gt_from_dataset(
        val_ds,
        image_ids=gt_image_ids,
        mask_resolution=288
    )

    # Check prediction scores (optional - can be commented out for speed)
    # print(f"\n[INFO] Analyzing prediction scores...")
    # all_scores = []
    # for p in all_predictions:
    #     if 'pred_logits' in p and len(p['pred_logits']) > 0:
    #         scores = torch.sigmoid(p['pred_logits']).squeeze(-1)
    #         all_scores.extend(scores.tolist())
    # if all_scores:
    #     print(f"[INFO] Prediction scores: min={min(all_scores):.4f}, max={max(all_scores):.4f}, mean={np.mean(all_scores):.4f}")

    # Convert predictions using SAM3's NMS pipeline or crack merging
    coco_predictions = convert_predictions_to_coco_format(
        all_predictions,
        all_image_ids,
        resolution=288,
        prob_threshold=prob_threshold,
        nms_iou_threshold=nms_iou,
        max_detections=100,
        merge_cracks=merge_cracks,
        merge_iou_threshold=merge_iou
    )

    if merge_cracks:
        print(f"\n[INFO] Total predictions after CRACK MERGING: {len(coco_predictions)}")
    else:
        print(f"\n[INFO] Total predictions after SAM3 NMS filtering: {len(coco_predictions)}")

    if save_per_image_csv:
        csv_output_path = per_image_csv_path
        if csv_output_path is None:
            csv_suffix = resolved_multi_prompt_group or resolved_prompt_mode
            csv_output_path = f"per_image_metrics_{csv_suffix}.csv"
        csv_output_path = Path(csv_output_path)
        save_per_image_metrics_csv(per_image_records, csv_output_path)
        print(f"Per-image CSV saved to: {csv_output_path.resolve()}")

    if save_pred_masks:
        export_selected_prompt_sensitivity_masks(
            dataset=val_ds,
            per_image_records=per_image_records,
            selected_cases_csv=selected_cases_csv,
            output_dir=pred_mask_dir,
            dataset_name=dataset_key,
            prompt_group=resolved_multi_prompt_group or resolved_prompt_mode,
        )

    if save_robustness_csv:
        robustness_records = aggregate_robustness_records(per_image_records)
        robustness_output_path = robustness_csv_path
        if robustness_output_path is None:
            csv_suffix = resolved_multi_prompt_group or resolved_prompt_mode
            robustness_output_path = f"robustness_metrics_{csv_suffix}.csv"
        robustness_output_path = Path(robustness_output_path)
        save_robustness_metrics_csv(robustness_records, robustness_output_path)
        print(f"Robustness CSV saved to: {robustness_output_path.resolve()}")
    else:
        robustness_records = aggregate_robustness_records(per_image_records)

    box_dependency_records = []
    if run_box_dependency_eval:
        box_dependency_records = aggregate_box_dependency_records(box_dependency_mode_records)
        if save_box_csv:
            box_output_path = box_csv_path
            if box_output_path is None:
                csv_suffix = resolved_multi_prompt_group or resolved_prompt_mode
                box_output_path = f"box_dependency_metrics_{csv_suffix}.csv"
            box_output_path = Path(box_output_path)
            save_box_dependency_csv(box_dependency_records, box_output_path)
            print(f"Box dependency CSV saved to: {box_output_path.resolve()}")

    if save_visualizations:
        vis_robustness_records = robustness_records
        if robustness_csv_input is not None:
            vis_robustness_records = load_robustness_metrics_csv(robustness_csv_input)
        vis_dir = vis_output_dir or "visualizations"
        export_topk_visualizations(
            dataset=val_ds,
            per_image_records=per_image_records,
            robustness_records=vis_robustness_records,
            output_dir=vis_dir,
            topk=vis_topk,
        )

    if compose_paper_figure_flag:
        figure_robustness_records = robustness_records
        if robustness_csv_input is not None:
            figure_robustness_records = load_robustness_metrics_csv(robustness_csv_input)
        figure_output = paper_figure_output or "paper_figure.png"
        compose_paper_figure(
            dataset=val_ds,
            per_image_records=per_image_records,
            robustness_records=figure_robustness_records,
            output_path=figure_output,
            topk=paper_figure_topk,
            image_ids=paper_figure_image_ids,
        )

    if len(coco_predictions) > 0:
        # Save temporary files for COCO evaluation
        import tempfile
        import os

        # Create temp directory for evaluation files
        temp_dir = tempfile.mkdtemp(prefix="sam3_eval_")
        gt_file = os.path.join(temp_dir, "gt.json")
        pred_file = os.path.join(temp_dir, "pred.json")

        with open(gt_file, 'w') as f:
            json.dump(coco_gt_dict, f)
        with open(pred_file, 'w') as f:
            json.dump(coco_predictions, f)

        # Compute mAP
        print("\n" + "="*80)
        print("COCO mAP EVALUATION")
        print("="*80)

        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                coco_gt = COCO(str(gt_file))
                coco_dt = coco_gt.loadRes(str(pred_file))
                coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
                coco_eval.params.useCats = False
                coco_eval.evaluate()
                coco_eval.accumulate()

        # Print mAP results
        coco_eval.summarize()

        map_segm = coco_eval.stats[0]
        map50_segm = coco_eval.stats[1]
        map75_segm = coco_eval.stats[2]

        # Compute cgF1
        print("\n" + "="*80)
        print("cgF1 EVALUATION")
        print("="*80)

        cgf1_evaluator = CGF1Evaluator(
            gt_path=str(gt_file),
            iou_type='segm',
            verbose=True
        )
        cgf1_results = cgf1_evaluator.evaluate(str(pred_file))

        cgf1 = cgf1_results.get('cgF1_eval_segm_cgF1', 0.0)
        cgf1_50 = cgf1_results.get('cgF1_eval_segm_cgF1@0.5', 0.0)
        cgf1_75 = cgf1_results.get('cgF1_eval_segm_cgF1@0.75', 0.0)
        binary_metrics = compute_binary_mask_metrics(coco_gt_dict, coco_predictions)

        # Print summary
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        print(f"mAP (IoU 0.50:0.95): {map_segm:.4f}")
        print(f"mAP@50: {map50_segm:.4f}")
        print(f"mAP@75: {map75_segm:.4f}")
        print(f"cgF1 (IoU 0.50:0.95): {cgf1:.4f}")
        print(f"cgF1@50: {cgf1_50:.4f}")
        print(f"cgF1@75: {cgf1_75:.4f}")
        print(f"Mean Dice: {binary_metrics['mean_dice']:.4f}")
        print(f"Mean IoU: {binary_metrics['mean_iou']:.4f}")
        print(f"Global Dice: {binary_metrics['global_dice']:.4f}")
        print(f"Global IoU: {binary_metrics['global_iou']:.4f}")
        print("="*80)

        # Cleanup temporary files
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

    else:
        print("\n[ERROR] No predictions generated! Cannot compute metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone validation script for SAM3 LoRA model with full metrics (mAP, cgF1) and SAM3 NMS"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (for LoRA settings). Not required if --use-base-model is set."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to LoRA weights file. Not required if --use-base-model is set."
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        required=True,
        help="Direct path to validation data directory containing _annotations.coco.json (e.g., /workspace/data2/valid)"
    )
    parser.add_argument(
        "--use-base-model",
        action="store_true",
        help="Use original SAM3 model without LoRA (for baseline comparison)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit validation to N samples (for debugging)"
    )
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.3,
        help="Probability threshold for filtering predictions (default: 0.3)"
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold (default: 0.7)"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Enable aggressive merging of overlapping segments (recommended for crack detection)"
    )
    parser.add_argument(
        "--merge-iou",
        type=float,
        default=0.15,
        help="IoU threshold for merging overlapping predictions (default: 0.15, lower = more aggressive)"
    )
    parser.add_argument(
        "--prompt-mode",
        type=str,
        default="canonical",
        choices=["canonical", "synonym"],
        help="Prompt type used to build validation query_text (default: canonical)"
    )
    parser.add_argument(
        "--prompts-config",
        type=str,
        default=None,
        help="Optional path to prompts.yaml. Defaults to config.prompts.config_path when available."
    )
    parser.add_argument(
        "--prompt-dataset-key",
        type=str,
        default=None,
        help="Optional dataset key inside prompts.yaml. Defaults to config.prompts.dataset_key when available."
    )
    parser.add_argument(
        "--save-per-image-csv",
        action="store_true",
        help="Save prompt-wise per-image Dice and IoU records to CSV."
    )
    parser.add_argument(
        "--per-image-csv-path",
        type=str,
        default=None,
        help="Optional output path for prompt-wise per-image CSV. Defaults to per_image_metrics_<group>.csv"
    )
    parser.add_argument(
        "--multi-prompt-group",
        type=str,
        default=None,
        choices=["canonical", "synonyms", "lexical_variations", "challenging"],
        help="Enable multi-prompt validation for one prompt group. Defaults to disabled."
    )
    parser.add_argument(
        "--box-mode",
        type=str,
        default="text_only",
        choices=list(BOX_PROMPT_MODES),
        help="Single box-prompt mode used for the main validation pass."
    )
    parser.add_argument(
        "--run-box-dependency-eval",
        action="store_true",
        help="Run text_only / loose_box / tight_box for each query and compute BRG per image."
    )
    parser.add_argument(
        "--save-box-csv",
        action="store_true",
        help="Save per-image box dependency metrics (text_only / loose / tight Dice + BRG) to CSV."
    )
    parser.add_argument(
        "--box-csv-path",
        type=str,
        default=None,
        help="Optional output path for box dependency CSV. Defaults to box_dependency_metrics_<group>.csv"
    )
    parser.add_argument(
        "--debug-box-shapes",
        action="store_true",
        help="Print box prompt tensor shapes once per box mode before calling model(input_batch)."
    )
    parser.add_argument(
        "--save-robustness-csv",
        action="store_true",
        help="Save per-image robustness metrics (WPD / BestPromptDice / PRG) to CSV."
    )
    parser.add_argument(
        "--robustness-csv-path",
        type=str,
        default=None,
        help="Optional output path for robustness CSV. Defaults to robustness_metrics_<group>.csv"
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Export top-k robustness visualization images."
    )
    parser.add_argument(
        "--vis-output-dir",
        type=str,
        default=None,
        help="Directory for visualization outputs. Defaults to ./visualizations"
    )
    parser.add_argument(
        "--vis-topk",
        type=int,
        default=10,
        help="Number of top-PRG samples to visualize (default: 10)"
    )
    parser.add_argument(
        "--robustness-csv-input",
        type=str,
        default=None,
        help="Optional robustness CSV used only for visualization ranking. If omitted, use current run results."
    )
    parser.add_argument(
        "--compose-paper-figure",
        action="store_true",
        help="Compose a paper-style multi-row figure with fixed 5-column layout."
    )
    parser.add_argument(
        "--paper-figure-output",
        type=str,
        default=None,
        help="Output path for the composed paper figure. Defaults to ./paper_figure.png"
    )
    parser.add_argument(
        "--paper-figure-topk",
        type=int,
        default=10,
        help="Number of top-PRG samples to include when image ids are not explicitly provided."
    )
    parser.add_argument(
        "--paper-figure-image-ids",
        nargs="*",
        type=int,
        default=None,
        help="Explicit image_id list for the paper figure. If set, overrides top-k selection."
    )
    parser.add_argument(
        "--save-pred-masks",
        action="store_true",
        help="Save selected-case original image, GT mask, and prediction mask PNG files."
    )
    parser.add_argument(
        "--pred-mask-dir",
        type=str,
        default=None,
        help="Directory for selected-case prediction mask PNG files."
    )
    parser.add_argument(
        "--selected-cases-csv",
        type=str,
        default=None,
        help="CSV produced by tools/select_prompt_sensitivity_cases.py."
    )
    args = parser.parse_args()

    # Validate argument combinations
    if not args.use_base_model:
        if args.config is None or args.weights is None:
            parser.error("--config and --weights are required when not using --use-base-model")

    validate(
        config_path=args.config,
        weights_path=args.weights,
        val_data_dir=args.val_data_dir,
        num_samples=args.num_samples,
        prob_threshold=args.prob_threshold,
        nms_iou=args.nms_iou,
        merge_cracks=args.merge,
        merge_iou=args.merge_iou,
        use_base_model=args.use_base_model,
        prompt_mode=args.prompt_mode,
        prompts_config=args.prompts_config,
        prompt_dataset_key=args.prompt_dataset_key,
        save_per_image_csv=args.save_per_image_csv,
        per_image_csv_path=args.per_image_csv_path,
        multi_prompt_group=args.multi_prompt_group,
        box_mode=args.box_mode,
        run_box_dependency_eval=args.run_box_dependency_eval,
        save_box_csv=args.save_box_csv,
        box_csv_path=args.box_csv_path,
        debug_box_shapes=args.debug_box_shapes,
        save_robustness_csv=args.save_robustness_csv,
        robustness_csv_path=args.robustness_csv_path,
        save_visualizations=args.save_visualizations,
        vis_output_dir=args.vis_output_dir,
        vis_topk=args.vis_topk,
        robustness_csv_input=args.robustness_csv_input,
        compose_paper_figure_flag=args.compose_paper_figure,
        paper_figure_output=args.paper_figure_output,
        paper_figure_topk=args.paper_figure_topk,
        paper_figure_image_ids=args.paper_figure_image_ids,
        save_pred_masks=args.save_pred_masks,
        pred_mask_dir=args.pred_mask_dir,
        selected_cases_csv=args.selected_cases_csv,
    )
