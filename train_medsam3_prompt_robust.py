#!/usr/bin/env python3
"""Prompt-robust MedSAM3 training entrypoint.

This script leaves the original SAM3/MedSAM3 backbone source untouched. It
wraps the built model with CPCA/GPSMA adapters and expands each training image
into prompt groups for robustness losses.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
from collections import defaultdict
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F
import yaml
from PIL import Image as PILImage
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights, save_lora_weights
from models.medsam3_prompt_robust_wrapper import MedSAM3PromptRobustWrapper
from prompt_utils import PromptManager
from sam3.model.model_misc import SAM3Output
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image,
    InferenceMetadata,
    Object,
)
from train_sam3_lora_native import (
    COCOSegmentDataset,
    decode_coco_segmentation,
    is_main_process,
    print_rank0,
    resolve_config_relative_path,
)


ROBUST_METHODS = (
    "ft_canon",
    "ft_norm",
    "ft_norm_group",
    "cpca",
    "gpsma",
    "gpsma_worst",
    "cpca_gpsma",
    "full",
    "cpca_v2",
    "cpca_v2_gpsma",
    "cpca_v2_gpsma_worst",
)


def apply_robust_method_flags(args: argparse.Namespace) -> None:
    mapping = {
        "ft_canon": dict(use_cpca=False, cpca_variant="v1", use_gpsma=False, use_group_consistency=False, use_worst_loss=False, use_proto_loss=False),
        "ft_norm": dict(use_cpca=False, cpca_variant="v1", use_gpsma=False, use_group_consistency=False, use_worst_loss=False, use_proto_loss=False),
        "ft_norm_group": dict(use_cpca=False, cpca_variant="v1", use_gpsma=False, use_group_consistency=True, use_worst_loss=False, use_proto_loss=False),
        "cpca": dict(use_cpca=True, cpca_variant="v1", use_gpsma=False, use_group_consistency=True, use_worst_loss=False, use_proto_loss=True),
        "gpsma": dict(use_cpca=False, cpca_variant="v1", use_gpsma=True, use_group_consistency=True, use_worst_loss=False, use_proto_loss=False),
        "gpsma_worst": dict(use_cpca=False, cpca_variant="v1", use_gpsma=True, use_group_consistency=True, use_worst_loss=True, use_proto_loss=False),
        "cpca_gpsma": dict(use_cpca=True, cpca_variant="v1", use_gpsma=True, use_group_consistency=True, use_worst_loss=False, use_proto_loss=True),
        "full": dict(use_cpca=True, cpca_variant="v1", use_gpsma=True, use_group_consistency=True, use_worst_loss=True, use_proto_loss=True),
        "cpca_v2": dict(use_cpca=True, cpca_variant="v2", use_gpsma=False, use_group_consistency=True, use_worst_loss=False, use_proto_loss=False),
        "cpca_v2_gpsma": dict(use_cpca=True, cpca_variant="v2", use_gpsma=True, use_group_consistency=True, use_worst_loss=False, use_proto_loss=False),
        "cpca_v2_gpsma_worst": dict(use_cpca=True, cpca_variant="v2", use_gpsma=True, use_group_consistency=True, use_worst_loss=True, use_proto_loss=False),
    }
    for key, value in mapping[args.robust_method].items():
        setattr(args, key, value)


def print_robust_method_config(args: argparse.Namespace) -> None:
    print_rank0(
        f"[RobustMethod] {args.robust_method}:\n"
        f"  use_cpca={bool(args.use_cpca)}\n"
        f"  cpca_variant={args.cpca_variant}\n"
        f"  use_gpsma={bool(args.use_gpsma)}\n"
        f"  use_group_consistency={bool(args.use_group_consistency)}\n"
        f"  use_worst_loss={bool(args.use_worst_loss)}\n"
        f"  use_proto_loss={bool(args.use_proto_loss)}"
    )


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_config(raw_cfg: Dict, args: argparse.Namespace) -> Dict:
    cfg = dict(raw_cfg)
    if "training" not in cfg:
        cfg = {
            "model": {
                "checkpoint_path": cfg.get("sam3_ckpt", "checkpoints/sam3.pt"),
                "load_from_hf": cfg.get("load_from_hf", False),
            },
            "lora": cfg.get(
                "lora",
                {
                    "rank": 16,
                    "alpha": 32,
                    "dropout": 0.1,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
                    "apply_to_vision_encoder": False,
                    "apply_to_text_encoder": True,
                    "apply_to_geometry_encoder": False,
                    "apply_to_detr_encoder": True,
                    "apply_to_detr_decoder": True,
                    "apply_to_mask_decoder": True,
                },
            ),
            "training": {
                "data_dir": cfg.get("data_dir"),
                "batch_size": cfg.get("batch_size", 1),
                "num_workers": cfg.get("num_workers", 0),
                "num_epochs": cfg.get("epochs", 20),
                "weight_decay": cfg.get("weight_decay", 0.01),
                "mixed_precision": cfg.get("mixed_precision", "bf16"),
                "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps", 1),
                "seed": cfg.get("seed", 42),
            },
            "prompts": {
                "enabled": True,
                "config_path": cfg.get("prompts_config", "configs/prompts.yaml"),
                "dataset_key": cfg.get("dataset_key"),
            },
            "output": {
                "output_dir": cfg.get("output_dir", f"work_dir/{cfg.get('dataset_key', 'dataset')}/prompt_robust/{cfg.get('robust_method', 'full')}"),
                "pretrained_lora_weights": cfg.get("pretrained_lora_weights"),
            },
            "robust": cfg,
        }
    cfg.setdefault("robust", {})
    cfg.setdefault("training", {})
    cfg.setdefault("output", {})
    cfg.setdefault("prompts", {})

    # Full prompt-robust configs intentionally expose common experiment knobs
    # at the top level. Keep them authoritative so changing `epochs: 2` also
    # updates the nested SAM3-style `training.num_epochs`.
    top_level_training_aliases = {
        "data_dir": "data_dir",
        "batch_size": "batch_size",
        "num_workers": "num_workers",
        "epochs": "num_epochs",
    }
    for top_key, training_key in top_level_training_aliases.items():
        if cfg.get(top_key) is not None:
            cfg["training"][training_key] = cfg[top_key]
    if cfg.get("output_dir") is not None:
        cfg["output"]["output_dir"] = cfg["output_dir"]
    if cfg.get("prompts_config") is not None:
        cfg["prompts"]["config_path"] = cfg["prompts_config"]
    if cfg.get("dataset_key") is not None:
        cfg["prompts"]["dataset_key"] = cfg["dataset_key"]

    robust_cfg = cfg["robust"]
    for key in (
        "robust_method",
        "num_group_prompts",
        "group_loss_weight",
        "proto_loss_weight",
        "worst_loss_weight",
        "worst_tau",
        "use_proto_loss",
        "cpca_variant",
        "cpca_v2_residual_scale",
        "cpca_v2_semantic_loss_weight",
        "lora_lr",
        "cpca_lr",
        "gpsma_lr",
        "decoder_lr",
        "freeze_backbone",
        "train_lora",
        "train_cpca",
        "train_gpsma",
        "train_detector",
    ):
        if hasattr(args, key) and getattr(args, key) is not None:
            robust_cfg[key] = getattr(args, key)
    cfg["training"]["data_dir"] = args.data_dir or cfg["training"].get("data_dir")
    if getattr(args, "epochs", None) is not None:
        cfg["epochs"] = args.epochs
        cfg["training"]["num_epochs"] = args.epochs
    cfg["output"]["output_dir"] = args.output_dir or cfg["output"].get("output_dir")
    cfg["prompts"]["config_path"] = args.prompts_config or cfg.get("prompts", {}).get("config_path")
    cfg["prompts"]["dataset_key"] = args.prompt_dataset_key or cfg.get("prompts", {}).get("dataset_key")
    return cfg


def load_prompt_groups(prompt_config_path: Path, dataset_key: str, manager: PromptManager) -> Dict[str, Dict[str, List[str]]]:
    raw = load_yaml(str(prompt_config_path))
    dataset_cfg = (raw.get("datasets") or {}).get(dataset_key, {})
    groups = {"synonyms": {}, "lexical_variations": {}, "challenging": {}}
    for group_name in groups:
        for canonical, values in (dataset_cfg.get(group_name) or {}).items():
            canonical_norm = manager.normalize(canonical)
            cleaned = []
            for value in values or []:
                norm = manager.normalize(value)
                if norm and norm not in cleaned:
                    cleaned.append(norm)
            groups[group_name][canonical_norm] = cleaned
    return groups


class PromptRobustCOCODataset(COCOSegmentDataset):
    def __init__(
        self,
        data_dir,
        split,
        prompt_manager: Optional[PromptManager],
        prompt_groups: Dict[str, Dict[str, List[str]]],
        robust_method: str,
        prompt_group_names: Sequence[str],
        num_group_prompts: int,
        canonical_prompt_override: Optional[str] = None,
        synonym_prompts_override: Optional[Sequence[str]] = None,
    ):
        parent_method = "ft_norm" if robust_method != "ft_canon" else "ft_canon"
        super().__init__(
            data_dir=data_dir,
            split=split,
            prompt_manager=prompt_manager if parent_method == "ft_norm" else None,
            enable_prompt_pair_sampling=False,
            method_variant=parent_method,
        )
        self.prompt_manager = prompt_manager
        self.prompt_groups = prompt_groups
        self.robust_method = robust_method
        self.prompt_group_names = list(prompt_group_names)
        self.num_group_prompts = max(1, int(num_group_prompts))
        self.canonical_prompt_override = canonical_prompt_override
        self.synonym_prompts_override = [p.strip() for p in (synonym_prompts_override or []) if p.strip()]
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _canonical_prompt(self, category_name: str) -> str:
        if self.canonical_prompt_override:
            return self.prompt_manager.normalize(self.canonical_prompt_override) if self.prompt_manager else self.canonical_prompt_override.lower()
        if self.prompt_manager is not None:
            return self.prompt_manager.get_canonical_prompt(category_name)
        return category_name.lower()

    def _candidate_prompts(self, canonical_prompt: str) -> List[str]:
        candidates = []
        if self.synonym_prompts_override:
            candidates.extend(self.synonym_prompts_override)
        for group_name in self.prompt_group_names:
            if group_name == "canonical":
                continue
            candidates.extend(self.prompt_groups.get(group_name, {}).get(canonical_prompt, []))
        deduped = []
        for prompt in candidates:
            norm = self.prompt_manager.normalize(prompt) if self.prompt_manager is not None else prompt.lower().strip()
            if norm and norm != canonical_prompt and norm not in deduped:
                deduped.append(norm)
        return deduped

    def _select_prompts(self, category_name: str) -> List[str]:
        canonical = self._canonical_prompt(category_name)
        if self.robust_method == "ft_canon":
            return [canonical]
        candidates = self._candidate_prompts(canonical)
        if self.robust_method == "ft_norm":
            pool = [canonical] + candidates
            return [random.choice(pool) if pool else canonical]
        extra_count = max(0, self.num_group_prompts - 1)
        if not candidates:
            return [canonical]
        if len(candidates) >= extra_count:
            sampled = random.sample(candidates, extra_count)
        else:
            sampled = [random.choice(candidates) for _ in range(extra_count)]
        return [canonical] + sampled

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = self.split_dir / img_info["file_name"]
        pil_image = PILImage.open(img_path).convert("RGB")
        orig_w, orig_h = pil_image.size
        pil_image = pil_image.resize((self.resolution, self.resolution), PILImage.BILINEAR)
        image_tensor = self.transform(pil_image)
        annotations = self.img_to_anns.get(img_id, [])
        scale_w = self.resolution / orig_w
        scale_h = self.resolution / orig_h
        objects, object_class_names, object_category_ids = [], [], []

        for obj_idx, ann in enumerate(annotations):
            bbox_coco = ann.get("bbox")
            if bbox_coco is None:
                continue
            category_id = ann.get("category_id", 0)
            class_name = self.categories.get(category_id, "object")
            object_class_names.append(class_name)
            object_category_ids.append(category_id)
            x, y, w, h = bbox_coco
            box_tensor = torch.tensor(
                [
                    (x + w / 2.0) * scale_w / self.resolution,
                    (y + h / 2.0) * scale_h / self.resolution,
                    w * scale_w / self.resolution,
                    h * scale_h / self.resolution,
                ],
                dtype=torch.float32,
            )
            segment = None
            segmentation = ann.get("segmentation")
            if segmentation:
                try:
                    mask_np = decode_coco_segmentation(segmentation, orig_h, orig_w)
                    mask_t = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
                    mask_t = F.interpolate(mask_t, size=(self.resolution, self.resolution), mode="nearest")
                    segment = mask_t.squeeze() > 0.5
                except Exception as exc:
                    print(f"Warning: Error processing mask for image {img_id}, ann {obj_idx}: {exc}")
            objects.append(
                Object(
                    bbox=box_tensor,
                    area=(box_tensor[2] * box_tensor[3]).item(),
                    object_id=obj_idx,
                    segment=segment,
                )
            )

        class_to_query_info = defaultdict(lambda: {"object_ids": [], "category_id": 0, "category_name": ""})
        for obj, class_name, category_id in zip(objects, object_class_names, object_category_ids):
            class_key = class_name.lower()
            class_to_query_info[class_key]["object_ids"].append(obj.object_id)
            class_to_query_info[class_key]["category_id"] = category_id
            class_to_query_info[class_key]["category_name"] = class_name

        queries = []
        if class_to_query_info:
            for query_info in class_to_query_info.values():
                for query_text in self._select_prompts(query_info["category_name"]):
                    queries.append(
                        self._build_query(
                            query_text,
                            query_info["object_ids"],
                            img_id,
                            orig_h,
                            orig_w,
                            query_info["category_id"],
                        )
                    )
        else:
            queries.append(self._build_query("object", [], img_id, orig_h, orig_w, 0))

        return Datapoint(
            find_queries=queries,
            images=[Image(data=image_tensor, objects=objects, size=(self.resolution, self.resolution))],
            raw_images=[pil_image],
        )


def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        for field in obj.__dataclass_fields__:
            setattr(obj, field, move_to_device(getattr(obj, field), device))
        return obj
    return obj


def select_query_mask_logits(final_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    masks = final_outputs["pred_masks"]
    if masks.dim() == 3:
        return masks.unsqueeze(1)
    logits = final_outputs.get("pred_logits")
    if logits is None or logits.dim() < 2:
        return masks[:, :1]
    scores = logits.squeeze(-1).detach()
    best_idx = scores.argmax(dim=1)
    batch_idx = torch.arange(masks.shape[0], device=masks.device)
    return masks[batch_idx, best_idx].unsqueeze(1)


def build_query_gt_masks(input_batch, final_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    pred = select_query_mask_logits(final_outputs)
    q, _, h, w = pred.shape
    stage_targets = input_batch.find_targets[0]
    num_boxes = stage_targets.num_boxes.detach().cpu().tolist()
    segments = stage_targets.segments
    out = []
    offset = 0
    for i in range(q):
        n = int(num_boxes[i]) if i < len(num_boxes) else 0
        if segments is None or n <= 0:
            gt = pred.new_zeros((1, h, w))
        else:
            seg = segments[offset : offset + n].float().unsqueeze(1)
            if seg.shape[-2:] != (h, w):
                seg = F.interpolate(seg, size=(h, w), mode="nearest")
            gt = (seg.max(dim=0).values > 0.5).to(dtype=pred.dtype)
        offset += n
        out.append(gt)
    return torch.stack(out, dim=0)


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.dim()))
    intersection = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    return 1.0 - ((2.0 * intersection + eps) / (denom + eps))


def dice_scores_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = (torch.sigmoid(logits) > 0.5).to(targets.dtype)
    dims = tuple(range(1, pred.dim()))
    intersection = (pred * targets).sum(dim=dims)
    denom = pred.sum(dim=dims) + targets.sum(dim=dims)
    return (2.0 * intersection + eps) / (denom + eps)


def dice_loss_group_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(2, probs.dim()))
    intersection = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    return 1.0 - ((2.0 * intersection + eps) / (denom + eps))


def dice_scores_group_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = (torch.sigmoid(logits) > 0.5).to(targets.dtype)
    dims = tuple(range(2, pred.dim()))
    intersection = (pred * targets).sum(dim=dims)
    denom = pred.sum(dim=dims) + targets.sum(dim=dims)
    return (2.0 * intersection + eps) / (denom + eps)


def query_group_keys(input_batch, prompt_manager: Optional[PromptManager], categories: Dict[int, str]) -> List[tuple]:
    stage_inputs = input_batch.find_inputs[0]
    stage_meta = input_batch.find_metadatas[0]
    text_ids = stage_inputs.text_ids.detach().cpu().tolist()
    local_img_ids = stage_inputs.img_ids.detach().cpu().tolist()
    category_ids = stage_meta.original_category_id.detach().cpu().tolist()
    keys = []
    for text_id, local_img_id, category_id in zip(text_ids, local_img_ids, category_ids):
        category_name = categories.get(int(category_id), "object")
        canonical = prompt_manager.get_canonical_prompt(category_name) if prompt_manager else category_name.lower()
        keys.append((int(local_img_id), int(category_id), canonical, input_batch.find_text_batch[text_id]))
    return keys


def build_prompt_group_batches(
    flat_logits: torch.Tensor,
    flat_targets: torch.Tensor,
    keys: List[tuple],
) -> List[Dict[str, object]]:
    grouped: "OrderedDict[tuple, List[int]]" = OrderedDict()
    for idx, key in enumerate(keys):
        grouped.setdefault(key[:3], []).append(idx)

    buckets: Dict[int, List[tuple]] = defaultdict(list)
    for group_key, indices in grouped.items():
        buckets[len(indices)].append((group_key, indices))

    group_batches = []
    for prompt_count, items in sorted(buckets.items(), key=lambda item: item[0]):
        if prompt_count <= 0:
            continue
        logits_list = []
        targets_list = []
        prompt_lists = []
        group_keys = []
        for group_key, indices in items:
            group_keys.append(group_key)
            prompt_lists.append([keys[idx][3] for idx in indices])
        for prompt_idx in range(prompt_count):
            gather_idx = torch.tensor(
                [indices[prompt_idx] for _, indices in items],
                device=flat_logits.device,
                dtype=torch.long,
            )
            logits_list.append(flat_logits.index_select(0, gather_idx))
            targets_list.append(flat_targets.index_select(0, gather_idx))
        group_batches.append(
            {
                "group_logits": torch.stack(logits_list, dim=1),
                "group_targets": torch.stack(targets_list, dim=1),
                "prompt_lists": prompt_lists,
                "group_keys": group_keys,
            }
        )
    return group_batches


def get_prompt_robust_wrapper(model: torch.nn.Module) -> Optional[MedSAM3PromptRobustWrapper]:
    wrapper = model.module if hasattr(model, "module") else model
    if isinstance(wrapper, MedSAM3PromptRobustWrapper):
        return wrapper
    return None


def reset_cpca_debug_state(model: torch.nn.Module) -> None:
    wrapper = get_prompt_robust_wrapper(model)
    if wrapper is not None:
        wrapper.reset_cpca_debug_state()


def get_cpca_debug_state(model: torch.nn.Module) -> Dict[str, object]:
    wrapper = get_prompt_robust_wrapper(model)
    if wrapper is None:
        return {
            "cpca_variant": None,
            "cpca_hook_used": False,
            "cpca_text_embedding_shape": None,
            "cpca_layout": None,
            "cpca_num_prompt_slots": None,
            "cpca_canonical_from_current_group": False,
            "cpca_group_from_current_group": False,
            "canonical_embedding_shape": None,
            "group_embedding_shape": None,
            "cpca_calibrated_embedding_shape": None,
            "cpca_v2_active": False,
            "cpca_v2_reliability_mean": None,
            "cpca_v2_reliability_min": None,
            "cpca_v2_reliability_max": None,
            "cpca_v2_calibration_strength_mean": None,
            "cpca_v2_calibration_delta_norm": None,
        }
    return wrapper.get_cpca_debug_state()


def set_cpca_context(
    model: torch.nn.Module,
    num_group_prompts: Optional[int],
    batch_size: Optional[int],
    canonical_prompt_index: int = 0,
) -> None:
    wrapper = get_prompt_robust_wrapper(model)
    if wrapper is not None:
        wrapper.set_cpca_context(
            num_group_prompts=num_group_prompts,
            batch_size=batch_size,
            canonical_prompt_index=canonical_prompt_index,
        )


def infer_batch_size(input_batch) -> Optional[int]:
    img_batch = getattr(input_batch, "img_batch", None)
    if isinstance(img_batch, torch.Tensor) and img_batch.dim() > 0:
        return int(img_batch.shape[0])
    return None


def infer_num_group_prompts_from_keys(keys: List[tuple]) -> int:
    grouped = defaultdict(int)
    for key in keys:
        grouped[key[:3]] += 1
    return max(grouped.values()) if grouped else 1


def refine_group_logits_with_gpsma(
    model: torch.nn.Module,
    group_logits: torch.Tensor,
    use_gpsma: bool,
) -> tuple[torch.Tensor, bool]:
    wrapper = get_prompt_robust_wrapper(model)
    if not use_gpsma or wrapper is None or wrapper.gpsma is None:
        return group_logits, False
    refined_logits_list = []
    for prompt_idx in range(group_logits.shape[1]):
        logits_k = group_logits[:, prompt_idx]
        refined_k = wrapper.gpsma(mask_logits=logits_k, group_mask_logits=group_logits)
        refined_logits_list.append(refined_k)
    return torch.stack(refined_logits_list, dim=1), True


def compute_prompt_group_losses(
    model: torch.nn.Module,
    group_batches: List[Dict[str, object]],
    use_gpsma: bool,
    use_group_consistency: bool,
    use_worst_loss: bool,
    tau: float,
) -> Dict[str, object]:
    if not group_batches:
        param = next(model.parameters())
        zero = param.new_zeros(())
        return {
            "loss_seg": zero,
            "loss_group": zero,
            "loss_worst": zero,
            "per_prompt_dice": [],
            "prompt_list": [],
            "group_logits_shapes": [],
            "refined_group_logits_shapes": [],
            "gpsma_group_logits_used": False,
        }

    loss_seg_sum = None
    loss_group_terms = []
    loss_worst_terms = []
    per_prompt_dice = []
    prompt_list = []
    group_logits_shapes = []
    refined_group_logits_shapes = []
    gpsma_used_any = False
    num_prompt_items = 0

    for group_batch in group_batches:
        group_logits = group_batch["group_logits"]
        group_targets = group_batch["group_targets"]
        refined_group_logits, gpsma_used = refine_group_logits_with_gpsma(
            model,
            group_logits,
            use_gpsma=use_gpsma,
        )
        gpsma_used_any = gpsma_used_any or gpsma_used
        group_logits_shapes.append(tuple(group_logits.shape))
        refined_group_logits_shapes.append(tuple(refined_group_logits.shape))

        bce = F.binary_cross_entropy_with_logits(
            refined_group_logits,
            group_targets,
            reduction="none",
        ).mean(dim=(2, 3, 4))
        dice_losses = dice_loss_group_with_logits(refined_group_logits, group_targets)
        losses_per_prompt = bce + dice_losses
        loss_seg_sum = losses_per_prompt.sum() if loss_seg_sum is None else loss_seg_sum + losses_per_prompt.sum()
        num_prompt_items += int(losses_per_prompt.numel())

        if use_group_consistency and refined_group_logits.shape[1] > 1:
            probs = torch.sigmoid(refined_group_logits)
            mean_probs = probs.mean(dim=1, keepdim=True)
            loss_group_terms.append(((probs - mean_probs) ** 2).mean())

        if use_worst_loss:
            alpha = torch.softmax(losses_per_prompt / max(tau, 1e-6), dim=1)
            loss_worst_terms.append((alpha * losses_per_prompt).sum(dim=1).mean())

        dice_scores = dice_scores_group_with_logits(refined_group_logits, group_targets)
        per_prompt_dice.extend(float(x) for x in dice_scores.detach().flatten().cpu().tolist())
        for prompts in group_batch["prompt_lists"]:
            prompt_list.extend(prompts)

    assert loss_seg_sum is not None
    loss_seg = loss_seg_sum / max(1, num_prompt_items)
    zero = loss_seg.new_zeros(())
    loss_group = torch.stack(loss_group_terms).mean() if loss_group_terms else zero
    loss_worst = torch.stack(loss_worst_terms).mean() if loss_worst_terms else zero
    return {
        "loss_seg": loss_seg,
        "loss_group": loss_group,
        "loss_worst": loss_worst,
        "per_prompt_dice": per_prompt_dice,
        "prompt_list": prompt_list,
        "group_logits_shapes": group_logits_shapes,
        "refined_group_logits_shapes": refined_group_logits_shapes,
        "gpsma_group_logits_used": gpsma_used_any,
    }


def compute_proto_loss(model: torch.nn.Module) -> tuple[torch.Tensor, bool, int]:
    wrapper = model.module if hasattr(model, "module") else model
    if not isinstance(wrapper, MedSAM3PromptRobustWrapper) or wrapper.cpca is None:
        return next(wrapper.parameters()).new_zeros(()), False, 0
    if wrapper.last_cpca_output is None or wrapper.last_cpca_canonical is None:
        return next(wrapper.parameters()).new_zeros(()), False, 0
    if not wrapper.cpca_canonical_from_current_group:
        return next(wrapper.parameters()).new_zeros(()), False, int(wrapper.cpca_num_prompt_slots or 1)
    calibrated = wrapper.last_cpca_output
    canonical = wrapper.last_cpca_canonical.detach()
    num_prompts = int(wrapper.cpca_num_prompt_slots or wrapper.cpca_context_num_group_prompts or 1)
    if num_prompts <= 1 or calibrated.shape[0] < num_prompts or calibrated.shape[0] % num_prompts != 0:
        return calibrated.new_zeros(()), False, max(1, num_prompts)
    grouped_calibrated = calibrated.view(-1, num_prompts, calibrated.shape[-1])
    grouped_canonical = canonical.view(-1, num_prompts, canonical.shape[-1])
    noncanonical = grouped_calibrated[:, 1:, :]
    canonical_vec = grouped_canonical[:, 0:1, :].expand_as(noncanonical).detach()
    if noncanonical.numel() == 0:
        return calibrated.new_zeros(()), False, num_prompts
    loss = (1.0 - F.cosine_similarity(noncanonical.float(), canonical_vec.float(), dim=-1)).mean()
    return loss.to(calibrated.dtype), True, num_prompts


def iter_named_params(model: torch.nn.Module, substrings: Iterable[str]):
    lowered = tuple(s.lower() for s in substrings)
    for name, param in model.named_parameters():
        if any(s in name.lower() for s in lowered):
            yield name, param


def configure_trainable_params(model, args):
    for _, param in model.named_parameters():
        param.requires_grad = not args.freeze_backbone
    groups, seen = [], set()
    cpca_label = "cpca_v2" if args.cpca_variant == "v2" else "cpca"
    counts = {"lora": 0, "cpca": 0, "cpca_v2": 0, "gpsma": 0, "detector": 0, "frozen": 0}

    def add_group(label, params, lr):
        picked = []
        for _, param in params:
            if id(param) in seen:
                continue
            param.requires_grad = True
            picked.append(param)
            seen.add(id(param))
        counts[label] = sum(p.numel() for p in picked)
        if picked:
            groups.append({"params": picked, "lr": lr, "name": label})

    if args.train_lora:
        add_group("lora", iter_named_params(model, ("lora_A", "lora_B", "lora")), args.lora_lr)
    if args.train_cpca or args.use_cpca:
        add_group(cpca_label, iter_named_params(model, ("cpca",)), args.cpca_lr)
    if args.train_gpsma or args.use_gpsma:
        add_group("gpsma", iter_named_params(model, ("gpsma",)), args.gpsma_lr)
    if args.train_detector:
        add_group("detector", iter_named_params(model, ("segmentation_head", "mask_decoder", "decoder", "class_embed", "bbox_embed")), args.decoder_lr)

    trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    counts["frozen"] = frozen
    print_rank0(f"[Params] trainable total = {trainable_total:,}")
    labels_to_print = ("lora", cpca_label, "gpsma", "detector", "frozen")
    for label in labels_to_print:
        print_rank0(f"[Params] {label} = {counts[label]:,}")
    if args.use_cpca and counts[cpca_label] == 0:
        print_rank0("[Warning] CPCA enabled but no CPCA trainable parameters found.")
    if args.use_gpsma and counts["gpsma"] == 0:
        print_rank0("[Warning] GPSMA enabled but no GPSMA trainable parameters found.")
    if not groups:
        print_rank0("[Warning] No trainable parameter groups found; optimizer will be empty.")
    return groups


def save_run_artifacts(out_dir: Path, config_path: str, cfg: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, out_dir / "train_medsam3_prompt_robust.py")
    if config_path and Path(config_path).exists():
        shutil.copy2(config_path, out_dir / Path(config_path).name)
    with open(out_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    try:
        diff = subprocess.run(["git", "diff", "--"], cwd=Path(__file__).resolve().parent, capture_output=True, text=True, check=False)
        (out_dir / "git_diff.patch").write_text(diff.stdout, encoding="utf-8")
    except Exception as exc:
        (out_dir / "git_diff.patch").write_text(f"git diff unavailable: {exc}\n", encoding="utf-8")


def get_trainable_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Return a compact checkpoint state for LoRA/adapters/detector params."""
    trainable_names = {
        name
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    state = model.state_dict()
    compact = {}
    for name, tensor in state.items():
        if name in trainable_names or "lora_" in name.lower() or ".cpca." in name.lower() or ".gpsma." in name.lower():
            compact[name] = tensor.detach().cpu()
    return compact


def safe_torch_save(payload: Dict, path: Path) -> bool:
    """Write a checkpoint through a temporary file and keep training alive on I/O failure."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
        return True
    except Exception as exc:
        print_rank0(f"[Checkpoint Warning] Failed to save {path}: {exc}")
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        return False


def require_cuda_for_sam3() -> None:
    """SAM3 image builder currently allocates position encodings on CUDA."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    if not cuda_available or device_count < 1:
        raise RuntimeError(
            "CUDA is required before building SAM3, but no CUDA GPU is visible to this process. "
            f"CUDA_VISIBLE_DEVICES={cuda_visible}, "
            f"torch.cuda.is_available()={cuda_available}, "
            f"torch.cuda.device_count()={device_count}. "
            "Check that the selected GPU id is allocated/visible in this shell or container. "
            "For example, run `nvidia-smi` and "
            "`python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.device_count())\"`."
        )


def train(args: argparse.Namespace) -> None:
    apply_robust_method_flags(args)
    raw_cfg = load_yaml(args.config)
    cfg = normalize_config(raw_cfg, args)
    robust_cfg = cfg.get("robust", {})
    for key, value in robust_cfg.items():
        if hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, value)
    apply_robust_method_flags(args)
    print_robust_method_config(args)

    seed = int(cfg.get("training", {}).get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    require_cuda_for_sam3()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device is not None and device.type == "cuda":
        torch.cuda.set_device(args.device)
    out_dir = Path(cfg["output"]["output_dir"])
    save_run_artifacts(out_dir, args.config, cfg)

    prompt_config_path = Path(cfg["prompts"]["config_path"])
    if not prompt_config_path.exists():
        prompt_config_path = resolve_config_relative_path(Path(args.config).resolve(), cfg["prompts"]["config_path"])
    dataset_key = cfg["prompts"]["dataset_key"]
    prompt_manager = PromptManager(prompt_config_path, dataset_key, warning_fn=print_rank0)
    prompt_groups = load_prompt_groups(prompt_config_path, dataset_key, prompt_manager)
    prompt_group_names = [item.strip() for item in args.prompt_group.split(",") if item.strip()]

    model_cfg = cfg.get("model", {})
    base_model = build_sam3_image_model(
        device=device.type,
        compile=False,
        checkpoint_path=model_cfg.get("checkpoint_path"),
        load_from_HF=model_cfg.get("load_from_hf", model_cfg.get("checkpoint_path") is None),
        bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        eval_mode=False,
    )
    lora_cfg = cfg["lora"]
    base_model = apply_lora_to_model(
        base_model,
        LoRAConfig(
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
        ),
    )
    pretrained_lora = args.weights or cfg.get("output", {}).get("pretrained_lora_weights")
    if pretrained_lora:
        print_rank0(f"Loading initial LoRA weights from: {pretrained_lora}")
        load_lora_weights(base_model, pretrained_lora)

    cpca_cfg = {}
    if args.cpca_variant == "v2":
        cpca_cfg["residual_scale"] = float(args.cpca_v2_residual_scale)
    model = MedSAM3PromptRobustWrapper(
        base_model,
        use_cpca=args.use_cpca,
        use_gpsma=args.use_gpsma,
        cpca_variant=args.cpca_variant or "v1",
        text_embed_dim=args.text_embed_dim,
        mask_channels=args.mask_channels,
        cpca_cfg=cpca_cfg,
        auto_apply_gpsma=False,
    ).to(device)
    param_groups = configure_trainable_params(model, args)
    optimizer = AdamW(param_groups, weight_decay=float(cfg["training"].get("weight_decay", 0.01))) if param_groups else None

    train_ds = PromptRobustCOCODataset(
        data_dir=cfg["training"]["data_dir"],
        split="train",
        prompt_manager=prompt_manager,
        prompt_groups=prompt_groups,
        robust_method=args.robust_method,
        prompt_group_names=prompt_group_names,
        num_group_prompts=args.num_group_prompts,
        canonical_prompt_override=args.canonical_prompt,
        synonym_prompts_override=args.synonym_prompts.split(",") if args.synonym_prompts else None,
    )
    loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"].get("batch_size", 1)),
        shuffle=True,
        collate_fn=lambda batch: collate_fn_api(batch, dict_key="input", with_seg_masks=True),
        num_workers=int(cfg["training"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    use_amp = device.type == "cuda" and str(cfg["training"].get("mixed_precision", "")).lower() in {"fp16", "bf16", "true"}
    amp_dtype = torch.bfloat16 if str(cfg["training"].get("mixed_precision", "")).lower() == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    grad_accum = max(1, int(cfg["training"].get("gradient_accumulation_steps", 1)))
    epochs = int(cfg["training"].get("num_epochs", cfg["training"].get("epochs", 20)))
    best_mean_dice = -1.0
    log_path = out_dir / "train_log.jsonl"
    model.train()
    cpca_debug_printed = False
    cpca_warning_printed = False
    cpca_prototype_warning_printed = False

    for epoch in range(epochs):
        totals = defaultdict(float)
        steps = 0
        optimizer.zero_grad(set_to_none=True) if optimizer is not None else None
        pbar = tqdm(loader, desc=f"{args.robust_method} epoch {epoch + 1}/{epochs}", disable=not is_main_process())
        for batch_idx, batch_dict in enumerate(pbar):
            input_batch = move_to_device(batch_dict["input"], device)
            keys = query_group_keys(input_batch, prompt_manager, train_ds.categories)
            set_cpca_context(
                model,
                num_group_prompts=infer_num_group_prompts_from_keys(keys),
                batch_size=infer_batch_size(input_batch),
                canonical_prompt_index=0,
            )
            reset_cpca_debug_state(model)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                outputs_list = model(input_batch)
                cpca_debug = get_cpca_debug_state(model)
                if args.use_cpca and not cpca_debug["cpca_hook_used"] and not cpca_warning_printed:
                    print_rank0("[CPCA] Warning: CPCA enabled but text embedding was not calibrated.")
                    cpca_warning_printed = True
                if (
                    args.use_cpca
                    and cpca_debug["cpca_hook_used"]
                    and (
                        not cpca_debug["cpca_canonical_from_current_group"]
                        or not cpca_debug["cpca_group_from_current_group"]
                    )
                    and not cpca_prototype_warning_printed
                ):
                    print_rank0("[CPCA] Warning: CPCA enabled but canonical/group prototypes are unavailable.")
                    cpca_prototype_warning_printed = True
                if (
                    args.use_cpca
                    and cpca_debug["cpca_canonical_from_current_group"]
                    and cpca_debug["cpca_group_from_current_group"]
                    and not cpca_debug_printed
                ):
                    print_rank0(
                        "[CPCA] Using current prompt group as prototype bank: "
                        f"text_emb={cpca_debug['cpca_text_embedding_shape']}, "
                        f"canonical={cpca_debug['canonical_embedding_shape']}, "
                        f"group={cpca_debug['group_embedding_shape']}, "
                        f"layout={cpca_debug['cpca_layout']}"
                    )
                    cpca_debug_printed = True
                with SAM3Output.iteration_mode(outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE) as outputs_iter:
                    final_outputs = list(outputs_iter)[-1][-1]
                flat_logits = select_query_mask_logits(final_outputs)
                flat_targets = build_query_gt_masks(input_batch, final_outputs)
                group_batches = build_prompt_group_batches(flat_logits, flat_targets, keys)
                group_loss_payload = compute_prompt_group_losses(
                    model,
                    group_batches,
                    args.use_gpsma,
                    args.use_group_consistency,
                    args.use_worst_loss,
                    args.worst_tau,
                )
                loss_seg = group_loss_payload["loss_seg"]
                loss_group = group_loss_payload["loss_group"]
                loss_worst = group_loss_payload["loss_worst"]
                if args.use_cpca and args.use_proto_loss:
                    loss_proto, proto_loss_valid, proto_loss_num_prompts = compute_proto_loss(model)
                else:
                    loss_proto = flat_logits.new_zeros(())
                    proto_loss_valid = False
                    proto_loss_num_prompts = 0
                loss_total = (
                    loss_seg
                    + args.group_loss_weight * loss_group
                    + args.proto_loss_weight * loss_proto
                    + args.worst_loss_weight * loss_worst
                )
                loss_for_backward = loss_total / grad_accum

            if optimizer is not None:
                scaler.scale(loss_for_backward).backward()
                if (batch_idx + 1) % grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                per_prompt_dice_values = group_loss_payload["per_prompt_dice"]
                mean_dice = float(sum(per_prompt_dice_values) / len(per_prompt_dice_values)) if per_prompt_dice_values else 0.0
                worst_dice = float(min(per_prompt_dice_values)) if per_prompt_dice_values else 0.0
                best_dice = float(max(per_prompt_dice_values)) if per_prompt_dice_values else 0.0
                prg = best_dice - worst_dice
                prompt_list = group_loss_payload["prompt_list"]
                payload = {
                    "epoch": epoch + 1,
                    "batch": batch_idx,
                    "loss_total": float(loss_total.detach().item()),
                    "loss_seg": float(loss_seg.detach().item()),
                    "loss_group": float(loss_group.detach().item()),
                    "loss_proto": float(loss_proto.detach().item()),
                    "loss_worst": float(loss_worst.detach().item()),
                    "prompt_list": prompt_list,
                    "per_prompt_dice": [round(float(x), 6) for x in per_prompt_dice_values],
                    "worst_prompt_dice": worst_dice,
                    "best_prompt_dice": best_dice,
                    "PRG": prg,
                    "group_logits_shape": [list(shape) for shape in group_loss_payload["group_logits_shapes"]],
                    "refined_group_logits_shape": [list(shape) for shape in group_loss_payload["refined_group_logits_shapes"]],
                    "gpsma_group_logits_used": bool(group_loss_payload["gpsma_group_logits_used"]),
                    "cpca_variant": cpca_debug["cpca_variant"],
                    "cpca_hook_used": bool(cpca_debug["cpca_hook_used"]),
                    "cpca_text_embedding_shape": cpca_debug["cpca_text_embedding_shape"],
                    "cpca_layout": cpca_debug["cpca_layout"],
                    "cpca_num_prompt_slots": cpca_debug["cpca_num_prompt_slots"],
                    "cpca_canonical_from_current_group": bool(cpca_debug["cpca_canonical_from_current_group"]),
                    "cpca_group_from_current_group": bool(cpca_debug["cpca_group_from_current_group"]),
                    "canonical_embedding_shape": cpca_debug["canonical_embedding_shape"],
                    "group_embedding_shape": cpca_debug["group_embedding_shape"],
                    "cpca_calibrated_embedding_shape": cpca_debug["cpca_calibrated_embedding_shape"],
                    "proto_loss_valid": bool(proto_loss_valid),
                    "proto_loss_num_prompts": int(proto_loss_num_prompts),
                    "cpca_v2_active": bool(cpca_debug["cpca_v2_active"]),
                    "cpca_v2_reliability_mean": cpca_debug["cpca_v2_reliability_mean"],
                    "cpca_v2_reliability_min": cpca_debug["cpca_v2_reliability_min"],
                    "cpca_v2_reliability_max": cpca_debug["cpca_v2_reliability_max"],
                    "cpca_v2_calibration_strength_mean": cpca_debug["cpca_v2_calibration_strength_mean"],
                    "cpca_v2_calibration_delta_norm": cpca_debug["cpca_v2_calibration_delta_norm"],
                }
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload) + "\n")
                for key in ("loss_total", "loss_seg", "loss_group", "loss_proto", "loss_worst"):
                    totals[key] += payload[key]
                totals["mean_dice"] += mean_dice
                totals["wpd"] += worst_dice
                totals["best_prompt_dice"] += best_dice
                totals["prg"] += prg
                steps += 1
                pbar.set_postfix(
                    loss=f"{payload['loss_total']:.4f}",
                    seg=f"{payload['loss_seg']:.4f}",
                    group=f"{payload['loss_group']:.4f}",
                    worst=f"{payload['loss_worst']:.4f}",
                    dice=f"{mean_dice:.4f}",
                )

        if optimizer is not None and len(loader) % grad_accum != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        denom = max(1, steps)
        epoch_mean_dice = totals["mean_dice"] / denom
        epoch_wpd = totals["wpd"] / denom
        epoch_prg = totals["prg"] / denom
        print_rank0(
            f"Epoch {epoch + 1}/{epochs}: loss_total={totals['loss_total']/denom:.6f} "
            f"loss_seg={totals['loss_seg']/denom:.6f} loss_group={totals['loss_group']/denom:.6f} "
            f"loss_proto={totals['loss_proto']/denom:.6f} loss_worst={totals['loss_worst']/denom:.6f} "
            f"mean_dice={epoch_mean_dice:.4f} wpd={epoch_wpd:.4f} prg={epoch_prg:.4f}"
        )
        compact_model_state = get_trainable_state_dict(model)
        ckpt = {
            "epoch": epoch + 1,
            "method": args.robust_method,
            "checkpoint_format": "trainable_state_dict",
            "model": compact_model_state,
            "optimizer": None,
            "config": cfg,
            "metrics": {"mean_dice": epoch_mean_dice, "wpd": epoch_wpd, "prg": epoch_prg},
        }
        last_path = out_dir / "model_last.pth"
        safe_torch_save(ckpt, last_path)
        save_lora_weights(model, str(out_dir / "last_lora_weights.pt"))
        if epoch_mean_dice >= best_mean_dice:
            best_mean_dice = epoch_mean_dice
            best_name = f"model_best_{args.robust_method}_ep{epoch + 1}_Dice_{epoch_mean_dice:.4f}_WPD_{epoch_wpd:.4f}_PRG_{epoch_prg:.4f}.pth"
            best_path = out_dir / best_name
            if safe_torch_save(ckpt, best_path):
                try:
                    shutil.copy2(best_path, out_dir / "model_best.pth")
                except Exception as exc:
                    print_rank0(f"[Checkpoint Warning] Failed to copy model_best.pth: {exc}")
            save_lora_weights(model, str(out_dir / "best_lora_weights.pt"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MedSAM3 prompt robustness adapters")
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--robust_method", choices=ROBUST_METHODS, default="full")
    parser.add_argument("--use_cpca", action="store_true")
    parser.add_argument("--cpca_variant", choices=["v1", "v2"], default=None)
    parser.add_argument("--use_gpsma", action="store_true")
    parser.add_argument("--use_group_consistency", action="store_true")
    parser.add_argument("--use_worst_loss", action="store_true")
    parser.add_argument("--use_proto_loss", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prompt_group", default="canonical,synonyms,lexical_variations,challenging")
    parser.add_argument("--prompts_config", default=None)
    parser.add_argument("--prompt_dataset_key", default=None)
    parser.add_argument("--canonical_prompt", default=None)
    parser.add_argument("--synonym_prompts", default=None)
    parser.add_argument("--num_group_prompts", type=int, default=2)
    parser.add_argument("--group_loss_weight", type=float, default=0.1)
    parser.add_argument("--proto_loss_weight", type=float, default=0.05)
    parser.add_argument("--worst_loss_weight", type=float, default=0.1)
    parser.add_argument("--worst_tau", type=float, default=0.5)
    parser.add_argument("--cpca_v2_residual_scale", type=float, default=0.05)
    parser.add_argument("--cpca_v2_semantic_loss_weight", type=float, default=0.0)
    parser.add_argument("--cpca_lr", type=float, default=1e-4)
    parser.add_argument("--gpsma_lr", type=float, default=1e-4)
    parser.add_argument("--lora_lr", type=float, default=1e-4)
    parser.add_argument("--decoder_lr", type=float, default=1e-5)
    parser.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train_detector", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train_cpca", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train_gpsma", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--text_embed_dim", type=int, default=None)
    parser.add_argument("--mask_channels", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
