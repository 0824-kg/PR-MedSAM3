#!/usr/bin/env python3
"""Prompt robustness validation wrapper for MedSAM3 LoRA checkpoints."""

from __future__ import annotations

import argparse
import csv
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import yaml

from validate_sam3_lora import validate


GROUPS = ("canonical", "synonyms", "lexical_variations", "challenging")
AUTHOR_LORA_FORBIDDEN_CHECKPOINT_SUBSTRINGS = (
    "work_dir",
    "prompt_robust",
    "ft_canon",
    "ft_norm_group",
    "gpsma",
    "cpca",
    "model_best.pth",
    "model_last.pth",
)


def default_config_for_dataset(dataset_key: str) -> str:
    candidates = {
        "busi": "configs/prompt_robust_busi_full.yaml",
        "kvasir": "configs/prompt_robust_kvasir_full.yaml",
        "sessile_kvasir": "configs/prompt_robust_kvasir_full.yaml",
        "isic2018": "configs/prompt_robust_isic2018_full.yaml",
    }
    return candidates.get(dataset_key, "configs/prompt_robust_isic2018_full.yaml")


def ensure_nested_config(config_path: str, data_dir: str, dataset_key: str, prompts_config: str) -> str:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "training" in cfg and "model" in cfg and "lora" in cfg:
        return config_path
    nested = {
        "model": {
            "checkpoint_path": cfg.get("sam3_ckpt", "checkpoints/sam3.pt"),
            "load_from_hf": False,
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
            "data_dir": data_dir,
            "batch_size": cfg.get("batch_size", 1),
            "num_workers": cfg.get("num_workers", 0),
        },
        "prompts": {
            "enabled": True,
            "config_path": prompts_config,
            "dataset_key": dataset_key,
        },
    }
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.safe_dump(nested, tmp, sort_keys=False)
    tmp.close()
    return tmp.name


def resolve_val_dir(data_dir: str) -> str:
    root = Path(data_dir)
    for split in ("valid", "val", "test"):
        if (root / split / "_annotations.coco.json").exists():
            return str(root / split)
    return str(root)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_author_lora_checkpoint(checkpoint: str) -> None:
    checkpoint_lower = checkpoint.replace("\\", "/").lower()
    if any(token in checkpoint_lower for token in AUTHOR_LORA_FORBIDDEN_CHECKPOINT_SUBSTRINGS):
        raise SystemExit(
            "[AuthorLoRA] Refusing to evaluate a trained prompt-robust checkpoint as author_lora: "
            f"{checkpoint}"
        )


def write_per_image_csv(records: List[Dict[str, str]], path: Path, method: str, checkpoint: str, box_mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image_id", "prompt_group", "query_text", "dice", "iou", "method", "checkpoint", "box_mode"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "image_id": record.get("image_id", ""),
                    "prompt_group": record.get("prompt_group", ""),
                    "query_text": record.get("query_text", ""),
                    "dice": record.get("dice", "0"),
                    "iou": record.get("iou", "0"),
                    "method": method,
                    "checkpoint": checkpoint,
                    "box_mode": box_mode,
                }
            )


def write_robustness_summary(records: List[Dict[str, str]], path: Path, method: str, dataset: str, box_mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for record in records:
        grouped[record.get("prompt_group", "")].append(record)
    fields = [
        "method",
        "dataset",
        "prompt_group",
        "mean_dice",
        "wpd",
        "best_prompt_dice",
        "prg",
        "std_dice",
        "num_images",
        "num_unique_prompts",
        "num_predictions",
    ]
    if box_mode != "text_only":
        fields.extend(["text_only_dice", "loose_box_dice", "tight_box_dice", "brg"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for group, group_records in grouped.items():
            dices = [float(r.get("dice", 0.0)) for r in group_records]
            by_image = defaultdict(list)
            unique_prompts = set()
            for record in group_records:
                by_image[record.get("image_id", "")].append(float(record.get("dice", 0.0)))
                query_text = record.get("query_text", "")
                if query_text:
                    unique_prompts.add(query_text)
            worst_by_image = [min(v) for v in by_image.values() if v]
            best_by_image = [max(v) for v in by_image.values() if v]
            mean_dice = sum(dices) / len(dices) if dices else 0.0
            wpd = sum(worst_by_image) / len(worst_by_image) if worst_by_image else 0.0
            best = sum(best_by_image) / len(best_by_image) if best_by_image else 0.0
            std = 0.0
            if len(dices) > 1:
                std = (sum((x - mean_dice) ** 2 for x in dices) / len(dices)) ** 0.5
            row = {
                "method": method,
                "dataset": dataset,
                "prompt_group": group,
                "mean_dice": f"{mean_dice:.6f}",
                "wpd": f"{wpd:.6f}",
                "best_prompt_dice": f"{best:.6f}",
                "prg": f"{(best - wpd):.6f}",
                "std_dice": f"{std:.6f}",
                "num_images": len(by_image),
                "num_unique_prompts": len(unique_prompts),
                "num_predictions": len(group_records),
            }
            if box_mode != "text_only":
                row.update({"text_only_dice": "", "loose_box_dice": "", "tight_box_dice": "", "brg": ""})
            writer.writerow(row)


def run(args: argparse.Namespace) -> None:
    groups = GROUPS if args.prompt_group == "all" else (args.prompt_group,)
    config_path = args.config or default_config_for_dataset(args.dataset_key)
    config_path = ensure_nested_config(config_path, args.data_dir, args.dataset_key, args.prompts_config)
    val_dir = resolve_val_dir(args.data_dir)
    method = args.method_name or args.method or Path(args.checkpoint).parent.name
    if args.eval_mode == "author_lora":
        if method != "author_lora":
            method = "author_lora"
        validate_author_lora_checkpoint(args.checkpoint)
        print("[AuthorLoRA] clean evaluation mode")
        print("use_cpca=False")
        print("use_gpsma=False")
        print("use_group_consistency=False")
        print("use_worst_loss=False")
        print("use_proto_loss=False")
        print(f"checkpoint={args.checkpoint}")
    all_records: List[Dict[str, str]] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="prompt_robust_val_"))

    for group in groups:
        tmp_per_image = tmp_dir / f"per_image_{group}.csv"
        tmp_robust = tmp_dir / f"robust_{group}.csv"
        if args.eval_mode == "author_lora":
            robust_checkpoint = None
            lora_weights = args.checkpoint
        else:
            robust_checkpoint = args.checkpoint if Path(args.checkpoint).suffix.lower() == ".pth" else None
            lora_weights = None if robust_checkpoint is not None else args.checkpoint
        validate(
            config_path=config_path,
            weights_path=lora_weights,
            val_data_dir=val_dir,
            prompt_mode="canonical",
            prompts_config=args.prompts_config,
            prompt_dataset_key=args.dataset_key,
            save_per_image_csv=True,
            per_image_csv_path=str(tmp_per_image),
            multi_prompt_group=group,
            save_robustness_csv=True,
            robustness_csv_path=str(tmp_robust),
            save_visualizations=args.save_visualizations,
            vis_output_dir=args.vis_output_dir,
            save_pred_masks=args.save_pred_masks,
            pred_mask_dir=args.pred_mask_dir,
            selected_cases_csv=args.selected_cases_csv,
            box_mode=args.box_mode,
            run_box_dependency_eval=args.box_mode != "text_only",
            save_box_csv=args.box_mode != "text_only",
            box_csv_path=str(tmp_dir / f"box_{group}.csv"),
            robust_checkpoint_path=robust_checkpoint,
        )
        all_records.extend(read_csv(tmp_per_image))

    if args.save_per_image_csv:
        write_per_image_csv(all_records, Path(args.per_image_csv_path), method, args.checkpoint, args.box_mode)
    if args.save_robustness_csv:
        write_robustness_summary(all_records, Path(args.robustness_csv_path), method, args.dataset_key, args.box_mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate prompt robustness for MedSAM3")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--dataset_key", required=True)
    parser.add_argument("--prompts_config", default="configs/prompts.yaml")
    parser.add_argument("--prompt_group", choices=["canonical", "synonyms", "lexical_variations", "challenging", "all"], default="all")
    parser.add_argument("--save_per_image_csv", action="store_true")
    parser.add_argument("--per_image_csv_path", default="per_image_prompt_results.csv")
    parser.add_argument("--save_robustness_csv", action="store_true")
    parser.add_argument("--robustness_csv_path", default="robustness_summary.csv")
    parser.add_argument("--save_visualizations", action="store_true")
    parser.add_argument("--vis_output_dir", default=None)
    parser.add_argument("--save_pred_masks", action="store_true")
    parser.add_argument("--pred_mask_dir", default=None)
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--vis_dir", default=None)
    parser.add_argument("--selected_cases_csv", default=None)
    parser.add_argument("--box_mode", choices=["text_only", "text_plus_loose_box", "text_plus_tight_box"], default="text_only")
    parser.add_argument("--method", default=None)
    parser.add_argument("--method_name", default=None)
    parser.add_argument("--eval_mode", choices=["prompt_robust", "author_lora"], default="prompt_robust")
    args = parser.parse_args()
    if args.save_vis:
        args.save_visualizations = True
    if args.vis_dir is not None:
        args.vis_output_dir = args.vis_dir
    return args


if __name__ == "__main__":
    run(parse_args())
