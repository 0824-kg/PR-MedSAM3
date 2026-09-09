#!/usr/bin/env python3
"""Experiment 4: single-prompt cross-concept/irrelevant proxy specificity."""

from __future__ import annotations

import argparse
import gc
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from scripts.analysis_utils import limit_ids, require_empty_stage, require_preflight
from scripts.config_utils import load_config, resolve_project_path
from scripts.inference_core import InferenceSession
from scripts.io_utils import command_string, sha256_file, utc_now, write_csv, write_json
from scripts.metric_utils_mp import foreground_activation_ratio, foreground_suppression_ratio
from scripts.prompt_inventory_utils import (
    canonical_inventory_digest, ensure_frozen_csv, ensure_frozen_json,
    ensure_frozen_text, inventory_rows, supplemental_inventory,
)


METHODS = ("MedSAM3", "MedSAM3-FT-Canon", "Group-Consistency", "Proposed")
PROMPT_TYPES = ("valid", "cross_concept", "irrelevant")


def freeze_inventory(config: dict, output_root: Path) -> tuple[dict, str]:
    path = resolve_project_path(config["prompt_files"]["specificity"])
    rows = inventory_rows(path)
    digest = canonical_inventory_digest(path)
    duplicates = []
    _, inventory = supplemental_inventory(path)
    for dataset, groups in inventory.items():
        seen_values = set()
        for category, prompts in groups.items():
            for prompt in prompts:
                if prompt in seen_values:
                    duplicates.append((dataset, category, prompt))
                seen_values.add(prompt)
    if duplicates:
        raise RuntimeError(f"specificity prompt duplicates detected: {duplicates}")
    raw_sha = sha256_file(path)
    manifest_dir = output_root / "manifests"
    ensure_frozen_csv(manifest_dir / "specificity_prompt_inventory.csv", rows, list(rows[0]))
    ensure_frozen_text(
        manifest_dir / "specificity_prompt_inventory.sha256",
        f"{digest}  canonical_normalized_specificity_inventory\n{raw_sha}  {path}\n",
    )
    ensure_frozen_json(manifest_dir / "specificity_prompt_freeze.json", {
        "fixed_utc": utc_now(), "source_file": str(path), "source_sha256": raw_sha,
        "normalized_inventory_sha256": digest, "duplicate_count": 0,
        "fixed_before_inference": True,
    }, volatile_fields={"fixed_utc"})
    return inventory, digest


def aggregate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["method"], row["image_id"], row["prompt_type"])].append(row)
    per_image = []
    for (dataset, method, image_id, prompt_type), values in sorted(groups.items()):
        per_image.append({
            "dataset": dataset, "method": method, "image_id": image_id, "prompt_type": prompt_type,
            "prompt_count": len(values), "mean_target_Dice": float(np.mean([float(v["dice"]) for v in values])),
            "mean_predicted_area": float(np.mean([float(v["predicted_area"]) for v in values])),
            "mean_foreground_ratio": float(np.mean([float(v["predicted_foreground_ratio"]) for v in values])),
            "empty_prediction_rate": float(np.mean([int(v["empty_prediction"]) for v in values])),
            "mean_connected_components": float(np.mean([int(v["connected_components"]) for v in values])),
            "mean_foreground_probability": float(np.mean([float(v["mean_foreground_probability"]) for v in values])),
            "maximum_foreground_probability": float(np.max([float(v["maximum_foreground_probability"]) for v in values])),
        })
    summary = []
    for dataset in sorted({r["dataset"] for r in rows}):
        for method in METHODS:
            chosen = [r for r in per_image if r["dataset"] == dataset and r["method"] == method]
            by_type = {kind: [r for r in chosen if r["prompt_type"] == kind] for kind in PROMPT_TYPES}
            valid = by_type["valid"]
            cross = by_type["cross_concept"]
            irrelevant = by_type["irrelevant"]
            valid_dice = float(np.mean([r["mean_target_Dice"] for r in valid]))
            cross_dice = float(np.mean([r["mean_target_Dice"] for r in cross]))
            irrelevant_dice = float(np.mean([r["mean_target_Dice"] for r in irrelevant]))
            valid_area = [r["mean_predicted_area"] for r in valid]
            cross_area = [r["mean_predicted_area"] for r in cross]
            irrelevant_area = [r["mean_predicted_area"] for r in irrelevant]
            summary.append({
                "dataset": dataset, "method": method, "n_images": len(valid),
                "Valid_prompt_Dice": valid_dice,
                "Cross_concept_target_Dice": cross_dice,
                "Irrelevant_prompt_target_Dice": irrelevant_dice,
                "Cross_concept_foreground_ratio": float(np.mean([r["mean_foreground_ratio"] for r in cross])),
                "Irrelevant_foreground_ratio": float(np.mean([r["mean_foreground_ratio"] for r in irrelevant])),
                "Cross_concept_empty_rate": float(np.mean([r["empty_prediction_rate"] for r in cross])),
                "Irrelevant_empty_rate": float(np.mean([r["empty_prediction_rate"] for r in irrelevant])),
                "Specificity_Dice_Gap": valid_dice - cross_dice,
                "Irrelevant_Dice_Gap": valid_dice - irrelevant_dice,
                "Cross_concept_Activation_Ratio": foreground_activation_ratio(cross_area, valid_area),
                "Cross_concept_Foreground_Suppression_Ratio": foreground_suppression_ratio(cross_area, valid_area),
                "Irrelevant_Activation_Ratio": foreground_activation_ratio(irrelevant_area, valid_area),
                "Irrelevant_Foreground_Suppression_Ratio": foreground_suppression_ratio(irrelevant_area, valid_area),
                "runtime_ms_per_image": float(sum(float(r["runtime_ms"]) for r in rows if r["dataset"] == dataset and r["method"] == method) / len(valid)),
                "peak_memory_mb": float(max(float(r["peak_memory_mb"]) for r in rows if r["dataset"] == dataset and r["method"] == method)),
                "interpretation": "cross-concept/irrelevant target-suppression proxy; not clinical false-positive specificity",
            })
    return per_image, summary


def plot_results(summary: list[dict], stage: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [f"{r['dataset']}\n{r['method']}" for r in summary]
    x = np.arange(len(labels))
    specs = [
        (("Cross_concept_foreground_ratio", "Irrelevant_foreground_ratio"), "valid_vs_negative_foreground_ratio.pdf", "Foreground ratio"),
        (("Cross_concept_empty_rate", "Irrelevant_empty_rate"), "valid_vs_negative_empty_rate.pdf", "Empty rate"),
        (("Specificity_Dice_Gap", "Irrelevant_Dice_Gap"), "specificity_gap.pdf", "Target Dice gap"),
    ]
    for metrics, filename, ylabel in specs:
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * .65), 4.5))
        width = .38
        ax.bar(x - width / 2, [float(r[metrics[0]]) for r in summary], width, label=metrics[0])
        ax.bar(x + width / 2, [float(r[metrics[1]]) for r in summary], width, label=metrics[1])
        ax.set_ylabel(ylabel); ax.set_xticks(x, labels, rotation=50, ha="right", fontsize=7)
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=.25)
        fig.tight_layout(); fig.savefig(stage / filename); plt.close(fig)


def run(config_path: Path, output_root: Path, num_samples: int | None) -> dict:
    require_preflight(output_root)
    if not (output_root / "03_heldout_prompts" / "run_metadata.json").is_file():
        raise FileNotFoundError("Experiment 3 must complete before specificity prompts are frozen")
    stage = require_empty_stage(output_root, "04_prompt_specificity")
    _, config = load_config(config_path)
    inventory, digest = freeze_inventory(config, output_root)
    rows = []
    started = utc_now()
    for dataset_name, dataset_cfg in config["datasets"].items():
        for method in METHODS:
            method_cfg = dataset_cfg["methods"][method]
            session = InferenceSession(
                config_path=resolve_project_path(method_cfg["config"]),
                checkpoint_path=resolve_project_path(method_cfg["checkpoint"]), method=method,
                annotation_file=resolve_project_path(dataset_cfg["annotation_file"]),
                image_root=resolve_project_path(dataset_cfg["image_root"]),
                amp_dtype=config["protocol"]["amp_dtype"],
                empty_dense_logit=config["protocol"]["empty_dense_logit"],
                seed=config["protocol"]["inference_seed"],
            )
            if len(session.dataset.image_ids) != int(dataset_cfg["expected_test_count"]):
                raise RuntimeError(f"test count changed after preflight for {dataset_name}")
            for image_id in limit_ids(session.dataset.image_ids, num_samples):
                for prompt_type in PROMPT_TYPES:
                    for prompt in inventory[dataset_name][prompt_type]:
                        result = session.forward_independent(image_id, [prompt])
                        row = session.rows(image_id, result)[0]
                        embedding = result.embedding["per_prompt"][0]
                        rows.append({
                            "dataset": dataset_name, "split": "test", "method": method,
                            "image_id": image_id, "prompt_type": prompt_type,
                            "inference_protocol": "single_prompt_K1_no_canonical_leakage", **row,
                            "runtime_ms": result.runtime_ms, "peak_memory_mb": result.peak_memory_mb,
                            "checkpoint": str(session.checkpoint_path),
                            "checkpoint_sha256": session.checkpoint_sha256_before,
                            "model_config_sha256": session.model_config_sha256,
                            "seed": session.seed,
                            "specificity_inventory_sha256": digest,
                            "prompt_embedding_sha256": embedding.get("sha256", ""),
                            "prompt_embedding_shape": str(embedding.get("shape", "")),
                        })
            session.verify_checkpoint_unchanged()
            torch_module = session.lib["torch"]
            del session
            gc.collect()
            torch_module.cuda.empty_cache()
    per_image, summary = aggregate(rows)
    if canonical_inventory_digest(resolve_project_path(config["prompt_files"]["specificity"])) != digest:
        raise RuntimeError("specificity prompt inventory changed after it was frozen")
    write_csv(stage / "specificity_per_prompt.csv", rows, list(rows[0]))
    write_csv(stage / "specificity_per_image.csv", per_image, list(per_image[0]))
    write_csv(stage / "specificity_summary.csv", summary, list(summary[0]))
    plot_results(summary, stage)
    meta = {"status": "PASS", "started_utc": started, "finished_utc": utc_now(), "command": command_string(), "num_samples": num_samples, "formal_complete": num_samples is None, "inventory_fixed_before_inference": True, "specificity_inventory_sha256": digest, "config_sha256": sha256_file(config_path), "proxy_analysis_only": True}
    write_json(stage / "run_metadata.json", meta)
    print("PROMPT_SPECIFICITY = PASS" if num_samples is None else "PROMPT_SPECIFICITY_SMOKE = PASS")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-samples", type=int)
    args = parser.parse_args()
    run(Path(args.config).resolve(), Path(args.output_root).resolve(), args.num_samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
