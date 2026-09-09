#!/usr/bin/env python3
"""Experiment 3: frozen held-out prompt-expression evaluation."""

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

from scripts.analysis_utils import image_category_rows, limit_ids, require_empty_stage, require_preflight
from scripts.config_utils import load_config, resolve_project_path
from scripts.inference_core import InferenceSession
from scripts.io_utils import command_string, sha256_file, utc_now, write_csv, write_json
from scripts.legacy_seen_utils import load_seen_rows
from scripts.prompt_inventory_utils import (
    archived_seen_inventory, canonical_inventory_digest, collision_report, inventory_rows,
    ensure_frozen_csv, ensure_frozen_json, ensure_frozen_text, supplemental_inventory,
)


METHODS = ("MedSAM3", "MedSAM3-FT-Canon", "Group-Consistency", "Proposed")
HELDOUT_CATEGORIES = ("heldout_synonyms", "heldout_lexical", "heldout_descriptive")


def freeze_inventory(config: dict, output_root: Path) -> tuple[dict, str]:
    seen_path = resolve_project_path(config["prompt_files"]["seen"])
    heldout_path = resolve_project_path(config["prompt_files"]["heldout"])
    rows = inventory_rows(heldout_path)
    collisions = collision_report(seen_path, heldout_path, exclude_categories={"canonical"})
    manifest_dir = output_root / "manifests"
    ensure_frozen_csv(manifest_dir / "heldout_prompt_inventory.csv", rows, list(rows[0]))
    collision_fields = ["dataset", "category", "prompt", "collision_type"]
    ensure_frozen_csv(
        manifest_dir / "heldout_prompt_collision_report.csv", collisions, collision_fields
    )
    digest = canonical_inventory_digest(heldout_path)
    raw_sha = sha256_file(heldout_path)
    ensure_frozen_text(
        manifest_dir / "heldout_prompt_inventory.sha256",
        f"{digest}  canonical_normalized_heldout_inventory\n{raw_sha}  {heldout_path}\n",
    )
    ensure_frozen_json(manifest_dir / "heldout_prompt_freeze.json", {
        "fixed_utc": utc_now(), "source_file": str(heldout_path),
        "source_sha256": raw_sha, "normalized_inventory_sha256": digest,
        "collision_count": len(collisions), "fixed_before_inference": True,
    }, volatile_fields={"fixed_utc"})
    if collisions:
        raise RuntimeError("held-out prompt collision detected; inference stopped")
    _, inventory = supplemental_inventory(heldout_path)
    return inventory, digest


def aggregate(rows: list[dict], seen_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    per_image = image_category_rows(rows)
    summaries = []
    for dataset in sorted({r["dataset"] for r in rows}):
        for method in METHODS:
            selected = [r for r in per_image if r["dataset"] == dataset and r["method"] == method]
            item = {"dataset": dataset, "method": method, "n_images": len({r["image_id"] for r in selected})}
            for category in HELDOUT_CATEGORIES:
                cat = [r for r in selected if r["category"] == category]
                item[f"{category}_WPD"] = float(np.mean([float(r["WPD"]) for r in cat]))
                item[f"{category}_PRG"] = float(np.mean([float(r["PRG"]) for r in cat]))
            item["heldout_Avg_WPD"] = float(np.mean([item[f"{c}_WPD"] for c in HELDOUT_CATEGORIES]))
            item["heldout_Avg_PRG"] = float(np.mean([item[f"{c}_PRG"] for c in HELDOUT_CATEGORIES]))
            prompt_selected = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            item["prompt_wise_mean_Dice"] = float(np.mean([float(r["dice"]) for r in prompt_selected]))
            item["empty_prediction_rate"] = float(np.mean([int(r["empty_prediction"]) for r in prompt_selected]))
            item["runtime_ms_per_image"] = float(
                sum(float(r["runtime_ms"]) for r in prompt_selected) / item["n_images"]
            )
            item["peak_memory_mb"] = float(max(float(r["peak_memory_mb"]) for r in prompt_selected))
            summaries.append(item)

    comparisons = []
    seen_grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in seen_rows:
        seen_grouped[(row["dataset"], row["method"], row["image_id"], row["category"])].append(row)
    held_grouped = {(r["dataset"], r["method"], r["image_id"], r["category"]): r for r in per_image}
    mapping = {"heldout_synonyms": "synonyms", "heldout_lexical": "lexical", "heldout_descriptive": "challenging"}
    for (dataset, method, image_id, held_cat), held in sorted(held_grouped.items()):
        seen_cat = mapping[held_cat]
        values = seen_grouped.get((dataset, method, image_id, seen_cat))
        if not values:
            raise RuntimeError(f"missing paired archived seen result: {dataset} {method} {image_id} {seen_cat}")
        dice = np.asarray([float(v["dice"]) for v in values])
        seen_wpd, seen_prg = float(dice.min()), float(dice.max() - dice.min())
        comparisons.append({
            "dataset": dataset, "method": method, "image_id": image_id,
            "heldout_category": held_cat, "paired_seen_category": seen_cat,
            "seen_WPD": seen_wpd, "heldout_WPD": float(held["WPD"]),
            "seen_to_heldout_WPD_drop": seen_wpd - float(held["WPD"]),
            "seen_PRG": seen_prg, "heldout_PRG": float(held["PRG"]),
            "seen_to_heldout_PRG_change": float(held["PRG"]) - seen_prg,
        })
    return per_image, summaries, comparisons


def plot_results(summary: list[dict], stage: Path) -> None:
    import matplotlib.pyplot as plt

    for metric, filename in (("heldout_Avg_WPD", "seen_heldout_wpd.pdf"), ("heldout_Avg_PRG", "seen_heldout_prg.pdf")):
        labels = [f"{r['dataset']}\n{r['method']}" for r in summary]
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * .65), 4.5))
        ax.bar(np.arange(len(labels)), [float(r[metric]) for r in summary], color="#8172b3")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_xticks(np.arange(len(labels)), labels, rotation=50, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout(); fig.savefig(stage / filename); plt.close(fig)


def run(config_path: Path, output_root: Path, num_samples: int | None) -> dict:
    require_preflight(output_root)
    if not (output_root / "02_prompt_ensemble" / "run_metadata.json").is_file():
        raise FileNotFoundError("Experiment 2 must complete before held-out inventory is frozen")
    stage = require_empty_stage(output_root, "03_heldout_prompts")
    _, config = load_config(config_path)
    heldout, digest = freeze_inventory(config, output_root)
    seen = archived_seen_inventory(resolve_project_path(config["prompt_files"]["seen"]))
    rows, all_seen_rows, seen_reference_rows = [], [], []
    started = utc_now()
    for dataset_name, dataset_cfg in config["datasets"].items():
        canonical = heldout[dataset_name]["canonical"][0]
        seen_mode = dataset_cfg["seen_reference_mode"]
        for method in METHODS:
            method_cfg = dataset_cfg["methods"][method]
            if seen_mode == "archived_csv":
                archived, _ = load_seen_rows(
                    resolve_project_path(method_cfg["seen_result_dir"]), seen[dataset_name],
                    int(dataset_cfg["expected_test_count"]),
                )
                standardized = [
                    {
                        "dataset": dataset_name, "method": method, "split": "test",
                        "mode": "archived_csv", "metric_role": "evaluation",
                        "reference_provenance": "archived_main_experiment_csv",
                        "checkpoint": str(resolve_project_path(method_cfg["checkpoint"])),
                        "checkpoint_sha256": sha256_file(resolve_project_path(method_cfg["checkpoint"])),
                        "model_config_sha256": sha256_file(resolve_project_path(method_cfg["config"])),
                        "seed": "ARCHIVED_METADATA", **r,
                    }
                    for r in archived
                ]
                all_seen_rows.extend(standardized)
                seen_reference_rows.extend(standardized)
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
            selected_ids = limit_ids(session.dataset.image_ids, num_samples)
            if seen_mode == "no_retraining_reevaluation":
                for image_id in selected_ids:
                    for seen_category in ("canonical", "synonyms", "lexical", "challenging"):
                        seen_prompts = seen[dataset_name][seen_category]
                        seen_result = session.forward_archived_group(image_id, seen_prompts)
                        for seen_row in session.rows(image_id, seen_result):
                            standardized = {
                                "dataset": dataset_name, "method": method, "split": "test",
                                "mode": "no_retraining_reevaluation", "metric_role": "evaluation",
                                "reference_provenance": "new_fixed_inventory_checkpoint_only_evaluation",
                                "category": seen_category, **seen_row,
                                "runtime_ms_group_total": seen_result.runtime_ms,
                                "peak_memory_mb": seen_result.peak_memory_mb,
                                "checkpoint": str(session.checkpoint_path),
                                "checkpoint_sha256": session.checkpoint_sha256_before,
                                "model_config_sha256": session.model_config_sha256,
                                "seed": session.seed,
                                "seen_inventory_source": str(resolve_project_path(config["prompt_files"]["seen"])),
                                "seen_inventory_sha256": sha256_file(resolve_project_path(config["prompt_files"]["seen"])),
                            }
                            all_seen_rows.append(standardized)
                            seen_reference_rows.append(standardized)
            for image_id in selected_ids:
                for category in HELDOUT_CATEGORIES:
                    for prompt in heldout[dataset_name][category]:
                        if method == "Proposed":
                            result = session.forward_group(image_id, [canonical, prompt], corrected=True)
                            row = session.rows(image_id, result)[1]
                            branch = "corrected_k2_heldout_branch"
                            embedding = result.embedding
                        else:
                            result = session.forward_independent(image_id, [prompt])
                            row = session.rows(image_id, result)[0]
                            branch = "single_prompt"
                            embedding = result.embedding["per_prompt"][0]
                        rows.append({
                            "dataset": dataset_name, "split": "test", "method": method,
                            "mode": "heldout", "image_id": image_id, "category": category,
                            "metric_role": "evaluation", "branch": branch, **row,
                            "runtime_ms": result.runtime_ms, "peak_memory_mb": result.peak_memory_mb,
                            "checkpoint": str(session.checkpoint_path),
                            "checkpoint_sha256": session.checkpoint_sha256_before,
                            "model_config_sha256": session.model_config_sha256,
                            "seed": session.seed,
                            "heldout_inventory_sha256": digest,
                            "prompt_embedding_sha256": embedding.get("sha256", ""),
                            "prompt_embedding_shape": str(embedding.get("shape", "")),
                        })
            session.verify_checkpoint_unchanged()
            torch_module = session.lib["torch"]
            del session
            gc.collect()
            torch_module.cuda.empty_cache()
    per_image, summary, comparison = aggregate(rows, all_seen_rows)
    if canonical_inventory_digest(resolve_project_path(config["prompt_files"]["heldout"])) != digest:
        raise RuntimeError("held-out prompt inventory changed after it was frozen")
    write_csv(stage / "heldout_per_prompt.csv", rows, list(rows[0]))
    write_csv(stage / "heldout_per_image.csv", per_image, list(per_image[0]))
    write_csv(stage / "heldout_summary.csv", summary, list(summary[0]))
    write_csv(stage / "seen_vs_heldout.csv", comparison, list(comparison[0]))
    seen_per_image = image_category_rows(seen_reference_rows)
    seen_manifest = []
    for dataset_name in config["datasets"]:
        for method in METHODS:
            selected = [r for r in seen_reference_rows if r["dataset"] == dataset_name and r["method"] == method]
            runtime_by_image_category = {}
            for row in selected:
                if "runtime_ms_group_total" in row:
                    key = (row["image_id"], row["category"])
                    runtime_by_image_category[key] = max(
                        runtime_by_image_category.get(key, 0.0), float(row["runtime_ms_group_total"])
                    )
            runtime_by_image = {}
            for (image_id, _category), elapsed in runtime_by_image_category.items():
                runtime_by_image[image_id] = runtime_by_image.get(image_id, 0.0) + elapsed
            seen_manifest.append({
                "dataset": dataset_name, "method": method,
                "reference_mode": config["datasets"][dataset_name]["seen_reference_mode"],
                "n_images": len({r["image_id"] for r in selected}), "row_count": len(selected),
                "source": "new inference output" if config["datasets"][dataset_name]["seen_reference_mode"] == "no_retraining_reevaluation" else "archived CSV",
                "not_original_table_result": "true" if config["datasets"][dataset_name]["seen_reference_mode"] == "no_retraining_reevaluation" else "false",
                "runtime_ms_per_image": float(np.mean(list(runtime_by_image.values()))) if runtime_by_image else "ARCHIVED_NOT_REMEASURED",
                "peak_memory_mb": float(max(float(r["peak_memory_mb"]) for r in selected if "peak_memory_mb" in r)) if any("peak_memory_mb" in r for r in selected) else "ARCHIVED_NOT_REMEASURED",
            })
    write_csv(
        stage / "seen_reference_per_prompt.csv", seen_reference_rows,
        sorted({key for row in seen_reference_rows for key in row}),
    )
    write_csv(stage / "seen_reference_per_image.csv", seen_per_image, list(seen_per_image[0]))
    write_csv(stage / "seen_reference_manifest.csv", seen_manifest, list(seen_manifest[0]))
    plot_results(summary, stage)
    meta = {
        "status": "PASS", "started_utc": started, "finished_utc": utc_now(),
        "command": command_string(), "num_samples": num_samples,
        "formal_complete": num_samples is None, "inventory_fixed_before_inference": True,
        "heldout_inventory_sha256": digest, "config_sha256": sha256_file(config_path),
        "seen_reference_policy": {
            dataset: item["seen_reference_mode"] for dataset, item in config["datasets"].items()
        },
        "isic_seen_reevaluation_is_not_original_table_result": True,
    }
    write_json(stage / "run_metadata.json", meta)
    print("HELDOUT_PROMPTS = PASS" if num_samples is None else "HELDOUT_PROMPTS_SMOKE = PASS")
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
