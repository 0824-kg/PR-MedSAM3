"""Wrapper adapters for prompt-robust MedSAM3 experiments."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple
import types

import torch
from torch import Tensor, nn

from .prompt_robust_modules import (
    ConceptPrototypeCalibrationAdapter,
    ReliabilityAwareConceptCalibrationAdapter,
    GroupPromptStableMaskAdapter,
)


TEXT_EMBED_KEYS = (
    "text_emb",
    "text_embedding",
    "prompt_emb",
    "prompt_embeddings",
    "concept_emb",
    "language_emb",
)
CANONICAL_KEYS = ("canonical_emb", "canonical_text_emb", "canonical_prompt_emb")
GROUP_EMB_KEYS = ("group_embs", "group_text_embs", "prompt_group_embs")
MASK_KEYS = ("mask_logits", "pred_masks", "masks", "low_res_masks")


class MedSAM3PromptRobustWrapper(nn.Module):
    """Non-invasive wrapper around a MedSAM3/SAM3 model.

    The wrapper supports explicit embedding kwargs, but also patches
    ``base_model.backbone.forward_text`` when available so SAM3 native
    ``model(input_batch)`` calls can apply CPCA to ``language_features`` without
    editing SAM3 source files.
    """

    def __init__(
        self,
        base_model: nn.Module,
        use_cpca: bool = False,
        use_gpsma: bool = False,
        text_embed_dim: Optional[int] = None,
        mask_channels: int = 1,
        cpca_variant: str = "v1",
        cpca_cfg: Optional[Dict[str, Any]] = None,
        gpsma_cfg: Optional[Dict[str, Any]] = None,
        auto_apply_gpsma: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        self.use_cpca = use_cpca
        self.use_gpsma = use_gpsma
        self.text_embed_dim = text_embed_dim
        self.mask_channels = mask_channels
        self.cpca_variant = cpca_variant
        self.cpca_cfg = cpca_cfg or {}
        self.gpsma_cfg = gpsma_cfg or {}
        self.auto_apply_gpsma = auto_apply_gpsma
        self._warned_mask_missing = False
        self._warned_text_missing = False
        self.cpca: Optional[ConceptPrototypeCalibrationAdapter] = None
        self.gpsma: Optional[GroupPromptStableMaskAdapter] = None
        self.last_cpca_output: Optional[Tensor] = None
        self.last_cpca_canonical: Optional[Tensor] = None
        self.cpca_hook_used = False
        self.cpca_text_embedding_shape = None
        self.cpca_layout = None
        self.cpca_num_prompt_slots = None
        self.cpca_canonical_from_current_group = False
        self.cpca_group_from_current_group = False
        self.canonical_embedding_shape = None
        self.group_embedding_shape = None
        self.cpca_calibrated_embedding_shape = None
        self.cpca_v2_active = False
        self.cpca_v2_reliability_mean = None
        self.cpca_v2_reliability_min = None
        self.cpca_v2_reliability_max = None
        self.cpca_v2_calibration_strength_mean = None
        self.cpca_v2_calibration_delta_norm = None
        self.cpca_context_num_group_prompts = None
        self.cpca_context_batch_size = None
        self.cpca_context_canonical_prompt_index = 0

        if self.use_cpca and text_embed_dim is None:
            text_embed_dim = self._infer_text_embed_dim(base_model)
            self.text_embed_dim = text_embed_dim
        if self.use_cpca and text_embed_dim is not None:
            if self.cpca_variant == "v2":
                self.cpca = ReliabilityAwareConceptCalibrationAdapter(dim=text_embed_dim, **self.cpca_cfg)
            else:
                self.cpca = ConceptPrototypeCalibrationAdapter(dim=text_embed_dim, **self.cpca_cfg)
        if self.use_gpsma:
            self.gpsma = GroupPromptStableMaskAdapter(in_channels=mask_channels, **self.gpsma_cfg)
        if self.use_cpca:
            self._patch_forward_text_if_available()

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            base_model = super().__getattr__("base_model")
            return getattr(base_model, name)

    @staticmethod
    def _find_first_key(kwargs_or_dict: Dict[str, Any], candidate_keys: Iterable[str]) -> Tuple[Optional[str], Any]:
        for key in candidate_keys:
            if key in kwargs_or_dict and kwargs_or_dict[key] is not None:
                return key, kwargs_or_dict[key]
        return None, None

    @staticmethod
    def _infer_text_embed_dim(model: nn.Module) -> Optional[int]:
        backbone = getattr(model, "backbone", None)
        language = getattr(backbone, "language_backbone", None)
        resizer = getattr(language, "resizer", None)
        if hasattr(resizer, "out_features"):
            return int(resizer.out_features)
        for name, module in model.named_modules():
            if "resizer" in name.lower() and hasattr(module, "out_features"):
                return int(module.out_features)
        return None

    def _ensure_cpca(self, dim: int, device: torch.device, dtype: torch.dtype) -> None:
        if self.cpca is None:
            if self.cpca_variant == "v2":
                self.cpca = ReliabilityAwareConceptCalibrationAdapter(dim=dim, **self.cpca_cfg)
            else:
                self.cpca = ConceptPrototypeCalibrationAdapter(dim=dim, **self.cpca_cfg)
            self.cpca.to(device=device, dtype=dtype)

    @staticmethod
    def _sam3_language_to_bnc(x: Tensor) -> Tuple[Tensor, bool]:
        # SAM3 language_features are [seq_len, num_prompts, dim]. CPCA expects
        # [B, N, C], so we transpose to [num_prompts, seq_len, dim].
        if x.dim() == 3:
            return x.transpose(0, 1), True
        return x, False

    @staticmethod
    def _bnc_to_sam3_language(x: Tensor, transposed: bool) -> Tensor:
        return x.transpose(0, 1) if transposed and x.dim() == 3 else x

    def set_cpca_context(
        self,
        num_group_prompts: Optional[int] = None,
        batch_size: Optional[int] = None,
        canonical_prompt_index: int = 0,
    ) -> None:
        self.cpca_context_num_group_prompts = num_group_prompts
        self.cpca_context_batch_size = batch_size
        self.cpca_context_canonical_prompt_index = canonical_prompt_index

    def _build_current_group_prototypes(
        self,
        prompt_emb: Tensor,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[str], Optional[int], bool, bool]:
        if self.cpca is None or prompt_emb.dim() != 3:
            return None, None, None, None, False, False
        num_group_prompts = self.cpca_context_num_group_prompts
        batch_size = self.cpca_context_batch_size
        layout = self.cpca._infer_text_layout(
            prompt_emb,
            num_group_prompts=num_group_prompts,
            batch_size=batch_size,
        )
        prompt_slots = prompt_emb.shape[1]
        if layout == "seq_first_merged_group" and num_group_prompts is not None:
            prompt_slots = num_group_prompts
        elif layout in {"seq_first_group", "seq_first"}:
            prompt_slots = prompt_emb.shape[1]
        elif layout == "batch_first":
            prompt_slots = 1

        if num_group_prompts is None:
            num_group_prompts = prompt_slots
        if num_group_prompts <= 1 or prompt_slots <= 1:
            return None, None, layout, prompt_slots, False, False

        canonical_idx = min(max(int(self.cpca_context_canonical_prompt_index), 0), num_group_prompts - 1)
        if layout == "seq_first_merged_group":
            l, merged, c = prompt_emb.shape
            if batch_size is None or merged != batch_size * num_group_prompts:
                return None, None, layout, prompt_slots, False, False
            grouped = prompt_emb.view(l, batch_size, num_group_prompts, c)
            canonical = (
                grouped[:, :, canonical_idx : canonical_idx + 1, :]
                .expand(-1, -1, num_group_prompts, -1)
                .reshape(l, merged, c)
                .detach()
            )
            return canonical, prompt_emb.detach(), layout, prompt_slots, True, True

        if layout in {"seq_first_group", "seq_first"} and prompt_emb.shape[1] >= num_group_prompts:
            canonical = prompt_emb[:, canonical_idx : canonical_idx + 1, :].detach()
            return canonical, prompt_emb.detach(), layout, prompt_slots, True, True

        return None, None, layout, prompt_slots, False, False

    def apply_cpca_to_text_embedding(
        self,
        prompt_emb: Tensor,
        canonical_emb: Optional[Tensor] = None,
        group_embs: Optional[Tensor] = None,
    ) -> Tensor:
        if not self.use_cpca:
            return prompt_emb
        self.cpca_hook_used = True
        self.cpca_text_embedding_shape = list(prompt_emb.shape)
        self._ensure_cpca(prompt_emb.shape[-1], prompt_emb.device, prompt_emb.dtype)
        assert self.cpca is not None
        layout = self.cpca._infer_text_layout(
            prompt_emb,
            num_group_prompts=self.cpca_context_num_group_prompts,
            batch_size=self.cpca_context_batch_size,
        )
        prompt_slots = self.cpca_context_num_group_prompts
        if prompt_slots is None and prompt_emb.dim() == 3:
            prompt_slots = prompt_emb.shape[1] if layout != "batch_first" else 1
        if canonical_emb is None or group_embs is None:
            auto_canonical, auto_group, auto_layout, auto_slots, canonical_from_group, group_from_group = (
                self._build_current_group_prototypes(prompt_emb)
            )
            layout = auto_layout or layout
            prompt_slots = auto_slots or prompt_slots
            if canonical_emb is None:
                canonical_emb = auto_canonical
                self.cpca_canonical_from_current_group = canonical_from_group
            if group_embs is None:
                group_embs = auto_group
                self.cpca_group_from_current_group = group_from_group
        self.cpca_layout = layout
        self.cpca_num_prompt_slots = prompt_slots
        self.canonical_embedding_shape = list(canonical_emb.shape) if isinstance(canonical_emb, Tensor) else None
        self.group_embedding_shape = list(group_embs.shape) if isinstance(group_embs, Tensor) else None
        if self.cpca_variant == "v2":
            calibrated = self.cpca(
                prompt_emb,
                canonical_emb=canonical_emb,
                group_embs=group_embs,
                num_group_prompts=self.cpca_context_num_group_prompts,
                batch_size=self.cpca_context_batch_size,
                canonical_prompt_index=self.cpca_context_canonical_prompt_index,
            )
            debug_info = getattr(self.cpca, "last_debug_info", {}) or {}
            self.cpca_v2_active = bool(debug_info.get("cpca_v2_active", False))
            self.cpca_v2_reliability_mean = debug_info.get("reliability_mean")
            self.cpca_v2_reliability_min = debug_info.get("reliability_min")
            self.cpca_v2_reliability_max = debug_info.get("reliability_max")
            self.cpca_v2_calibration_strength_mean = debug_info.get("calibration_strength_mean")
            self.cpca_v2_calibration_delta_norm = debug_info.get("calibration_delta_norm")
        else:
            calibrated = self.cpca(
                prompt_emb,
                canonical_emb=canonical_emb,
                group_embs=group_embs,
                num_group_prompts=self.cpca_context_num_group_prompts,
                batch_size=self.cpca_context_batch_size,
            )
        self.last_cpca_output = self.cpca.last_calibrated_vec
        self.last_cpca_canonical = self.cpca.last_canonical_vec
        output = calibrated
        self.cpca_calibrated_embedding_shape = list(output.shape)
        return output

    def reset_cpca_debug_state(self) -> None:
        self.cpca_hook_used = False
        self.cpca_text_embedding_shape = None
        self.cpca_layout = None
        self.cpca_num_prompt_slots = None
        self.cpca_canonical_from_current_group = False
        self.cpca_group_from_current_group = False
        self.canonical_embedding_shape = None
        self.group_embedding_shape = None
        self.cpca_calibrated_embedding_shape = None
        self.cpca_v2_active = False
        self.cpca_v2_reliability_mean = None
        self.cpca_v2_reliability_min = None
        self.cpca_v2_reliability_max = None
        self.cpca_v2_calibration_strength_mean = None
        self.cpca_v2_calibration_delta_norm = None

    def get_cpca_debug_state(self) -> Dict[str, Any]:
        return {
            "cpca_variant": self.cpca_variant,
            "cpca_hook_used": bool(self.cpca_hook_used),
            "cpca_text_embedding_shape": self.cpca_text_embedding_shape,
            "cpca_layout": self.cpca_layout,
            "cpca_num_prompt_slots": self.cpca_num_prompt_slots,
            "cpca_canonical_from_current_group": bool(self.cpca_canonical_from_current_group),
            "cpca_group_from_current_group": bool(self.cpca_group_from_current_group),
            "canonical_embedding_shape": self.canonical_embedding_shape,
            "group_embedding_shape": self.group_embedding_shape,
            "cpca_calibrated_embedding_shape": self.cpca_calibrated_embedding_shape,
            "cpca_v2_active": bool(self.cpca_v2_active),
            "cpca_v2_reliability_mean": self.cpca_v2_reliability_mean,
            "cpca_v2_reliability_min": self.cpca_v2_reliability_min,
            "cpca_v2_reliability_max": self.cpca_v2_reliability_max,
            "cpca_v2_calibration_strength_mean": self.cpca_v2_calibration_strength_mean,
            "cpca_v2_calibration_delta_norm": self.cpca_v2_calibration_delta_norm,
        }

    def _patch_forward_text_if_available(self) -> None:
        backbone = getattr(self.base_model, "backbone", None)
        if backbone is None or not hasattr(backbone, "forward_text"):
            return
        if getattr(backbone, "_prompt_robust_forward_text_patched", False):
            return
        original_forward_text = backbone.forward_text
        wrapper_self = self

        def patched_forward_text(backbone_self, *args, **kwargs):
            output = original_forward_text(*args, **kwargs)
            if (
                wrapper_self.use_cpca
                and isinstance(output, dict)
                and isinstance(output.get("language_features"), torch.Tensor)
            ):
                output = dict(output)
                output["language_features"] = wrapper_self.apply_cpca_to_text_embedding(
                    output["language_features"]
                )
            return output

        backbone.forward_text = types.MethodType(patched_forward_text, backbone)
        backbone._prompt_robust_forward_text_patched = True

    def _apply_cpca_to_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.use_cpca:
            return kwargs
        key, prompt_emb = self._find_first_key(kwargs, TEXT_EMBED_KEYS)
        if key is None:
            backbone = getattr(self.base_model, "backbone", None)
            text_hook_active = bool(
                backbone is not None
                and getattr(backbone, "_prompt_robust_forward_text_patched", False)
            )
            if not text_hook_active and not self._warned_text_missing:
                print("[PromptRobustWrapper] Warning: text embedding key not found, CPCA skipped before base_model.")
                self._warned_text_missing = True
            return kwargs
        _, canonical_emb = self._find_first_key(kwargs, CANONICAL_KEYS)
        _, group_embs = self._find_first_key(kwargs, GROUP_EMB_KEYS)
        kwargs = dict(kwargs)
        kwargs[key] = self.apply_cpca_to_text_embedding(prompt_emb, canonical_emb, group_embs)
        return kwargs

    def _apply_gpsma_to_tensor(self, tensor: Tensor, group_mask_logits: Optional[Tensor] = None) -> Tensor:
        if not self.use_gpsma:
            return tensor
        if tensor.dim() == 3:
            tensor_4d = tensor.unsqueeze(1)
            squeeze = True
        elif tensor.dim() == 4:
            tensor_4d = tensor
            squeeze = False
        else:
            return tensor
        if self.gpsma is None:
            self.gpsma = GroupPromptStableMaskAdapter(
                in_channels=tensor_4d.shape[1],
                **self.gpsma_cfg,
            ).to(device=tensor_4d.device, dtype=tensor_4d.dtype)
        if tensor_4d.shape[1] != self.gpsma.in_channels and self.gpsma.in_channels == 1:
            b, c, h, w = tensor_4d.shape
            flat = tensor_4d.reshape(b * c, 1, h, w)
            refined = self.gpsma(flat, group_mask_logits=None).reshape(b, c, h, w)
        else:
            refined = self.gpsma(tensor_4d, group_mask_logits=group_mask_logits)
        return refined.squeeze(1) if squeeze else refined

    def _replace_output_mask_logits(self, output: Any, new_logits: Optional[Tensor] = None) -> Any:
        if isinstance(output, Tensor):
            return self._apply_gpsma_to_tensor(output) if new_logits is None else new_logits
        if isinstance(output, dict):
            for key in MASK_KEYS:
                value = output.get(key)
                if isinstance(value, Tensor):
                    output[key] = self._apply_gpsma_to_tensor(value) if new_logits is None else new_logits
                    return output
            return output
        if isinstance(output, tuple):
            output_list = list(output)
            for idx, value in enumerate(output_list):
                replaced = self._replace_output_mask_logits(value, new_logits=None)
                if replaced is not value:
                    output_list[idx] = replaced
                    return tuple(output_list)
            return output
        if isinstance(output, list):
            for idx, value in enumerate(output):
                replaced = self._replace_output_mask_logits(value, new_logits=None)
                if replaced is not value:
                    output[idx] = replaced
                    return output
            return output
        return output

    def _has_mask_logits(self, output: Any) -> bool:
        if isinstance(output, Tensor):
            return output.dim() in (3, 4)
        if isinstance(output, dict):
            return any(isinstance(output.get(key), Tensor) for key in MASK_KEYS)
        if isinstance(output, (list, tuple)):
            return any(self._has_mask_logits(item) for item in output)
        return False

    def forward(self, *args, **kwargs):
        kwargs = self._apply_cpca_to_kwargs(kwargs)
        output = self.base_model(*args, **kwargs)
        if self.use_gpsma and self.auto_apply_gpsma:
            if self._has_mask_logits(output):
                output = self._replace_output_mask_logits(output)
            elif not self._warned_mask_missing:
                print("[PromptRobustWrapper] Warning: mask logits key not found, GPSMA skipped.")
                self._warned_mask_missing = True
        return output

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
