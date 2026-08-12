"""Prompt robustness adapters for MedSAM3/SAM3.

The modules in this file are intentionally small residual adapters. Their last
projection/convolution is zero-initialized, so enabling them starts as an
approximately identity transformation of the frozen backbone signals.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class _ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.init_identity()

    def init_identity(self) -> None:
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ConceptPrototypeCalibrationAdapter(nn.Module):
    """Calibrate prompt concept embeddings against canonical/group prototypes.

    Supports prompt tensors with shape ``[B, C]`` or ``[B, N, C]``. For token
    embeddings, the adapter computes a mean-pooled concept vector and broadcasts
    the residual update back to all tokens so the output shape is unchanged.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        residual_scale: float = 0.1,
        text_layout: str = "auto",
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if text_layout not in {"auto", "seq_first", "batch_first"}:
            raise ValueError(f"text_layout must be auto, seq_first, or batch_first; got {text_layout}")
        if dim % num_heads != 0:
            num_heads = 1
        hidden_dim = hidden_dim or max(dim, dim * 2)
        self.dim = dim
        self.residual_scale = residual_scale
        self.text_layout = text_layout
        self.prototype_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate_mlp = _ResidualMLP(dim * 4, dim, hidden_dim, dropout)
        self.delta_mlp = _ResidualMLP(dim * 2, dim, hidden_dim, dropout)
        self.last_prompt_vec: Optional[Tensor] = None
        self.last_calibrated_vec: Optional[Tensor] = None
        self.last_canonical_vec: Optional[Tensor] = None
        self.last_layout: Optional[str] = None
        self.last_num_prompt_slots: Optional[int] = None

    def init_identity(self) -> None:
        self.gate_mlp.init_identity()
        self.delta_mlp.init_identity()

    def _infer_text_layout(
        self,
        x: Tensor,
        num_group_prompts: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> str:
        if x.dim() == 2:
            return "flat"
        if x.dim() != 3:
            raise ValueError(f"text embedding must have shape [B,C], [B,L,C], or [L,B,C], got {tuple(x.shape)}")
        if self.text_layout == "batch_first":
            return "batch_first"
        if self.text_layout == "seq_first":
            return "seq_first"
        if num_group_prompts is not None and num_group_prompts > 0:
            if batch_size is not None and batch_size > 1 and x.shape[1] == batch_size * num_group_prompts:
                return "seq_first_merged_group"
            if x.shape[1] == num_group_prompts:
                return "seq_first_group"
        return "seq_first" if x.shape[0] > x.shape[1] else "batch_first"

    def _pool_concept_vector(
        self,
        x: Tensor,
        layout: str,
        num_group_prompts: Optional[int] = None,
        batch_size: Optional[int] = None,
        keep_group: bool = False,
    ) -> Tensor:
        if x.dim() == 2:
            return x
        if x.dim() != 3:
            raise ValueError(f"text embedding must be 2D or 3D, got {tuple(x.shape)}")
        if layout == "batch_first":
            return x.mean(dim=1)
        if layout in {"seq_first", "seq_first_group"}:
            return x.mean(dim=0)
        if layout == "seq_first_merged_group":
            if num_group_prompts is None or batch_size is None:
                raise ValueError("num_group_prompts and batch_size are required for seq_first_merged_group")
            l, merged, c = x.shape
            if merged != batch_size * num_group_prompts:
                raise ValueError(
                    f"merged text slots {merged} != batch_size*num_group_prompts "
                    f"{batch_size}*{num_group_prompts}"
                )
            pooled = x.view(l, batch_size, num_group_prompts, c).mean(dim=0)
            return pooled if keep_group else pooled.reshape(batch_size * num_group_prompts, c)
        raise ValueError(f"Unsupported text layout: {layout}")

    def _restore_delta_to_embedding(
        self,
        delta_vec: Tensor,
        x: Tensor,
        layout: str,
        num_group_prompts: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Tensor:
        if x.dim() == 2:
            return delta_vec
        if layout == "batch_first":
            return delta_vec.unsqueeze(1).expand_as(x)
        if layout in {"seq_first", "seq_first_group", "seq_first_merged_group"}:
            return delta_vec.unsqueeze(0).expand_as(x)
        raise ValueError(f"Unsupported text layout: {layout}")

    def _to_vec_like(
        self,
        x: Optional[Tensor],
        ref_vec: Tensor,
        layout: str,
        num_group_prompts: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Tensor:
        if x is None:
            return ref_vec
        if x.dim() == 3:
            pooled = self._pool_concept_vector(
                x,
                layout,
                num_group_prompts=num_group_prompts,
                batch_size=batch_size,
            )
            if pooled.shape[0] == ref_vec.shape[0]:
                return pooled
            if pooled.shape[0] == 1:
                return pooled.expand(ref_vec.shape[0], -1)
            if num_group_prompts and pooled.shape[0] == ref_vec.shape[0] // num_group_prompts:
                return pooled[:, None, :].expand(-1, num_group_prompts, -1).reshape(ref_vec.shape[0], -1)
            return pooled
        if x.dim() == 2:
            if x.shape[0] == ref_vec.shape[0]:
                return x
            if x.shape[0] == 1:
                return x.expand(ref_vec.shape[0], -1)
        raise ValueError(
            f"canonical_emb shape {tuple(x.shape)} is incompatible with prompt vector {tuple(ref_vec.shape)}"
        )

    def _prepare_group(
        self,
        group_embs: Optional[Tensor],
        ref_vec: Tensor,
        layout: str,
        num_group_prompts: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Tensor:
        if group_embs is None:
            return ref_vec.unsqueeze(1)
        if group_embs.dim() == 2:
            return group_embs.unsqueeze(0).expand(ref_vec.shape[0], -1, -1)
        if group_embs.dim() == 3:
            if layout == "seq_first_merged_group":
                group_vec = self._pool_concept_vector(
                    group_embs,
                    layout,
                    num_group_prompts=num_group_prompts,
                    batch_size=batch_size,
                    keep_group=True,
                )
                bank = group_vec[:, None, :, :].expand(-1, group_vec.shape[1], -1, -1)
                return bank.reshape(ref_vec.shape[0], group_vec.shape[1], ref_vec.shape[-1])
            group_vec = self._pool_concept_vector(
                group_embs,
                layout,
                num_group_prompts=num_group_prompts,
                batch_size=batch_size,
            )
            if group_vec.dim() == 2:
                return group_vec.unsqueeze(0).expand(ref_vec.shape[0], -1, -1)
            if group_vec.shape[0] == ref_vec.shape[0]:
                return group_vec
        raise ValueError(
            f"group_embs shape {tuple(group_embs.shape)} is incompatible with prompt vector {tuple(ref_vec.shape)}"
        )

    def forward(
        self,
        prompt_emb: Tensor,
        canonical_emb: Optional[Tensor] = None,
        group_embs: Optional[Tensor] = None,
        num_group_prompts: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Tensor:
        layout = self._infer_text_layout(
            prompt_emb,
            num_group_prompts=num_group_prompts,
            batch_size=batch_size,
        )
        prompt_vec = self._pool_concept_vector(
            prompt_emb,
            layout,
            num_group_prompts=num_group_prompts,
            batch_size=batch_size,
        )
        canonical_vec = self._to_vec_like(
            canonical_emb,
            prompt_vec,
            layout,
            num_group_prompts=num_group_prompts,
            batch_size=batch_size,
        ).to(
            device=prompt_vec.device, dtype=prompt_vec.dtype
        )
        group_bank = self._prepare_group(
            group_embs,
            prompt_vec,
            layout,
            num_group_prompts=num_group_prompts,
            batch_size=batch_size,
        ).to(
            device=prompt_vec.device, dtype=prompt_vec.dtype
        )

        d = prompt_vec - canonical_vec
        attn_out, _ = self.prototype_attn(
            query=prompt_vec.unsqueeze(1),
            key=group_bank,
            value=group_bank,
            need_weights=False,
        )
        a = attn_out.squeeze(1)
        gate_input = torch.cat([prompt_vec, canonical_vec, d, a], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_input))
        delta = self.delta_mlp(torch.cat([d, a], dim=-1))
        calibrated_vec = prompt_vec + self.residual_scale * gate * delta

        self.last_prompt_vec = prompt_vec
        self.last_calibrated_vec = calibrated_vec
        self.last_canonical_vec = canonical_vec
        self.last_layout = layout
        self.last_num_prompt_slots = num_group_prompts

        delta = calibrated_vec - prompt_vec
        restored_delta = self._restore_delta_to_embedding(
            delta,
            prompt_emb,
            layout,
            num_group_prompts=num_group_prompts,
            batch_size=batch_size,
        )
        return prompt_emb + restored_delta


class ReliabilityAwareConceptCalibrationAdapter(nn.Module):
    """CPCA-v2: reliability-aware selective concept calibration.

    This adapter keeps prompt-specific information by using a weak residual
    update gated by prompt reliability. Canonical prompts are kept fixed, while
    non-canonical prompts can read from both the canonical concept and the group
    consensus concept.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        residual_scale: float = 0.05,
        text_layout: str = "auto",
        reliability_bias: float = 2.0,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if text_layout not in {"auto", "seq_first", "batch_first"}:
            raise ValueError(f"text_layout must be auto, seq_first, or batch_first; got {text_layout}")
        if dim % num_heads != 0:
            num_heads = 1
        hidden_dim = hidden_dim or max(dim, dim * 2)
        self.dim = dim
        self.residual_scale = residual_scale
        self.text_layout = text_layout
        self.anchor_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.reliability_mlp = _ResidualMLP(dim * 7, 1, hidden_dim, dropout)
        reliability_last = self.reliability_mlp.net[-1]
        if isinstance(reliability_last, nn.Linear):
            nn.init.zeros_(reliability_last.weight)
            nn.init.constant_(reliability_last.bias, reliability_bias)
        self.delta_mlp = _ResidualMLP(dim * 5, dim, hidden_dim, dropout)
        self.last_prompt_vec: Optional[Tensor] = None
        self.last_calibrated_vec: Optional[Tensor] = None
        self.last_canonical_vec: Optional[Tensor] = None
        self.last_group_consensus_vec: Optional[Tensor] = None
        self.last_layout: Optional[str] = None
        self.last_num_prompt_slots: Optional[int] = None
        self.last_debug_info = {}

    # Reuse the layout-aware helpers from CPCA-v1 without sharing parameters.
    _infer_text_layout = ConceptPrototypeCalibrationAdapter._infer_text_layout
    _pool_concept_vector = ConceptPrototypeCalibrationAdapter._pool_concept_vector
    _restore_delta_to_embedding = ConceptPrototypeCalibrationAdapter._restore_delta_to_embedding
    _to_vec_like = ConceptPrototypeCalibrationAdapter._to_vec_like

    def _canonical_and_consensus(
        self,
        prompt_vec: Tensor,
        canonical_emb: Optional[Tensor],
        layout: str,
        num_group_prompts: Optional[int],
        batch_size: Optional[int],
        canonical_prompt_index: int,
    ) -> Tuple[Tensor, Tensor, int]:
        num_prompts = int(num_group_prompts or 1)
        if num_prompts > 1 and prompt_vec.shape[0] % num_prompts == 0:
            grouped = prompt_vec.view(-1, num_prompts, prompt_vec.shape[-1])
            canonical_idx = min(max(int(canonical_prompt_index), 0), num_prompts - 1)
            canonical_vec = grouped[:, canonical_idx : canonical_idx + 1, :].expand_as(grouped)
            consensus_vec = grouped.mean(dim=1, keepdim=True).expand_as(grouped)
            return (
                canonical_vec.reshape_as(prompt_vec),
                consensus_vec.reshape_as(prompt_vec),
                num_prompts,
            )

        canonical_vec = self._to_vec_like(
            canonical_emb,
            prompt_vec,
            layout,
            num_group_prompts=num_group_prompts,
            batch_size=batch_size,
        ).to(device=prompt_vec.device, dtype=prompt_vec.dtype)
        consensus_vec = prompt_vec.mean(dim=0, keepdim=True).expand_as(prompt_vec)
        return canonical_vec, consensus_vec, max(1, num_prompts)

    def forward(
        self,
        prompt_emb: Tensor,
        canonical_emb: Optional[Tensor] = None,
        group_embs: Optional[Tensor] = None,
        num_group_prompts: Optional[int] = None,
        batch_size: Optional[int] = None,
        canonical_prompt_index: int = 0,
        return_debug_info: bool = False,
    ):
        layout = self._infer_text_layout(
            prompt_emb,
            num_group_prompts=num_group_prompts,
            batch_size=batch_size,
        )
        prompt_vec = self._pool_concept_vector(
            prompt_emb,
            layout,
            num_group_prompts=num_group_prompts,
            batch_size=batch_size,
        )
        canonical_vec, group_consensus_vec, num_prompts = self._canonical_and_consensus(
            prompt_vec,
            canonical_emb=canonical_emb,
            layout=layout,
            num_group_prompts=num_group_prompts,
            batch_size=batch_size,
            canonical_prompt_index=canonical_prompt_index,
        )

        d_c = prompt_vec - canonical_vec
        d_g = prompt_vec - group_consensus_vec
        anchors = torch.stack([canonical_vec, group_consensus_vec], dim=1)
        anchor_context, _ = self.anchor_attn(
            query=prompt_vec.unsqueeze(1),
            key=anchors,
            value=anchors,
            need_weights=False,
        )
        anchor_context = anchor_context.squeeze(1)

        reliability_input = torch.cat(
            [prompt_vec, canonical_vec, group_consensus_vec, d_c, d_g, torch.abs(d_c), torch.abs(d_g)],
            dim=-1,
        )
        reliability = torch.sigmoid(self.reliability_mlp(reliability_input))
        calibration_strength = 1.0 - reliability

        if num_prompts > 1 and calibration_strength.shape[0] % num_prompts == 0:
            strength_grouped = calibration_strength.view(-1, num_prompts, 1)
            canonical_idx = min(max(int(canonical_prompt_index), 0), num_prompts - 1)
            strength_grouped[:, canonical_idx, :] = 0.0
            calibration_strength = strength_grouped.reshape_as(calibration_strength)

        delta_input = torch.cat([d_c, d_g, torch.abs(d_c), torch.abs(d_g), anchor_context], dim=-1)
        delta = self.delta_mlp(delta_input)
        calibrated_vec = prompt_vec + self.residual_scale * calibration_strength * delta
        vector_delta = calibrated_vec - prompt_vec
        restored_delta = self._restore_delta_to_embedding(
            vector_delta,
            prompt_emb,
            layout,
            num_group_prompts=num_group_prompts,
            batch_size=batch_size,
        )
        output = prompt_emb + restored_delta

        self.last_prompt_vec = prompt_vec
        self.last_calibrated_vec = calibrated_vec
        self.last_canonical_vec = canonical_vec
        self.last_group_consensus_vec = group_consensus_vec
        self.last_layout = layout
        self.last_num_prompt_slots = num_prompts
        delta_norm = vector_delta.float().norm(dim=-1).mean()
        self.last_debug_info = {
            "reliability_scores": reliability.detach(),
            "canonical_vec_shape": list(canonical_vec.shape),
            "group_consensus_shape": list(group_consensus_vec.shape),
            "calibration_gate_shape": list(calibration_strength.shape),
            "calibration_delta_norm": float(delta_norm.detach().cpu()),
            "cpca_v2_active": True,
            "reliability_mean": float(reliability.detach().float().mean().cpu()),
            "reliability_min": float(reliability.detach().float().min().cpu()),
            "reliability_max": float(reliability.detach().float().max().cpu()),
            "calibration_strength_mean": float(calibration_strength.detach().float().mean().cpu()),
        }
        if return_debug_info:
            return output, self.last_debug_info
        return output


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, dropout: float):
        super().__init__()
        groups = min(8, hidden_channels)
        while hidden_channels % groups != 0 and groups > 1:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )
        self.init_identity()

    def init_identity(self) -> None:
        last = self.net[-1]
        if isinstance(last, nn.Conv2d):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GroupPromptStableMaskAdapter(nn.Module):
    """Residual mask-logit adapter guided by group consensus logits."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 16,
        dropout: float = 0.1,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.residual_scale = residual_scale
        self.in_channels = in_channels
        self._build_blocks(in_channels)

    def _build_blocks(self, in_channels: int) -> None:
        self.in_channels = in_channels
        block_in = in_channels * 3
        self.spatial_gate = _ConvBlock(block_in, in_channels, self.hidden_channels, self.dropout)
        self.residual = _ConvBlock(block_in, in_channels, self.hidden_channels, self.dropout)

    def reset_channels(self, in_channels: int, device=None, dtype=None) -> None:
        self._build_blocks(in_channels)
        if device is not None:
            self.to(device=device, dtype=dtype)

    def init_identity(self) -> None:
        self.spatial_gate.init_identity()
        self.residual.init_identity()

    @staticmethod
    def _consensus(mask_logits: Tensor, group_mask_logits: Optional[Tensor]) -> Tensor:
        if group_mask_logits is None:
            return mask_logits.detach()
        if group_mask_logits.dim() != 5:
            raise ValueError(
                "group_mask_logits must have shape [B,K,C,H,W] or [K,B,C,H,W], "
                f"got {tuple(group_mask_logits.shape)}"
            )
        if group_mask_logits.shape[0] == mask_logits.shape[0]:
            return group_mask_logits.mean(dim=1)
        return group_mask_logits.mean(dim=0)

    def forward(self, mask_logits: Tensor, group_mask_logits: Optional[Tensor] = None) -> Tensor:
        if mask_logits.dim() != 4:
            raise ValueError(f"mask_logits must have shape [B,C,H,W], got {tuple(mask_logits.shape)}")
        if mask_logits.shape[1] != self.in_channels:
            self.reset_channels(mask_logits.shape[1], device=mask_logits.device, dtype=mask_logits.dtype)
        consensus = self._consensus(mask_logits, group_mask_logits).to(
            device=mask_logits.device, dtype=mask_logits.dtype
        )
        if consensus.shape[-2:] != mask_logits.shape[-2:]:
            consensus = F.interpolate(consensus, size=mask_logits.shape[-2:], mode="bilinear", align_corners=False)
        if consensus.shape[1] != mask_logits.shape[1]:
            if consensus.shape[1] == 1:
                consensus = consensus.expand(-1, mask_logits.shape[1], -1, -1)
            else:
                consensus = consensus[:, : mask_logits.shape[1]]
        deviation = torch.abs(mask_logits - consensus)
        x = torch.cat([mask_logits, consensus, deviation], dim=1)
        s = torch.sigmoid(self.spatial_gate(x))
        r = self.residual(x)
        return mask_logits + self.residual_scale * s * r
