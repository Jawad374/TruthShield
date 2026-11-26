"""Quality-aware dynamic fusion of the three TruthShield branches.

Inputs per clip:
  havic    (B, havic_dim)  cached HAVIC final feature (4*768 = 3072 from HAVIC_FT)
  freq     (B, 128)        FrequencyBranch embedding
  emotion  (B, 1)          emotion mismatch scalar in [0, 2]
  quality  (B, Q)          clip quality descriptors (blur, face size, bitrate, ...)

Each branch is projected to a common d_model space. A gating MLP sees the
projected features + quality vector and emits softmax routing weights over the
active branches. The weighted sum goes to a binary classification head.

Ablations: pass `active=("havic", "freq")` etc. Inactive branches get zero weight
(masked before the softmax), so the same class serves all 4 ablation modes.
"""
from __future__ import annotations

import torch
import torch.nn as nn

BRANCHES = ("havic", "freq", "emotion")


class DynamicFusion(nn.Module):
    def __init__(self, havic_dim: int = 3072, freq_dim: int = 128, quality_dim: int = 4,
                 d_model: int = 256, dropout: float = 0.3,
                 active: tuple[str, ...] = BRANCHES):
        super().__init__()
        unknown = set(active) - set(BRANCHES)
        if unknown or not active:
            raise ValueError(f"active must be a non-empty subset of {BRANCHES}, got {active}")
        self.active = tuple(b for b in BRANCHES if b in active)
        self.quality_dim = quality_dim

        def proj(d_in):
            return nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_model), nn.GELU(), nn.Dropout(dropout))

        self.proj = nn.ModuleDict({
            "havic": proj(havic_dim),
            "freq": proj(freq_dim),
            # a scalar needs a richer lift than one Linear: use a small MLP
            "emotion": nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, d_model), nn.GELU()),
        })
        self.gate = nn.Sequential(
            nn.Linear(len(BRANCHES) * d_model + quality_dim, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, len(BRANCHES)),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1),
        )
        mask = torch.tensor([b in self.active for b in BRANCHES])
        self.register_buffer("branch_mask", mask, persistent=False)

    def forward(self, havic: torch.Tensor, freq: torch.Tensor, emotion: torch.Tensor,
                quality: torch.Tensor | None = None):
        B = havic.shape[0]
        if emotion.dim() == 1:
            emotion = emotion.unsqueeze(-1)
        if quality is None:
            quality = havic.new_zeros(B, self.quality_dim)
        inputs = {"havic": havic, "freq": freq, "emotion": emotion}

        feats = []
        for b in BRANCHES:
            f = self.proj[b](inputs[b])
            # zero inactive branches so they cannot leak into the gate either
            feats.append(f if b in self.active else torch.zeros_like(f))
        stacked = torch.stack(feats, dim=1)  # (B, 3, d)

        gate_logits = self.gate(torch.cat([stacked.flatten(1), quality], dim=-1))
        mask: torch.Tensor = self.branch_mask  # type: ignore[assignment]
        gate_logits = gate_logits.masked_fill(~mask, float("-inf"))
        weights = gate_logits.softmax(dim=-1)  # (B, 3); zeros on inactive branches

        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)
        logits = self.classifier(fused).squeeze(-1)  # (B,)
        return logits, weights
