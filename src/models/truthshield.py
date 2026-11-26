"""Full TruthShield model: cached HAVIC feature + trainable frequency branch +
trainable emotion-consistency head -> quality-aware dynamic fusion.

`mode` selects the ablation:
    havic       HAVIC features only (re-trained head; the fair in-domain baseline)
    havic_freq  HAVIC + frequency branch
    havic_emo   HAVIC + emotion branch
    full        all three (TruthShield)
"""
from __future__ import annotations

import torch.nn as nn

from .dynamic_fusion import DynamicFusion
from .emotion_branch import EmotionConsistencyHead
from .frequency_branch import FrequencyBranch

MODES = {
    "havic": ("havic",),
    "havic_freq": ("havic", "freq"),
    "havic_emo": ("havic", "emotion"),
    "full": ("havic", "freq", "emotion"),
}
MODE_NAMES = {
    "havic": "HAVIC (re-trained head)",
    "havic_freq": "HAVIC + Frequency",
    "havic_emo": "HAVIC + Emotion",
    "full": "TruthShield (full fusion)",
}


class TruthShield(nn.Module):
    def __init__(self, mode: str = "full", havic_dim: int = 3072, freq_dim: int = 128,
                 emb_dim: int = 768, quality_dim: int = 4, d_model: int = 256, dropout: float = 0.3,
                 freq_width: int = 32, use_dwt: bool = True):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}")
        self.mode, self.active = mode, MODES[mode]
        self.freq = FrequencyBranch(freq_dim, use_dwt=use_dwt, width=freq_width) if "freq" in self.active else None
        self.emo = EmotionConsistencyHead(emb_dim, emb_dim, 64) if "emotion" in self.active else None
        self.fusion = DynamicFusion(havic_dim, freq_dim, quality_dim, d_model, dropout, active=self.active)
        self.freq_dim = freq_dim

    def forward(self, havic_feat, faces=None, face_emb=None, voice_emb=None, quality=None):
        B = havic_feat.shape[0]
        freq = self.freq(faces) if self.freq is not None else havic_feat.new_zeros(B, self.freq_dim)
        if self.emo is not None:
            mismatch, _, _ = self.emo(face_emb, voice_emb)
        else:
            mismatch = havic_feat.new_zeros(B, 1)
        logits, weights = self.fusion(havic_feat, freq, mismatch, quality)
        return {"logits": logits, "weights": weights, "mismatch": mismatch.squeeze(-1)}
