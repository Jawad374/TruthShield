"""Clue 1: cross-modal emotion consistency.

Two parts:
  * EmotionExtractor  - frozen pretrained models, run ONCE in cache_features.py (GPU).
      speech: superb/wav2vec2-base-superb-er   (4 classes: neu, hap, ang, sad)
      face:   trpakov/vit-face-expression      (7 classes: angry, disgust, fear,
                                                happy, neutral, sad, surprise)
  * EmotionConsistencyHead - small trainable module on the CACHED embeddings.
      Projects both into a shared 64-d space and returns the cosine divergence
      1 - cos(z_face, z_voice), which lies in [0, 2].

The heavy models are imported lazily so the head can be used/tested without
`transformers` installed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SPEECH_MODEL = "superb/wav2vec2-base-superb-er"
FACE_MODEL = "trpakov/vit-face-expression"

# Map the 7 FER classes onto the 4 speech classes (neu, hap, ang, sad) for a
# training-free mismatch score (useful as a sanity check / extra feature).
FACE7_TO_SPEECH4 = {
    "neutral": "neu", "happy": "hap", "surprise": "hap",
    "angry": "ang", "disgust": "ang", "fear": "sad", "sad": "sad",
}
SPEECH4 = ["neu", "hap", "ang", "sad"]


class EmotionConsistencyHead(nn.Module):
    def __init__(self, face_dim: int = 768, voice_dim: int = 768, latent_dim: int = 64):
        super().__init__()

        def proj(d):
            return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(), nn.Linear(256, latent_dim))

        self.face_proj = proj(face_dim)
        self.voice_proj = proj(voice_dim)

    def forward(self, face_emb: torch.Tensor, voice_emb: torch.Tensor):
        """face_emb (B, face_dim), voice_emb (B, voice_dim) ->
        mismatch (B, 1) in [0, 2], plus the two 64-d latents."""
        zf = F.normalize(self.face_proj(face_emb), dim=-1, eps=1e-8)
        zv = F.normalize(self.voice_proj(voice_emb), dim=-1, eps=1e-8)
        mismatch = (1.0 - (zf * zv).sum(-1, keepdim=True)).clamp(0.0, 2.0)
        return mismatch, zf, zv


def zero_shot_mismatch(face_probs7: torch.Tensor, speech_probs4: torch.Tensor, face_labels: list[str]) -> torch.Tensor:
    """Training-free cosine divergence between FER and SER distributions in the
    shared 4-class space. face_labels gives the FER model's id2label order."""
    M = torch.zeros(len(face_labels), 4, device=face_probs7.device)
    for i, lab in enumerate(face_labels):
        M[i, SPEECH4.index(FACE7_TO_SPEECH4[lab.lower()])] = 1.0
    f4 = face_probs7 @ M
    return 1.0 - F.cosine_similarity(f4, speech_probs4, dim=-1, eps=1e-8)


class EmotionExtractor:
    """Frozen feature extractor used only during caching."""

    def __init__(self, device: str = "cuda"):
        from transformers import (AutoFeatureExtractor, AutoImageProcessor,
                                  AutoModelForAudioClassification, AutoModelForImageClassification)

        self.device = device
        self.sp_fe = AutoFeatureExtractor.from_pretrained(SPEECH_MODEL)
        self.sp = AutoModelForAudioClassification.from_pretrained(SPEECH_MODEL).to(device).eval()
        self.fc_fe = AutoImageProcessor.from_pretrained(FACE_MODEL)
        self.fc = AutoModelForImageClassification.from_pretrained(FACE_MODEL).to(device).eval()
        self.face_labels = [self.fc.config.id2label[i] for i in range(self.fc.config.num_labels)]
        self.speech_labels = [self.sp.config.id2label[i] for i in range(self.sp.config.num_labels)]

    @torch.inference_mode()
    def voice(self, wav_16k: torch.Tensor) -> dict:
        """wav_16k: 1-D float tensor at 16 kHz -> pooled embedding + class probs."""
        inp = self.sp_fe(wav_16k.cpu().numpy(), sampling_rate=16000, return_tensors="pt").to(self.device)
        out = self.sp(**inp, output_hidden_states=True)
        emb = out.hidden_states[-1].mean(dim=1).squeeze(0)
        probs = out.logits.softmax(-1).squeeze(0)
        # reorder to SPEECH4 order
        order = [self.speech_labels.index(l) for l in SPEECH4]
        return {"voice_emb": emb.float().cpu(), "voice_probs": probs[order].float().cpu()}

    @torch.inference_mode()
    def face(self, faces_uint8: torch.Tensor) -> dict:
        """faces_uint8: (K, 3, H, W) uint8 -> frame-averaged embedding + class probs."""
        imgs = [f.permute(1, 2, 0).cpu().numpy() for f in faces_uint8]
        inp = self.fc_fe(images=imgs, return_tensors="pt").to(self.device)
        out = self.fc(**inp, output_hidden_states=True)
        emb = out.hidden_states[-1][:, 0].mean(dim=0)  # CLS token, averaged over frames
        probs = out.logits.softmax(-1).mean(dim=0)
        return {"face_emb": emb.float().cpu(), "face_probs": probs.float().cpu()}
