"""Clue 2: frequency-domain artifact branch.

Takes face crops (B, 3, 224, 224) in [0, 1], computes a log-magnitude 2D FFT
spectrum (plus optional Haar DWT high-frequency sub-bands), and encodes it with a
lightweight 4-stage residual CNN into a 128-dim embedding.

For clips, pass (B, K, 3, H, W): the K frame embeddings are mean-pooled.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


def fft_log_magnitude(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> centred log-magnitude spectrum, per-image standardised."""
    x = x - x.mean(dim=(-2, -1), keepdim=True)  # drop DC so it does not dominate
    spec = torch.fft.fftshift(torch.fft.fft2(x.float(), norm="ortho"), dim=(-2, -1))
    mag = torch.log(spec.abs() + EPS)
    mu = mag.mean(dim=(-2, -1), keepdim=True)
    sd = mag.std(dim=(-2, -1), keepdim=True)
    return (mag - mu) / (sd + EPS)


def haar_dwt_highpass(x: torch.Tensor) -> torch.Tensor:
    """Single-level Haar DWT; returns LH, HL, HH sub-bands upsampled to input size.

    (B, C, H, W) -> (B, 3*C, H, W). Implemented with fixed conv kernels so it runs
    on GPU and stays differentiable (needed for Grad-CAM).
    """
    _, C, H, W = x.shape
    k = 0.5 * x.new_tensor([
        [[1, 1], [-1, -1]],   # LH (horizontal edges)
        [[1, -1], [1, -1]],   # HL (vertical edges)
        [[1, -1], [-1, 1]],   # HH (diagonal)
    ])
    weight = k.unsqueeze(1).repeat(C, 1, 1, 1)  # (3C, 1, 2, 2)
    out = F.conv2d(x, weight, stride=2, groups=C)
    out = F.interpolate(out, size=(H, W), mode="nearest")
    return torch.log(out.abs() + EPS)


class ResidualBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.skip = (
            nn.Identity()
            if stride == 1 and c_in == c_out
            else nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))
        )

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.skip(x), inplace=True)


class FrequencyBranch(nn.Module):
    def __init__(self, embed_dim: int = 128, use_dwt: bool = True, width: int = 32, img_size: int = 224):
        super().__init__()
        self.use_dwt = use_dwt
        self.img_size = img_size
        in_ch = 3 + (9 if use_dwt else 0)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, 2, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True)
        )
        chans = [width, width * 2, width * 4, width * 8]
        layers, c = [], width
        for i, co in enumerate(chans):
            layers.append(ResidualBlock(c, co, stride=1 if i == 0 else 2))
            c = co
        self.layers = nn.Sequential(*layers)  # 4 residual stages
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(c, embed_dim))
        self.embed_dim = embed_dim

    def spectral_input(self, x: torch.Tensor) -> torch.Tensor:
        feats = [fft_log_magnitude(x)]
        if self.use_dwt:
            feats.append(haar_dwt_highpass(x))
        return torch.cat(feats, dim=1)

    def forward_frames(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) -> (N, embed_dim)"""
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"expected (N, 3, H, W), got {tuple(x.shape)}")
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        feat_map = self.layers(self.stem(self.spectral_input(x)))
        self._last_feature_map = feat_map  # hook point for Grad-CAM
        return self.head(feat_map.mean(dim=(-2, -1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) or (B, K, 3, H, W) -> (B, embed_dim)"""
        if x.dim() == 5:
            B, K = x.shape[:2]
            return self.forward_frames(x.flatten(0, 1)).view(B, K, -1).mean(dim=1)
        return self.forward_frames(x)
