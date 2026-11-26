"""Explainability for TruthShield (Phase 3).

Three complementary views, because the frequency branch works on spectra:

1. pixel_saliency()   - SmoothGrad of the final fake-logit w.r.t. the face PIXELS
                        (gradients flow back through the FFT/DWT), i.e. WHERE ON THE FACE
                        the evidence is. This is the "facial coordinates" heatmap.
2. spectral_gradcam() - Grad-CAM on the last conv feature map of the frequency branch.
                        Its input plane is the centred log-FFT (plus DWT bands), so the map
                        shows WHICH FREQUENCIES / spectral regions drove the decision.
3. mean_spectrum_difference() - average log-spectrum of fake minus real faces per
                        generator: the dataset-level artifact signature (e.g. grid peaks).

The HAVIC branch is cached (frozen, pre-computed), so it has no heatmap here; its
contribution is reported through the gate weights instead.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.models.frequency_branch import fft_log_magnitude


def _inputs(item: dict, norm, device):
    g = lambda k: item[k].unsqueeze(0).to(device)  # noqa: E731
    return dict(havic_feat=norm("havic_feat", g("havic_feat")), face_emb=norm("face_emb", g("face_emb")),
                voice_emb=norm("voice_emb", g("voice_emb")), quality=norm("quality", g("quality")))


def _blur(x: torch.Tensor, k: int = 9, sigma: float = 3.0) -> torch.Tensor:
    ax = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    g = (g / g.sum())
    x = F.conv2d(x, g.view(1, 1, 1, k), padding=(0, k // 2))
    return F.conv2d(x, g.view(1, 1, k, 1), padding=(k // 2, 0))


def _norm01(m: torch.Tensor) -> torch.Tensor:
    m = m - m.amin(dim=(-2, -1), keepdim=True)
    return m / (m.amax(dim=(-2, -1), keepdim=True) + 1e-8)


def pixel_saliency(model, item: dict, norm, device: str = "cuda", n_samples: int = 16, noise: float = 0.03):
    """-> (K, 224, 224) heatmaps in [0,1], fake logit, gate weights."""
    model.eval()
    faces = item["faces"].float().div(255).unsqueeze(0).to(device)  # (1,K,3,H,W)
    base = _inputs(item, norm, device)
    grads = torch.zeros_like(faces)
    for _ in range(n_samples):
        x = (faces + noise * torch.randn_like(faces)).clamp(0, 1).requires_grad_(True)
        out = model(faces=x, **base)
        out["logits"].sum().backward()
        grads += x.grad.detach()
    sal = (grads / n_samples).abs().amax(dim=2)[0]  # (K,H,W)
    sal = _norm01(_blur(sal.unsqueeze(1)).squeeze(1))
    with torch.no_grad():
        out = model(faces=faces, **base)
    return sal.cpu(), float(out["logits"]), out["weights"][0].cpu()


def spectral_gradcam(model, item: dict, norm, device: str = "cuda"):
    """-> (K, 224, 224) Grad-CAM maps over the frequency-branch input plane, in [0,1]."""
    model.eval()
    faces = item["faces"].float().div(255).unsqueeze(0).to(device)
    out = model(faces=faces, **_inputs(item, norm, device))
    fmap = model.freq._last_feature_map  # (K, C, h, w), kept in the graph
    fmap.retain_grad()
    out["logits"].sum().backward()
    w = fmap.grad.mean(dim=(-2, -1), keepdim=True)
    cam = F.relu((w * fmap).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=faces.shape[-2:], mode="bilinear", align_corners=False)
    return _norm01(cam.squeeze(1)).detach().cpu()


@torch.no_grad()
def spectrum_of(faces_u8: torch.Tensor) -> torch.Tensor:
    """(K,3,H,W) uint8 -> (H,W) mean centred log-FFT magnitude (grey)."""
    return fft_log_magnitude(faces_u8.float().div(255)).mean(dim=(0, 1))


@torch.no_grad()
def mean_spectrum_difference(cache, rows: np.ndarray, max_per_class: int = 200) -> dict:
    """{engine: (H,W) mean log-spectrum(fake) - mean log-spectrum(real)}."""
    eng = cache.meta["engine"].to_numpy()[rows]
    lab = cache.label[rows].numpy()
    out = {}
    for e in sorted(set(eng)):
        spec = {}
        for y in (0, 1):
            rr = rows[(eng == e) & (lab == y)][:max_per_class]
            if len(rr) == 0:
                break
            acc = None
            for r in rr:
                s = spectrum_of(torch.from_numpy(np.ascontiguousarray(cache.faces[r])))
                acc = s if acc is None else acc + s
            spec[y] = acc / len(rr)
        if len(spec) == 2:
            out[e] = (spec[1] - spec[0]).numpy()
    return out


def overlay(face_chw_u8: torch.Tensor, heat: torch.Tensor, alpha: float = 0.45) -> np.ndarray:
    import matplotlib

    img = face_chw_u8.permute(1, 2, 0).numpy().astype(np.float32) / 255
    col = matplotlib.colormaps["jet"](heat.numpy())[..., :3]
    return np.clip((1 - alpha) * img + alpha * col, 0, 1)


def explanation_figure(model, cache, rows, norm, device, path: str, titles=None, frame: int = 0):
    """One row per clip: face | log-FFT | spectral Grad-CAM | pixel saliency (on face)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    fig, ax = plt.subplots(n, 4, figsize=(12, 3.5 * n), squeeze=False)
    for i, r in enumerate(rows):
        item = {k: (v if not torch.is_tensor(v) else v) for k, v in _item(cache, r).items()}
        sal, logit, w = pixel_saliency(model, item, norm, device)
        cam = spectral_gradcam(model, item, norm, device)
        face = item["faces"][frame]
        spec = fft_log_magnitude(face.float().div(255).unsqueeze(0))[0].mean(0)
        p = 1 / (1 + np.exp(-logit))
        lab = "FAKE" if int(cache.label[r]) else "REAL"
        head = (titles[i] + "\n") if titles else ""
        ax[i, 0].imshow(face.permute(1, 2, 0).numpy())
        ax[i, 0].set_title(f"{head}{cache.meta['engine'].iloc[r]} | true {lab} | p(fake)={p:.2f}", fontsize=8)
        ax[i, 1].imshow(spec.numpy(), cmap="magma"); ax[i, 1].set_title("log |FFT| (centred)", fontsize=8)
        ax[i, 2].imshow(spec.numpy(), cmap="gray"); ax[i, 2].imshow(cam[frame].numpy(), cmap="jet", alpha=0.5)
        ax[i, 2].set_title("spectral Grad-CAM (which frequencies)", fontsize=8)
        ax[i, 3].imshow(overlay(face, sal[frame]))
        ax[i, 3].set_title(f"pixel saliency (where on face)\ngates H/F/E = {w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f}", fontsize=8)
        for a in ax[i]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def spectrum_difference_figure(diffs: dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(diffs)
    fig, ax = plt.subplots(1, n, figsize=(3.2 * n, 3.4), squeeze=False)
    # robust colour scale: a few extreme bins (e.g. a Nyquist corner) would wash out everything else
    lim = float(np.percentile(np.abs(np.stack(list(diffs.values()))), 99)) or 1.0
    im = None
    for a, (e, d) in zip(ax[0], diffs.items()):
        im = a.imshow(d, cmap="coolwarm", vmin=-lim, vmax=lim)
        a.set_title(e, fontsize=9); a.axis("off")
    fig.colorbar(im, ax=ax[0].tolist(), shrink=0.8, label="log-spectrum: fake − real")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _item(cache, r: int) -> dict:
    from src.data.dataset_loader import TENSOR_KEYS

    d = {k: cache.t[k][r] for k in TENSOR_KEYS}
    d["faces"] = torch.from_numpy(np.ascontiguousarray(cache.faces[r]))
    return d
