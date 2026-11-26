"""Video -> face crops + audio, replicating HAVIC's official preprocessing
(evaluation/sliding_window_infer.py) so cached HAVIC features match the baseline:

  * frames sampled every 0.2 s (5 fps), at most MAX_TIME*5 = 50 frames
  * RetinaFace (FaceX-Zoo) box, highest score, scale 1.0, PIL resize to 224x224
  * 16 kHz mono audio, the part aligned with the used frames, split into windows
  * kaldi fbank, 128 mel bins, interpolated to 1024 frames, normalised (-6.9960, 3.1205)

Differences from the official script (documented on purpose):
  * frames stay in memory (no PNG round-trip; PNG is lossless, so pixels are identical)
  * face detection runs on GPU (the official handler hard-codes CPU)
  * a frame with no detected face reuses the nearest detected box instead of dropping
    the whole video (counted in `n_missing_faces` so it can be filtered later)

Heavy dependencies (decord, ffmpeg, torchvision) are imported lazily.
"""
from __future__ import annotations

import math
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

FPS = 5
MAX_FRAMES = 50
WIN = 16
FBANK_MEAN, FBANK_STD = -6.9960, 3.1205


# ----------------------------------------------------------------------------- frames
def sample_frames(video_path: str, fps: int = FPS, max_frames: int = MAX_FRAMES):
    """Returns (frames RGB uint8 [N,H,W,3], num_timepoints, duration_s)."""
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0))
    total, vfps = len(vr), vr.get_avg_fps()
    if total < 1 or not vfps or vfps <= 0:
        raise ValueError("unreadable video (no frames / fps)")
    duration = total / vfps
    times = np.arange(0, duration, 1.0 / fps)
    idx = np.unique(np.clip((times * vfps).astype(int), 0, total - 1))[:max_frames]
    frames = vr.get_batch(idx).asnumpy()
    return frames, len(times), duration


# ----------------------------------------------------------------------------- faces
def fill_missing_boxes(boxes: list):
    """Replace None entries with the nearest detected box. Returns (boxes, n_missing)."""
    found = [i for i, b in enumerate(boxes) if b is not None]
    if not found:
        raise ValueError("no face detected in any frame")
    out = []
    for i, b in enumerate(boxes):
        out.append(b if b is not None else boxes[min(found, key=lambda j: abs(j - i))])
    return out, len(boxes) - len(found)


def crop_box(frame: np.ndarray, box, scale: float = 1.0):
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = map(int, box[:4])
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = (x1 - x0) * scale, (y1 - y0) * scale
    x0, x1 = max(int(cx - w / 2), 0), min(int(cx + w / 2), W)
    y0, y1 = max(int(cy - h / 2), 0), min(int(cy + h / 2), H)
    return frame[y0:y1, x0:x1]


def crops_to_tensor(frames: np.ndarray, boxes: list, size: int = 224) -> torch.Tensor:
    """-> uint8 tensor [N, 3, size, size], resized exactly like HAVIC (PIL + T.Resize)."""
    import torchvision.transforms as T
    from PIL import Image

    resize = T.Resize(size=(size, size))
    out = []
    for f, b in zip(frames, boxes):
        img = resize(Image.fromarray(np.ascontiguousarray(crop_box(f, b))))
        out.append(torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1))
    return torch.stack(out)


class GPUFaceDetector:
    """Wraps HAVIC's FaceX-Zoo RetinaFace but runs it on GPU."""

    def __init__(self, havic_root: str, device: str = "cuda"):
        import os
        import sys

        import yaml

        ev = os.path.join(havic_root, "evaluation")
        for p in (ev, os.path.join(ev, "FaceX_Zoo/face_sdk"), os.path.join(ev, "FaceX_Zoo/face_sdk/models/network_def")):
            if p not in sys.path:
                sys.path.append(p)
        from FaceX_Zoo.face_sdk.core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
        from FaceX_Zoo.face_sdk.core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader

        sdk = os.path.join(ev, "FaceX_Zoo", "face_sdk")
        conf = yaml.safe_load(open(os.path.join(sdk, "config", "model_conf.yaml")))
        loader = FaceDetModelLoader(os.path.join(sdk, "models"), "face_detection",
                                    conf["non-mask"]["face_detection"], meta_file="model_meta.json")
        model, cfg = loader.load_model()
        self.h = FaceDetModelHandler(model, device, cfg)
        self.device = device
        self.h.model = self.h.model.to(device).eval()

    @torch.no_grad()
    def detect(self, frame_rgb: np.ndarray):
        """Highest-scoring box (x0, y0, x1, y1) or None. Same maths as the official
        handler's inference_on_image, minus its hard-coded CPU device."""
        h = self.h
        bgr = frame_rgb[:, :, ::-1]
        img, scale = h._preprocess(bgr)
        H, W = bgr.shape[:2]
        h.device = self.device
        loc, conf, _ = h.model(torch.from_numpy(img).unsqueeze(0).to(self.device))
        dets = h._postprocess(loc, conf, scale.to(self.device), H, W)
        return dets[0][:4] if len(dets) else None


# ----------------------------------------------------------------------------- audio
def load_audio_16k(video_path: str) -> torch.Tensor:
    """Mono 16 kHz float waveform [1, T] in [-1, 1] (same as ffmpeg->wav->torchaudio)."""
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", video_path,
           "-f", "s16le", "-ac", "1", "-ar", "16000", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    wav = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if wav.size == 0:
        raise ValueError("no audio stream")
    return torch.from_numpy(wav).unsqueeze(0)


def fbank_1024(segment: torch.Tensor, sr: int = 16000) -> torch.Tensor:
    import torchaudio

    segment = segment - segment.mean()
    fb = torchaudio.compliance.kaldi.fbank(
        segment, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type="hanning", num_mel_bins=128, dither=0.0, frame_shift=10)
    fb = F.interpolate(fb.unsqueeze(0).transpose(1, 2), size=(1024,), mode="linear",
                       align_corners=False).transpose(1, 2).squeeze(0)
    return (fb - FBANK_MEAN) / FBANK_STD


# ----------------------------------------------------------------------------- HAVIC windows
def window_plan(n_frames: int, num_timepoints: int, n_samples: int, win: int = WIN, stride: int = 2):
    """Frame/audio window boundaries, replicating the official script's arithmetic.
    Returns (n_used_frames, frame_starts, audio_bounds)."""
    if n_frames < win:
        raise ValueError(f"only {n_frames} face frames (< {win})")
    rem = (n_frames - win) % stride
    n_used = n_frames - rem
    n_win = (n_used - win) // stride + 1
    starts = list(range(0, n_used - win + 1, stride))
    total_audio = int((n_used / num_timepoints) * n_samples)
    seg_len = int((total_audio * win) / (win + stride * (n_win - 1)))
    a_stride = int((stride / win) * seg_len)
    bounds = []
    for i in range(n_win):
        s, e = i * a_stride, i * a_stride + seg_len
        if e > total_audio:
            e = total_audio
            s = max(0, e - seg_len)
        bounds.append((int(s), int(e)))
    return n_used, starts, bounds


def havic_batches(faces_u8: torch.Tensor, wav: torch.Tensor, num_timepoints: int, stride: int = 2):
    """faces_u8 [N,3,224,224] uint8, wav [1,T] -> video [W,3,16,224,224] float, audio [W,1024,128]."""
    n_used, starts, bounds = window_plan(faces_u8.shape[0], num_timepoints, wav.shape[1], stride=stride)
    frames = faces_u8[:n_used].float() / 255.0
    video = torch.stack([frames[s:s + WIN].permute(1, 0, 2, 3) for s in starts])
    audio = torch.stack([fbank_1024(wav[:, s:e]) for s, e in bounds])
    return video, audio


# ----------------------------------------------------------------------------- quality
def quality_features(faces_u8: torch.Tensor, boxes: list, file_bytes: int,
                     duration: float, wav: torch.Tensor) -> torch.Tensor:
    """4 descriptors for the gating network, all log-scaled to similar ranges:
    [face size (px), face sharpness (Laplacian var), bitrate (kbit/s), audio loudness (dBFS)]."""
    areas = [max(1.0, (b[2] - b[0]) * (b[3] - b[1])) for b in boxes]
    face_px = math.log(float(np.sqrt(np.mean(areas))))
    g = faces_u8.float().mean(1, keepdim=True)
    lap = F.conv2d(g, g.new_tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]]))
    sharp = math.log(float(lap.var(dim=(1, 2, 3)).mean()) + 1.0)
    kbps = math.log(file_bytes * 8 / 1000 / max(duration, 1e-3) + 1.0)
    rms = float(wav.pow(2).mean().sqrt())
    loud = 20 * math.log10(rms + 1e-6) / 20.0  # dBFS scaled to ~[-3, 0]
    return torch.tensor([face_px, sharp, kbps, loud], dtype=torch.float32)


def pick_k(n: int, k: int):
    """k indices evenly spread over n frames."""
    return np.linspace(0, n - 1, num=min(k, n)).round().astype(int).tolist()
