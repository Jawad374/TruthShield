"""Fast dataset over the packed cache (features.pt + faces.npy) from cache_features.py.

Rows are selected by a split CSV (video_path, split) so the same cache serves the
main split, the leave-one-engine-out splits and the compressed-video test sets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TENSOR_KEYS = ["havic_feat", "havic_logit", "face_emb", "face_probs", "voice_emb", "voice_probs", "quality"]


class FeatureCache:
    """Loads the packed cache once; faces stay memory-mapped on disk."""

    def __init__(self, cache_dir: str):
        pk = torch.load(f"{cache_dir}/features.pt", weights_only=False)
        self.t = {k: pk[k].float() for k in TENSOR_KEYS}
        self.label = pk["label"].long()
        self.meta = pd.DataFrame(pk["meta"])
        self.meta["row"] = np.arange(len(self.meta))
        self.faces = np.load(f"{cache_dir}/faces.npy", mmap_mode="r")
        self.cache_dir = cache_dir

    def rows_for(self, split_csv: str, split: str) -> np.ndarray:
        s = pd.read_csv(split_csv)
        s = s[s["split"] == split]
        m = s.merge(self.meta[["video_path", "row"]], on="video_path", how="inner")
        return m["row"].to_numpy()

    def coverage(self, split_csv: str, split: str) -> tuple[int, int]:
        s = pd.read_csv(split_csv)
        return len(self.rows_for(split_csv, split)), int((s["split"] == split).sum())


class Normalizer:
    """Standardises HAVIC features and quality descriptors with TRAIN statistics only."""

    def __init__(self, cache: FeatureCache, train_rows: np.ndarray):
        self.stats = {}
        for k in ("havic_feat", "quality", "face_emb", "voice_emb"):
            x = cache.t[k][train_rows]
            self.stats[k] = (x.mean(0), x.std(0).clamp_min(1e-6))

    def __call__(self, k: str, x: torch.Tensor) -> torch.Tensor:
        if k not in self.stats:
            return x
        mu, sd = self.stats[k]
        return (x - mu.to(x.device)) / sd.to(x.device)

    def state_dict(self):
        return {k: (m.clone(), s.clone()) for k, (m, s) in self.stats.items()}

    @classmethod
    def from_state(cls, state):
        n = cls.__new__(cls)
        n.stats = state
        return n


class CachedClips(Dataset):
    def __init__(self, cache: FeatureCache, rows: np.ndarray, n_faces: int | None = None,
                 train: bool = False):
        self.c, self.rows, self.train = cache, np.asarray(rows), train
        self.n_faces = n_faces or cache.faces.shape[1]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = int(self.rows[i])
        faces = self.c.faces[r]  # (K, 3, 224, 224) uint8
        K = faces.shape[0]
        if self.n_faces < K:
            idx = (np.sort(np.random.choice(K, self.n_faces, replace=False)) if self.train
                   else np.linspace(0, K - 1, self.n_faces).round().astype(int))
            faces = faces[idx]
        faces = torch.from_numpy(np.ascontiguousarray(faces))
        if self.train and np.random.rand() < 0.5:
            faces = faces.flip(-1)  # horizontal flip: keeps spectral statistics, adds variety
        item = {k: self.c.t[k][r] for k in TENSOR_KEYS}
        item.update(faces=faces, label=self.c.label[r].float(), row=r)
        return item
