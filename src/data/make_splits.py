"""Build leakage-free train/val/test splits for HiFi-AVDF.

HiFi-AVDF pairs every real clip with a fake generated from it:
    sora2/1234.mp4  (real)   <->   sora2/1234_generated.mp4  (fake)
Both clips of a pair share identity, scene and audio, so a random per-clip split
leaks. We split by PAIR, stratified by generator engine (70/10/20 by default).

Also writes leave-one-engine-out (LOEO) splits: train+val on 5 engines,
test on the held-out one - the strongest generalisation test available here.

Usage:
    python -m src.data.make_splits --data_root /content/drive/MyDrive/TruthShield/data/HiFi-AVDF \
        --out_dir /content/drive/MyDrive/TruthShield/splits
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

ENGINES = ["kling2.5", "pixverseV5", "seedance1.0", "sora2", "veo3.1", "wan2.5"]


def load_index(data_root: str) -> pd.DataFrame:
    frames = []
    for csv in sorted(glob.glob(os.path.join(data_root, "*.csv"))):
        engine = os.path.splitext(os.path.basename(csv))[0]
        df = pd.read_csv(csv)
        df["engine"] = engine
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no engine CSVs found in {data_root}")
    df = pd.concat(frames, ignore_index=True)
    df["label"] = df["overall_label"].astype(int)
    stem = df["video_path"].map(lambda p: re.sub(r"_generated$", "", os.path.splitext(os.path.basename(p))[0]))
    df["pair_id"] = df["engine"] + "/" + stem
    return df[["video_path", "engine", "pair_id", "label"]]


def split_pairs(df: pd.DataFrame, ratios=(0.7, 0.1, 0.2), seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    assign = {}
    for _, g in df.groupby("engine"):
        pairs = g["pair_id"].unique()
        rng.shuffle(pairs)
        n_tr = int(round(ratios[0] * len(pairs)))
        n_va = int(round(ratios[1] * len(pairs)))
        for i, p in enumerate(pairs):
            assign[p] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
    out = df.copy()
    out["split"] = out["pair_id"].map(assign)
    return out


def loeo_split(df: pd.DataFrame, held_out: str, val_ratio: float = 0.1, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["split"] = np.where(out["engine"] == held_out, "test", "train")
    train_pairs = out.loc[out["split"] == "train", "pair_id"].unique()
    rng.shuffle(train_pairs)
    val_pairs = set(train_pairs[: int(round(val_ratio * len(train_pairs)))])
    out.loc[out["pair_id"].isin(val_pairs), "split"] = "val"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    df = load_index(a.data_root)
    main_split = split_pairs(df, seed=a.seed)
    main_split.to_csv(os.path.join(a.out_dir, "hifi_main.csv"), index=False)
    print(main_split.groupby(["split", "label"]).size().unstack())

    for eng in sorted(df["engine"].unique()):
        loeo_split(df, eng, seed=a.seed).to_csv(os.path.join(a.out_dir, f"hifi_loeo_{eng}.csv"), index=False)
    print(f"wrote splits to {a.out_dir}")


if __name__ == "__main__":
    main()
