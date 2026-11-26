"""Train one TruthShield configuration on the cached features.

    python train_fusion.py --cache_dir /content/cache --split_csv $ROOT/splits/hifi_main.csv \
        --mode full --seed 0 --out_dir $ROOT/runs/main

Saves <out_dir>/<mode>_s<seed>.pt with the best-validation-AUC weights, the
train-set normaliser and the validation-tuned decision threshold.
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.dataset_loader import CachedClips, FeatureCache, Normalizer
from src.models.truthshield import MODES, TruthShield
from src.utils.metrics import best_threshold, compute_metrics


def load_cfg(path: str | None) -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(path or os.path.join(here, "configs", "train_fusion.yaml")) as fh:
        return yaml.safe_load(fh)


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_model(mode: str, cfg: dict) -> TruthShield:
    return TruthShield(mode, dropout=cfg["dropout"], d_model=cfg["d_model"],
                       freq_width=cfg["freq_width"], use_dwt=cfg["use_dwt"])


def to_inputs(batch: dict, norm: Normalizer, device: str, need_faces: bool):
    g = lambda k: batch[k].to(device, non_blocking=True)  # noqa: E731
    return dict(
        havic_feat=norm("havic_feat", g("havic_feat")),
        faces=g("faces") if need_faces else None,
        face_emb=norm("face_emb", g("face_emb")),
        voice_emb=norm("voice_emb", g("voice_emb")),
        quality=norm("quality", g("quality")),
    )


@torch.no_grad()
def predict(model: TruthShield, loader: DataLoader, norm: Normalizer, device: str, amp: bool = True) -> dict:
    model.eval()
    out = {"prob": [], "label": [], "weights": [], "mismatch": [], "row": []}
    for b in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.startswith("cuda")):
            o = model(**to_inputs(b, norm, device, model.freq is not None))
        out["prob"].append(torch.sigmoid(o["logits"].float()).cpu())
        out["weights"].append(o["weights"].float().cpu())
        out["mismatch"].append(o["mismatch"].float().cpu())
        out["label"].append(b["label"])
        out["row"].append(b["row"])
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def make_loader(cache, rows, cfg, train: bool, device: str):
    ds = CachedClips(cache, rows, cfg["n_faces_train"] if train else cfg["n_faces_eval"], train=train)
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=train, drop_last=train and len(ds) > cfg["batch_size"],
                      num_workers=cfg["num_workers"], pin_memory=device.startswith("cuda"),
                      persistent_workers=cfg["num_workers"] > 0)


def train_one(cache: FeatureCache, split_csv: str, mode: str, seed: int, cfg: dict, out_dir: str,
              device: str = "cuda", verbose: bool = True) -> dict:
    set_seed(seed)
    tr, va = cache.rows_for(split_csv, "train"), cache.rows_for(split_csv, "val")
    if len(tr) == 0 or len(va) == 0:
        raise ValueError(f"empty train/val rows for {split_csv}")
    norm = Normalizer(cache, tr)
    model = build_model(mode, cfg).to(device)
    freq_params = list(model.freq.parameters()) if model.freq is not None else []
    freq_ids = {id(p) for p in freq_params}
    groups = [{"params": [p for p in model.parameters() if id(p) not in freq_ids], "lr": cfg["lr"]}]
    if freq_params:
        groups.append({"params": freq_params, "lr": cfg["lr_freq"]})
    opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and device.startswith("cuda"))
    loss_fn = nn.BCEWithLogitsLoss()
    tl, vl = make_loader(cache, tr, cfg, True, device), make_loader(cache, va, cfg, False, device)

    best, best_state, bad, t0 = -1.0, None, 0, time.time()
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        tot, n = 0.0, 0
        for b in tl:
            with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                o = model(**to_inputs(b, norm, device, model.freq is not None))
                loss = loss_fn(o["logits"].float(), b["label"].to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            tot += loss.item() * len(b["label"]); n += len(b["label"])
        sched.step()
        p = predict(model, vl, norm, device, cfg["amp"])
        auc = compute_metrics(p["label"], p["prob"])["AUC"]
        improved = auc > best + 1e-4
        if improved:
            best, best_state, bad = auc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if verbose:
            print(f"  [{mode} s{seed}] ep {ep:2d} loss {tot / max(n, 1):.4f} val AUC {auc:.4f}"
                  f"{' *' if improved else ''} ({time.time() - t0:.0f}s)", flush=True)
        if bad >= cfg["patience"]:
            break

    model.load_state_dict(best_state)
    p = predict(model, vl, norm, device, cfg["amp"])
    thr = best_threshold(p["label"], p["prob"])
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, f"{mode}_s{seed}.pt")
    torch.save({"mode": mode, "seed": seed, "cfg": cfg, "state_dict": best_state, "norm": norm.state_dict(),
                "threshold": thr, "val_auc": best, "split_csv": split_csv}, ckpt)
    return {"ckpt": ckpt, "val_auc": best, "threshold": thr}


def load_trained(ckpt_path: str, device: str = "cuda"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ck["mode"], ck["cfg"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), Normalizer.from_state(ck["norm"]), ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--split_csv", required=True)
    ap.add_argument("--mode", default="full", choices=list(MODES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    r = train_one(FeatureCache(a.cache_dir), a.split_csv, a.mode, a.seed, load_cfg(a.config), a.out_dir, a.device)
    print(r)


if __name__ == "__main__":
    main()
