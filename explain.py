"""Phase 3 figures: per-clip explanations + dataset-level spectral signatures.

    python explain.py --cache_dir /content/cache --splits_dir $ROOT/splits \
        --ckpt $ROOT/runs/main/full_s0.pt --out_dir $ROOT/results/figures

Picks, per generator, the most confident correctly caught fake, the worst missed
fake and the worst false alarm on the test split, and saves:
  explain_<engine>.png      face | log-FFT | spectral Grad-CAM | pixel saliency
  spectrum_difference.png   mean log-spectrum (fake - real) per generator
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from src.data.dataset_loader import FeatureCache
from src.explainability.gradcam import (explanation_figure, mean_spectrum_difference,
                                        spectrum_difference_figure)
from train_fusion import load_trained, make_loader, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--ckpt", required=True, help="a 'full' (or *_freq) checkpoint")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--engines", nargs="+", help="only these generators (run in parts)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    cache = FeatureCache(a.cache_dir)
    te = cache.rows_for(os.path.join(a.splits_dir, "hifi_main.csv"), "test")
    model, norm, ck = load_trained(a.ckpt, a.device)
    if model.freq is None:
        raise SystemExit("explanations need a checkpoint with the frequency branch (full / havic_freq)")
    p = predict(model, make_loader(cache, te, ck["cfg"], False, a.device), norm, a.device, ck["cfg"]["amp"])
    rows, prob, y = p["row"].astype(int), p["prob"], p["label"].astype(int)
    eng = cache.meta["engine"].to_numpy()[rows]
    thr = ck["threshold"]

    for e in a.engines or sorted(set(eng)):
        m = eng == e
        pick, titles = [], []
        fakes, reals = np.where(m & (y == 1))[0], np.where(m & (y == 0))[0]
        if len(fakes):
            pick.append(rows[fakes[np.argmax(prob[fakes])]]); titles.append("most confident fake")
            worst = fakes[np.argmin(prob[fakes])]
            if prob[worst] < thr:
                pick.append(rows[worst]); titles.append("MISSED fake")
        if len(reals):
            worst = reals[np.argmax(prob[reals])]
            if prob[worst] >= thr:
                pick.append(rows[worst]); titles.append("FALSE ALARM on real")
        if pick:
            f = explanation_figure(model, cache, pick, norm, a.device,
                                   os.path.join(a.out_dir, f"explain_{e}.png"), titles)
            print("saved", f, flush=True)

    f = spectrum_difference_figure(mean_spectrum_difference(cache, te),
                                   os.path.join(a.out_dir, "spectrum_difference.png"))
    print("saved", f)


if __name__ == "__main__":
    main()
