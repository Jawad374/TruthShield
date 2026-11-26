"""Benchmark + ablation harness (Prompt 5).

Protocols
---------
main    Train every ablation mode for every seed on the pair-level 70/10/20 split
        (or reuse checkpoints already in --runs_dir) and evaluate on the test split.
        Rows: HAVIC zero-shot (official weights, no training), HAVIC re-trained head,
        +Frequency, +Emotion, full TruthShield. Also per-engine results and the
        average gate weights of the full model.
loeo    Leave-one-engine-out: for each of the 6 generators, train on the other 5,
        test on the unseen one. The strongest generalisation evidence we have.
robust  No training: evaluate the `main` checkpoints on a cache built from
        re-compressed test videos (--eval_cache_dir), e.g. H.264 CRF 28 / 35.

Thresholded metrics (ACC/P/R/F1) use the threshold tuned on the validation set;
AUC/AP/EER are threshold-free. Every trained row is mean ± std over seeds.

    python evaluate_benchmark.py --protocol main --cache_dir /content/cache \
        --splits_dir $ROOT/splits --runs_dir $ROOT/runs --results_dir $ROOT/results
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from src.data.dataset_loader import FeatureCache
from src.models.truthshield import MODE_NAMES, MODES
from src.utils.metrics import compute_metrics
from train_fusion import load_cfg, load_trained, make_loader, predict, train_one

COLS = ["ACC", "AUC", "F1", "Precision", "Recall", "AP", "EER"]
ZS = "HAVIC zero-shot (official weights)"


def fmt_table(df: pd.DataFrame, cols=("ACC", "AUC", "F1", "Precision", "Recall", "EER")) -> str:
    """df indexed by model name, columns '<metric>_mean' / '<metric>_std'."""
    head = "| Model | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    lines = []
    for name, r in df.iterrows():
        cells = []
        for c in cols:
            m, s = r.get(f"{c}_mean", np.nan), r.get(f"{c}_std", np.nan)
            pct = c in ("ACC", "AUC", "AP", "EER")
            val = f"{100 * m:.2f}%" if pct else f"{m:.4f}"
            if not np.isnan(s) and s > 0:
                val += f" ± {100 * s:.2f}" if pct else f" ± {s:.4f}"
            cells.append(val)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return head + "\n".join(lines) + "\n"


def df_md(df: pd.DataFrame) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join([str(df.index.name or "")] + cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for idx, r in df.iterrows():
        out.append("| " + " | ".join([str(idx)] + [str(v) for v in r.tolist()]) + " |")
    return "\n".join(out)


def aggregate(raw: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    agg = raw.groupby(group, sort=False)[COLS].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg


def zero_shot(cache: FeatureCache, rows: np.ndarray) -> tuple[dict, np.ndarray]:
    prob = torch.sigmoid(cache.t["havic_logit"][rows]).numpy()
    return compute_metrics(cache.label[rows].numpy(), prob, 0.5), prob


def evaluate_ckpt(ckpt: str, cache: FeatureCache, rows: np.ndarray, cfg: dict, device: str):
    model, norm, ck = load_trained(ckpt, device)
    p = predict(model, make_loader(cache, rows, cfg, False, device), norm, device, cfg["amp"])
    return compute_metrics(p["label"], p["prob"], ck["threshold"]), p


def get_ckpt(cache, split_csv, mode, seed, cfg, run_dir, device, retrain):
    path = os.path.join(run_dir, f"{mode}_s{seed}.pt")
    if os.path.exists(path) and not retrain:
        print(f"  reuse {path}")
        return path
    print(f"  training {mode} seed {seed} ...", flush=True)
    return train_one(cache, split_csv, mode, seed, cfg, run_dir, device)["ckpt"]


def per_engine(cache, rows, prob, label_name):
    eng = cache.meta["engine"].to_numpy()[rows]
    y = cache.label[rows].numpy()
    out = []
    for e in sorted(set(eng)):
        m = eng == e
        r = compute_metrics(y[m], prob[m], 0.5)
        out.append({"model": label_name, "engine": e, "AUC": r["AUC"], "ACC": r["ACC"]})
    return out


# ----------------------------------------------------------------------------- parts
# Every unit of work (one mode x seed, one held-out engine, ...) is saved as its own
# part file as soon as it finishes. Finished parts are skipped on re-runs, and the
# tables are always rebuilt by merging whatever parts exist (--merge_only: no work).
def part_file(a, protocol, key):
    d = os.path.join(a.results_dir, "parts", protocol)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{key}.pt")


def run_part(a, protocol, key, fn):
    """Load part `key` if saved, else compute it with fn() and save it."""
    f = part_file(a, protocol, key)
    if os.path.exists(f) and not a.retrain:
        return torch.load(f, weights_only=False)
    if a.merge_only:
        return None
    print(f"--- part {protocol}/{key}", flush=True)
    res = fn()
    torch.save(res, f + ".tmp")
    os.replace(f + ".tmp", f)
    print(f"--- saved part {protocol}/{key}: AUC {res['metrics']['AUC']:.4f} ACC {res['metrics']['ACC']:.4f}", flush=True)
    return res


def model_part(a, cache, split_csv, rows, mode, seed, cfg, run_dir):
    ck = get_ckpt(cache, split_csv, mode, seed, cfg, run_dir, a.device, a.retrain)
    m, p = evaluate_ckpt(ck, cache, rows, cfg, a.device)
    return {"metrics": m, "prob": p["prob"], "weights": p["weights"], "rows": p["row"]}


def zs_part(cache, rows):
    m, prob = zero_shot(cache, rows)
    return {"metrics": m, "prob": prob, "weights": None, "rows": rows}


# ----------------------------------------------------------------------------- protocols
def run_main(a, cfg, cache):
    split_csv = os.path.join(a.splits_dir, "hifi_main.csv")
    te = cache.rows_for(split_csv, "test")
    got, want = cache.coverage(split_csv, "test")
    print(f"test clips: {got} cached of {want} in split")
    run_dir = os.path.join(a.runs_dir, "main")
    raw, eng_rows, preds = [], [], {}

    zs = run_part(a, "main", "havic_zeroshot", lambda: zs_part(cache, te))
    if zs is not None:
        raw.append({"model": ZS, "seed": -1, **zs["metrics"]})
        eng_rows += per_engine(cache, te, zs["prob"], ZS)
        preds[ZS] = zs["prob"]

    gates = []
    for mode in a.modes:
        probs = []
        for seed in a.seeds:
            r = run_part(a, "main", f"{mode}_s{seed}",
                         lambda: model_part(a, cache, split_csv, te, mode, seed, cfg, run_dir))
            if r is None:
                continue
            raw.append({"model": MODE_NAMES[mode], "seed": seed, **r["metrics"]})
            probs.append(r["prob"])
            if mode == "full":
                gates.append(r["weights"])
        if probs:
            preds[MODE_NAMES[mode]] = np.mean(probs, 0)
            eng_rows += per_engine(cache, te, preds[MODE_NAMES[mode]], MODE_NAMES[mode])

    if not raw:
        print("no parts finished yet"); return
    raw = pd.DataFrame(raw)
    table = fmt_table(aggregate(raw, ["model"]))
    eng = pd.DataFrame(eng_rows)
    eng_auc = eng.pivot(index="model", columns="engine", values="AUC").loc[list(preds)]
    eng_acc = eng.pivot(index="model", columns="engine", values="ACC").loc[list(preds)]
    pct = lambda d: d.map(lambda v: f"{100 * v:.1f}%")  # noqa: E731

    md = ["# TruthShield benchmark: HiFi-AVDF test split\n",
          f"Seeds: {a.seeds}. Thresholded metrics use the validation-tuned threshold; "
          f"zero-shot HAVIC uses 0.5 like the official script.\n",
          table, "\n## AUC per generator (seed-averaged scores)\n", df_md(pct(eng_auc)),
          "\n\n## Accuracy per generator (threshold 0.5)\n", df_md(pct(eng_acc))]
    if gates:
        w = np.mean(gates, 0)  # (N, 3)
        g = pd.DataFrame(w, columns=["HAVIC", "Frequency", "Emotion"])
        g["engine"] = cache.meta["engine"].to_numpy()[te]
        g["label"] = np.where(cache.label[te].numpy() == 1, "fake", "real")
        md += ["\n\n## Full model: average gate weights (how much each branch is trusted)\n",
               df_md(g.groupby(["engine", "label"])[["HAVIC", "Frequency", "Emotion"]].mean().map(lambda v: f"{v:.3f}"))]
    save(a, "main", "\n".join(md), raw, preds, cache, te)


def run_loeo(a, cfg, cache):
    raw = []
    engines = a.engines or sorted(cache.meta["engine"].unique())
    for eng in engines:
        split_csv = os.path.join(a.splits_dir, f"hifi_loeo_{eng}.csv")
        te = cache.rows_for(split_csv, "test")
        print(f"\n=== held-out engine: {eng} ({len(te)} test clips) ===")
        zs = run_part(a, "loeo", f"{eng}__havic_zeroshot", lambda: zs_part(cache, te))
        if zs is not None:
            raw.append({"held_out": eng, "model": ZS, "seed": -1, **zs["metrics"]})
        for mode in a.modes:
            for seed in a.seeds:
                r = run_part(a, "loeo", f"{eng}__{mode}_s{seed}",
                             lambda: model_part(a, cache, split_csv, te, mode, seed, cfg,
                                                os.path.join(a.runs_dir, f"loeo_{eng}")))
                if r is not None:
                    raw.append({"held_out": eng, "model": MODE_NAMES[mode], "seed": seed, **r["metrics"]})
    if not raw:
        print("no LOEO parts finished yet"); return
    raw = pd.DataFrame(raw)
    agg = raw.groupby(["model", "held_out"], sort=False)[["AUC", "ACC"]].mean().reset_index()
    auc = agg.pivot(index="model", columns="held_out", values="AUC")
    auc["MEAN"] = auc.mean(1)
    acc = agg.pivot(index="model", columns="held_out", values="ACC")
    acc["MEAN"] = acc.mean(1)
    order = [m for m in [ZS] + [MODE_NAMES[x] for x in a.modes] if m in auc.index]
    pct = lambda d: d.loc[order].map(lambda v: f"{100 * v:.1f}%")  # noqa: E731
    md = ["# Leave-one-engine-out (train on 5 generators, test on the unseen 6th)\n",
          f"Seeds: {a.seeds}\n", "## AUC\n", df_md(pct(auc)), "\n\n## Accuracy\n", df_md(pct(acc))]
    save(a, "loeo", "\n".join(md), raw)


def run_robust(a, cfg, cache):
    split_csv = os.path.join(a.splits_dir, "hifi_main.csv")
    ev = FeatureCache(a.eval_cache_dir)
    te = ev.rows_for(split_csv, "test")
    print(f"robustness set {a.eval_cache_dir}: {len(te)} test clips")
    run_dir = os.path.join(a.runs_dir, "main")
    tag = a.tag or os.path.basename(os.path.normpath(a.eval_cache_dir))
    raw = []
    zs = run_part(a, f"robust_{tag}", "havic_zeroshot", lambda: zs_part(ev, te))
    if zs is not None:
        raw.append({"model": ZS, "seed": -1, **zs["metrics"]})

    def eval_existing(mode, seed):
        ck = os.path.join(run_dir, f"{mode}_s{seed}.pt")
        if not os.path.exists(ck):
            raise FileNotFoundError(f"{ck} missing: run --protocol main first")
        m, p = evaluate_ckpt(ck, ev, te, cfg, a.device)
        return {"metrics": m, "prob": p["prob"], "weights": p["weights"], "rows": p["row"]}

    for mode in a.modes:
        for seed in a.seeds:
            r = run_part(a, f"robust_{tag}", f"{mode}_s{seed}", lambda: eval_existing(mode, seed))
            if r is not None:
                raw.append({"model": MODE_NAMES[mode], "seed": seed, **r["metrics"]})
    raw = pd.DataFrame(raw)
    md = [f"# Robustness: test videos re-compressed ({tag})\n", fmt_table(aggregate(raw, ["model"]))]
    save(a, f"robust_{tag}", "\n".join(md), raw)


def save(a, name, md, raw, preds=None, cache=None, rows=None):
    os.makedirs(a.results_dir, exist_ok=True)
    base = os.path.join(a.results_dir, name)
    open(base + ".md", "w", encoding="utf-8").write(md)
    raw.to_csv(base + "_raw.csv", index=False)
    if preds is not None:
        df = cache.meta.iloc[rows][["video_path", "engine"]].reset_index(drop=True)
        df["label"] = cache.label[rows].numpy()
        for k, v in preds.items():
            df[k] = v
        df.to_csv(base + "_predictions.csv", index=False)
    print("\n" + md)
    print(f"\nsaved {base}.md / _raw.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["main", "loeo", "robust"], default="main")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--eval_cache_dir", help="robust: cache built from compressed test videos")
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--config")
    ap.add_argument("--tag", default="")
    ap.add_argument("--engines", nargs="+", help="loeo: only these held-out engines (run in parts)")
    ap.add_argument("--retrain", action="store_true", help="ignore existing checkpoints and parts")
    ap.add_argument("--merge_only", action="store_true", help="no training: just merge saved parts into tables")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    a.seeds = a.seeds if a.seeds is not None else cfg["seeds"]
    cache = FeatureCache(a.cache_dir)
    print(f"cache: {len(cache.meta)} clips | modes {a.modes} | seeds {a.seeds} | device {a.device}")
    {"main": run_main, "loeo": run_loeo, "robust": run_robust}[a.protocol](a, cfg, cache)
    json.dump(vars(a), open(os.path.join(a.results_dir, f"args_{a.protocol}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
