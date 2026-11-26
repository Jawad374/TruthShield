"""One-time feature caching: raw .mp4 -> per-clip .pt files on Drive (Prompt 4).

For every clip it stores:
  havic_feat  [3072]          HAVIC_FT final feature (mean over sliding windows)
  havic_logit []              HAVIC_FT output logit (mean over windows) -> "HAVIC alone" row
  faces       [K,3,224,224]   uint8 face crops for the trainable frequency branch
  face_emb    [768], face_probs [7]    frozen facial-expression model
  voice_emb   [768], voice_probs [4]   frozen speech-emotion model
  quality     [4]             face size, sharpness, bitrate, loudness
  label, engine, pair_id, video_path, n_windows, n_missing_faces

Resumable: clips whose .pt already exists are skipped, so after a Colab
disconnect just run the same command again. Failures go to failures.csv.

Usage (Colab):
  python cache_features.py --split_csv $ROOT/splits/hifi_main_abs.csv \
      --havic_root /content/HAVIC --havic_weights $ROOT/weights/best_ft_model_clean.pth \
      --out_dir $ROOT/cache/hifi
  python cache_features.py --pack --out_dir $ROOT/cache/hifi      # merge into 2 files for training
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- HAVIC
def patch_havic_for_new_timm(havic_root: str):
    f = os.path.join(havic_root, "src", "models", "interaction_modules.py")
    s = open(f, encoding="utf-8").read()
    s2 = s.replace("qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)",
                   "qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)")
    if s2 != s:
        open(f, "w", encoding="utf-8").write(s2)


def import_havic_class(havic_root: str):
    """Import HAVIC_FT from HAVIC's own `src` package without clobbering ours."""
    patch_havic_for_new_timm(havic_root)
    # HAVIC's `src` has no __init__.py (namespace package), so any regular `src`
    # package on sys.path (ours) would win regardless of order: hide those entries.
    saved_mods = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "src" or k.startswith("src.")}
    saved_path = list(sys.path)
    sys.path[:] = [havic_root] + [p for p in sys.path
                                  if not os.path.isfile(os.path.join(p or os.getcwd(), "src", "__init__.py"))]
    try:
        from src.models.HAVIC import HAVIC_FT  # noqa: E402  (HAVIC's src)
    finally:
        sys.path[:] = saved_path
        for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            del sys.modules[k]
        sys.modules.update(saved_mods)
    return HAVIC_FT


def load_havic(havic_root: str, weights: str, device: str):
    """HAVIC_FT with fine-tuned weights (strict: no missing keys) + a hook that
    captures the 3072-d feature entering the final classifier."""
    model = import_havic_class(havic_root)()
    sd = torch.load(weights, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"HAVIC weights missing {len(missing)} keys, e.g. {missing[:5]}")
    model.eval().to(device)

    feats = {}
    model.classifier.register_forward_hook(lambda m, inp, out: feats.__setitem__("x", inp[0].detach()))
    return model, feats


# ----------------------------------------------------------------------------- per clip
def decode(row: dict):
    """CPU part (runs in a background thread): frames + audio."""
    from src.data.video_preprocessor import load_audio_16k, sample_frames

    frames, n_tp, dur = sample_frames(row["abs_path"])
    wav = load_audio_16k(row["abs_path"])
    return frames, n_tp, dur, wav


@torch.no_grad()
def process(row, decoded, det, havic, havic_feats, emo, args):
    from src.data.video_preprocessor import (crops_to_tensor, fill_missing_boxes, havic_batches,
                                             pick_k, quality_features)

    frames, n_tp, dur, wav = decoded
    boxes, n_missing = fill_missing_boxes([det.detect(f) for f in frames])
    faces = crops_to_tensor(frames, boxes)  # [N,3,224,224] uint8

    video, audio = havic_batches(faces, wav, n_tp, stride=args.stride)
    logits, hfeats = [], []
    for i in range(0, video.shape[0], args.havic_bs):
        with torch.autocast("cuda", dtype=torch.float16, enabled=args.device.startswith("cuda") and not args.fp32):
            out = havic(audio=audio[i:i + args.havic_bs].to(args.device),
                        video=video[i:i + args.havic_bs].to(args.device), is_training=False)
        logits.append(out.float().view(-1).cpu())
        hfeats.append(havic_feats["x"].float().cpu())
    logits, hfeats = torch.cat(logits), torch.cat(hfeats)

    k_idx = pick_k(faces.shape[0], args.k_frames)
    k_faces = faces[k_idx].clone()
    # emotion: voice over the aligned audio, face over the K crops
    n_used_audio = int(min(1.0, faces.shape[0] / n_tp) * wav.shape[1])
    v = emo.voice(wav[0, :n_used_audio])
    f = emo.face(k_faces)

    return {
        "havic_feat": hfeats.mean(0).half(),
        "havic_logit": logits.mean(),
        "faces": k_faces,
        "face_emb": f["face_emb"].half(), "face_probs": f["face_probs"],
        "voice_emb": v["voice_emb"].half(), "voice_probs": v["voice_probs"],
        "quality": quality_features(faces, boxes, os.path.getsize(row["abs_path"]), dur, wav),
        "label": int(row["label"]), "engine": row["engine"], "pair_id": row["pair_id"],
        "video_path": row["video_path"], "n_windows": int(video.shape[0]), "n_missing_faces": int(n_missing),
    }


def out_path(out_dir, row):
    return os.path.join(out_dir, "clips", row["video_path"].replace("/", "__").replace(".mp4", ".pt"))


# ----------------------------------------------------------------------------- main loops
def run_cache(args):
    sys.path.insert(0, HERE)
    from src.models.emotion_branch import EmotionExtractor
    from src.data.video_preprocessor import GPUFaceDetector

    with open(args.split_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        rows = rows[i::n]
    os.makedirs(os.path.join(args.out_dir, "clips"), exist_ok=True)
    todo = [r for r in rows if not os.path.exists(out_path(args.out_dir, r))]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(rows)} clips in split, {len(rows) - len(todo)} already cached, {len(todo)} to do", flush=True)
    if not todo:
        return

    print("loading models ...", flush=True)
    det = GPUFaceDetector(args.havic_root, args.device)
    havic, havic_feats = load_havic(args.havic_root, args.havic_weights, args.device)
    emo = EmotionExtractor(args.device)
    print("models loaded", flush=True)

    fail_path = os.path.join(args.out_dir, "failures.csv")
    t0, ok, bad = time.time(), 0, 0
    # decode the next few videos on CPU threads while the GPU works on the current one
    ahead = args.workers + 1
    pool = ThreadPoolExecutor(max_workers=args.workers)
    futures = {j: pool.submit(decode, todo[j]) for j in range(min(ahead, len(todo)))}
    for i, row in enumerate(todo):
        if i + ahead < len(todo):
            futures[i + ahead] = pool.submit(decode, todo[i + ahead])
        try:
            decoded = futures.pop(i).result()
            rec = process(row, decoded, det, havic, havic_feats, emo, args)
            dst = out_path(args.out_dir, row)
            torch.save(rec, dst + ".tmp")
            os.replace(dst + ".tmp", dst)  # atomic: no half-written files after a disconnect
            ok += 1
            msg = (f"true={'FAKE' if rec['label'] else 'REAL'} havic_p={torch.sigmoid(rec['havic_logit']):.3f} "
                   f"win={rec['n_windows']} miss_face={rec['n_missing_faces']}")
        except Exception as e:  # corrupt video, no face, no audio ...
            bad += 1
            msg = f"FAILED: {type(e).__name__}: {str(e)[:120]}"
            new = not os.path.exists(fail_path)
            with open(fail_path, "a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["video_path", "error"])
                w.writerow([row["video_path"], f"{type(e).__name__}: {e}"])
            if args.debug:
                traceback.print_exc()
        el = time.time() - t0
        left = el / (i + 1) * (len(todo) - i - 1)
        print(f"[{i + 1:4d}/{len(todo)}] {row['video_path']:34s} {msg} | {el / 60:5.1f} min, ~{left / 60:5.1f} min left",
              flush=True)
        if (i + 1) % 50 == 0:
            gc.collect(); torch.cuda.empty_cache()
    pool.shutdown(wait=False, cancel_futures=True)
    print(f"done: {ok} cached, {bad} failed (see {fail_path})")


def run_pack(args):
    """Merge per-clip files into features.pt (small tensors + metadata) and faces.npy (memmap)."""
    import glob

    files = sorted(glob.glob(os.path.join(args.out_dir, "clips", "*.pt")))
    if not files:
        raise SystemExit("no cached clips found")
    first = torch.load(files[0], weights_only=False)
    K = first["faces"].shape[0]
    faces = np.lib.format.open_memmap(os.path.join(args.out_dir, "faces.npy"), mode="w+",
                                      dtype=np.uint8, shape=(len(files), K, 3, 224, 224))
    keys = ["havic_feat", "havic_logit", "face_emb", "face_probs", "voice_emb", "voice_probs", "quality"]
    acc = {k: [] for k in keys}
    meta = {k: [] for k in ["label", "engine", "pair_id", "video_path", "n_windows", "n_missing_faces"]}
    for i, f in enumerate(files):
        r = torch.load(f, weights_only=False)
        fk = r["faces"]
        if fk.shape[0] < K:  # very short clip: repeat last frame
            fk = torch.cat([fk, fk[-1:].repeat(K - fk.shape[0], 1, 1, 1)])
        faces[i] = fk.numpy()
        for k in keys:
            acc[k].append(r[k].float())
        for k in meta:
            meta[k].append(r[k])
        if (i + 1) % 500 == 0:
            print(f"packed {i + 1}/{len(files)}", flush=True)
    faces.flush()
    out = {k: torch.stack(v) for k, v in acc.items()}
    out["label"] = torch.tensor(meta.pop("label"))
    out["meta"] = meta
    torch.save(out, os.path.join(args.out_dir, "features.pt"))
    print(f"wrote features.pt ({len(files)} clips) and faces.npy {faces.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_csv")
    ap.add_argument("--havic_root", default="/content/HAVIC")
    ap.add_argument("--havic_weights")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k_frames", type=int, default=8)
    ap.add_argument("--stride", type=int, default=2, help="HAVIC sliding-window stride (official: 2)")
    ap.add_argument("--havic_bs", type=int, default=8, help="windows per HAVIC forward pass")
    ap.add_argument("--workers", type=int, default=3, help="CPU threads decoding upcoming videos")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="only process N clips (smoke test)")
    ap.add_argument("--shard", default="", help="e.g. 0/2 to split work across sessions")
    ap.add_argument("--fp32", action="store_true", help="disable fp16 autocast for HAVIC (slower, exact)")
    ap.add_argument("--pack", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    if args.pack:
        run_pack(args)
    else:
        if not (args.split_csv and args.havic_weights):
            ap.error("--split_csv and --havic_weights are required for caching")
        run_cache(args)


if __name__ == "__main__":
    main()
