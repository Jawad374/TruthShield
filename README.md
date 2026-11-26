# TruthShield

Multimodal deepfake detection on modern generators. It extends HAVIC (audio-visual coherence) with:
- **Emotion consistency:** how far facial affect and vocal affect disagree.
- **Frequency artifacts:** FFT and DWT spectra fed to a residual CNN.
- **Quality-aware dynamic gating:** learns how much to trust each branch per clip.

See `TruthShield_Implementation_Blueprint.pdf` for the research plan.

## Workflow
- **Write code locally** in VS Code. Run the CPU tests with `python -m pytest -q tests`.
- **Run GPU work on Colab (T4)**, using the notebooks in `notebooks/`. Data, weights and cached features live in Google Drive under `MyDrive/TruthShield/`.
- **Sync** through GitHub. Push from here; the notebooks `git pull`.

## Layout
```
src/models/frequency_branch.py   FFT + Haar DWT -> 4-stage residual CNN -> 128-d   (Prompt 1)
src/models/emotion_branch.py     frozen SER/FER extractors + trainable 64-d cosine mismatch head (Prompt 2)
src/models/dynamic_fusion.py     quality-aware softmax gating + classifier, ablation masks (Prompt 3)
src/data/make_splits.py          pair-level 70/10/20 split + leave-one-engine-out splits
src/utils/metrics.py             ACC / AUC / AP / EER / P / R / F1, markdown tables
notebooks/00_setup_and_havic_baseline.ipynb   Colab: download, verify weights, HAVIC zero-shot baseline
third_party/HAVIC                official HAVIC code (git clone, not committed)
```

## Changes from the blueprint, and why
| Blueprint | Implemented | Reason |
|---|---|---|
| HAVIC features are 512-d | 3072-d (`HAVIC_FT` final_feat = 4×768), projected inside the fusion | This is the real dimension in `third_party/HAVIC/src/models/HAVIC.py`. |
| Cache all branch embeddings | Cache HAVIC features and emotion embeddings; for the frequency branch, cache **face crops** (K frames, uint8) | The frequency CNN is trainable, so it needs pixels. Grad-CAM needs pixels too. |
| Cache the emotion mismatch scalar | Cache the raw SER and FER embeddings; train the 64-d projections jointly | Untrained projections would give a random score. |
| Random 70/10/20 split | Split by real/fake **pair**, stratified by engine | Each real clip has a matching `_generated` fake. A per-clip split would leak. |
| DAN / FER+ | `trpakov/vit-face-expression` (HF) | Same purpose, loads through `transformers` with one line. |
| HiFi-AVDF engines listed as Sora 2, Kling 2.5, PixVerse V5, Wan 2.5, Veo 3.1 | Those plus **Seedance 1.0** (6 engines, 3,810 clips) | The released dataset includes Seedance. |

**Fair-comparison note:** HAVIC is zero-shot on HiFi-AVDF, but the TruthShield fusion is trained on HiFi-AVDF train. The ablation table should therefore also include **"HAVIC features + linear head trained on HiFi-AVDF"**. Then any gain can be credited to the new clues and not just to in-domain training. Leave-one-engine-out results make the strongest generalisation claim.
