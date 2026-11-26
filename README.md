# TruthShield

Multimodal deepfake detection on modern generators (Sora 2, Veo 3.1, Kling 2.5, PixVerse V5, Wan 2.5, Seedance 1.0). It extends HAVIC (audio-visual coherence) with:
- **Emotion consistency:** how far facial affect and vocal affect disagree.
- **Frequency artifacts:** FFT and Haar-DWT spectra fed to a residual CNN.
- **Quality-aware dynamic gating:** learns how much to trust each branch per clip.

Dataset: **HiFi-AVDF only** (3,810 clips, 6 generators). FakeAVCeleb and KoDF need access requests, so they are out of scope. Generalisation is shown with leave-one-engine-out instead.

## How to run (Colab, T4 GPU), in order
Every long job runs in **parts** that are saved to Google Drive as they finish. Re-running skips finished parts, and a merge step builds the final tables.

| Notebook | What it does | Parts |
|---|---|---|
| `00_setup_and_havic_baseline` | Downloads data and weights, builds splits, checks the weights load, runs HAVIC's official zero-shot script | 50 videos per part |
| `01_cache_features` | Runs every video once through HAVIC, the face detector and the emotion models, then packs the results | 1 generator per part (each video also saved individually) |
| `02_train_and_evaluate` | Trains the 4 ablation models × 3 seeds; runs leave-one-engine-out; makes the explanation figures | 1 model × seed per part |
| `03_robustness` | Re-encodes the test videos (H.264 CRF 28 / 35) and re-evaluates the trained models | 1 quality level × generator per part |

All outputs go to `MyDrive/TruthShield/results/`.

## Layout
```
src/models/frequency_branch.py   FFT + Haar DWT -> 4-stage residual CNN -> 128-d             (Prompt 1)
src/models/emotion_branch.py     frozen SER/FER extractors + trainable 64-d cosine mismatch  (Prompt 2)
src/models/dynamic_fusion.py     quality-aware softmax gating + classifier, ablation masks   (Prompt 3)
src/models/truthshield.py        full model; modes havic / havic_freq / havic_emo / full
src/data/make_splits.py          pair-level 70/10/20 split + leave-one-engine-out splits
src/data/video_preprocessor.py   HAVIC-identical frame/face/audio preprocessing (GPU face detection)
src/data/dataset_loader.py       packed-cache dataset, train-only normalisation
src/explainability/gradcam.py    pixel saliency, spectral Grad-CAM, mean spectrum difference
src/utils/metrics.py             ACC / AUC / AP / EER / P / R / F1
cache_features.py                one-time, resumable caching; --engines / --only_split / --crf  (Prompt 4)
train_fusion.py                  trains one configuration (early stopping on val AUC)
evaluate_benchmark.py            protocols main / loeo / robust, saved in parts, --merge_only   (Prompt 5)
explain.py                       Phase 3 figures
configs/train_fusion.yaml        hyper-parameters
tests/                           CPU tests: python -m pytest -q tests
```

## Results produced
1. **Main ablation table** (test split, mean ± std over 3 seeds): HAVIC zero-shot, HAVIC re-trained head, + Frequency, + Emotion, full TruthShield. Accuracy, AUC, F1, precision, recall and EER, plus per-generator AUC and accuracy.
2. **Gate analysis:** the average branch weights of the full model, per generator and per real/fake.
3. **Leave-one-engine-out:** AUC and accuracy on each generator when it was never seen in training.
4. **Robustness:** the same table on H.264-recompressed test videos (CRF 28 and CRF 35).
5. **Figures:** per-generator explanations (face, log-FFT, spectral Grad-CAM, pixel saliency) and the fake-minus-real spectrum signature.

## Changes from the blueprint, and why
| Blueprint | Implemented | Reason |
|---|---|---|
| HAVIC features are 512-d | 3072-d (`HAVIC_FT` final feature = 4×768), projected inside the fusion | This is the real dimension in HAVIC's `src/models/HAVIC.py`. |
| Cache all branch embeddings | Cache HAVIC features and emotion embeddings; for the frequency branch, cache **face crops** (8 frames, uint8) | The frequency CNN is trainable, so it needs pixels. The explanations need pixels too. |
| Cache the emotion mismatch scalar | Cache the raw SER and FER embeddings; train the 64-d projections jointly | Untrained projections would give a random score. |
| Random 70/10/20 split | Split by real/fake **pair**, stratified by engine | Each real clip has a matching `_generated` fake. A per-clip split would leak. |
| DAN / FER+ | `trpakov/vit-face-expression` (HF) | Same purpose, loads through `transformers` with one line. |
| 5 generators | 6 (the released dataset adds **Seedance 1.0**) | Matches the released dataset. |
| FakeAVCeleb / KoDF cross-dataset tests | Leave-one-engine-out on HiFi-AVDF | Those datasets need access requests. Testing on an unseen generator is the closest substitute. |
| Grad-CAM over facial coordinates | Pixel saliency (where on the face) + spectral Grad-CAM (which frequencies) | The frequency branch's feature maps are in the frequency domain, so Grad-CAM there cannot localise facial regions on its own. |

**Fair-comparison note:** HAVIC's official weights were never trained on HiFi-AVDF, while TruthShield trains on its train split. The table therefore also includes **"HAVIC (re-trained head)"**, a new classifier on HAVIC features trained on the same data. TruthShield's gain is measured against that row.
