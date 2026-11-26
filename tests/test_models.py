import pandas as pd
import pytest
import torch

from src.models.dynamic_fusion import BRANCHES, DynamicFusion
from src.models.emotion_branch import EmotionConsistencyHead
from src.models.frequency_branch import FrequencyBranch, fft_log_magnitude
from src.utils.metrics import compute_metrics


@pytest.mark.parametrize("use_dwt", [True, False])
def test_frequency_branch_shapes(use_dwt):
    m = FrequencyBranch(use_dwt=use_dwt).eval()
    assert m(torch.rand(2, 3, 224, 224)).shape == (2, 128)
    assert m(torch.randint(0, 255, (2, 4, 3, 224, 224), dtype=torch.uint8)).shape == (2, 128)


def test_fft_stable_on_constant_image():
    out = fft_log_magnitude(torch.zeros(1, 3, 224, 224))
    assert torch.isfinite(out).all()


def test_frequency_branch_backward():
    m = FrequencyBranch()
    x = torch.rand(2, 3, 224, 224, requires_grad=True)
    m(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_emotion_head_range():
    h = EmotionConsistencyHead()
    s, zf, zv = h(torch.randn(8, 768), torch.randn(8, 768))
    assert s.shape == (8, 1) and zf.shape == zv.shape == (8, 64)
    assert (s >= 0).all() and (s <= 2).all()
    same = torch.randn(4, 768)
    h2 = EmotionConsistencyHead(768, 768)
    h2.voice_proj.load_state_dict(h2.face_proj.state_dict())
    assert torch.allclose(h2(same, same)[0], torch.zeros(4, 1), atol=1e-5)


@pytest.mark.parametrize("active", [("havic",), ("havic", "freq"), ("havic", "emotion"), BRANCHES])
def test_fusion_ablation_modes(active):
    m = DynamicFusion(active=active)
    logits, w = m(torch.randn(5, 3072), torch.randn(5, 128), torch.rand(5) * 2, torch.randn(5, 4))
    assert logits.shape == (5,) and w.shape == (5, 3)
    assert torch.allclose(w.sum(-1), torch.ones(5))
    for i, b in enumerate(BRANCHES):
        if b not in active:
            assert (w[:, i] == 0).all()
    logits.sum().backward()


def test_fusion_inactive_inputs_ignored():
    m = DynamicFusion(active=("havic",)).eval()
    h = torch.randn(3, 3072)
    a = m(h, torch.randn(3, 128), torch.rand(3, 1))[0]
    b = m(h, torch.randn(3, 128), torch.rand(3, 1))[0]
    assert torch.allclose(a, b)


def test_metrics():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
    assert m["ACC"] == 1.0 and m["AUC"] == 1.0 and m["F1"] == 1.0


def test_pair_split_no_leakage(tmp_path):
    from src.data.make_splits import load_index, split_pairs
    rows = []
    for i in range(50):
        rows += [(f"sora2/{i}.mp4", 0, 0, 0), (f"sora2/{i}_generated.mp4", 1, 1, 1)]
    pd.DataFrame(rows, columns=["video_path", "audio_label", "visual_label", "overall_label"]).to_csv(
        tmp_path / "sora2.csv", index=False)
    df = split_pairs(load_index(str(tmp_path)))
    assert (df.groupby("pair_id")["split"].nunique() == 1).all()
    assert set(df["split"]) == {"train", "val", "test"}
