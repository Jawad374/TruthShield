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


def test_window_plan_matches_havic_official():
    from src.data.video_preprocessor import window_plan

    def official(nf, tp, ns, W=16, S=2):  # transcription of HAVIC sliding_window_infer.py
        n = nf - ((nf - W) % S)
        nwin = (n - W) // S + 1
        total = int((n / tp) * ns)
        seg = int((total * W) / (W + S * (nwin - 1)))
        st = int((S / W) * seg)
        b = []
        for i in range(nwin):
            s, e = i * st, i * st + seg
            if e > total:
                e, s = total, max(0, total - seg)
            b.append((int(s), int(e)))
        return n, list(range(0, n - W + 1, S)), b

    for nf, tp, ns, s in [(50, 50, 160000, 2), (47, 61, 97000, 2), (16, 16, 16000, 2), (33, 40, 250000, 4)]:
        assert window_plan(nf, tp, ns, stride=s) == official(nf, tp, ns, S=s)


def test_fill_missing_boxes():
    from src.data.video_preprocessor import fill_missing_boxes
    boxes, n = fill_missing_boxes([None, (1, 1, 2, 2), None, None, (5, 5, 6, 6)])
    assert n == 3 and [b[0] for b in boxes] == [1, 1, 1, 5, 5]
    with pytest.raises(ValueError):
        fill_missing_boxes([None, None])


@pytest.mark.parametrize("mode", ["havic", "havic_freq", "havic_emo", "full"])
def test_truthshield_modes(mode):
    from src.models.truthshield import TruthShield
    m = TruthShield(mode, d_model=32, freq_width=8)
    out = m(torch.randn(2, 3072), torch.randint(0, 255, (2, 2, 3, 224, 224), dtype=torch.uint8),
            torch.randn(2, 768), torch.randn(2, 768), torch.randn(2, 4))
    assert out["logits"].shape == (2,) and out["weights"].shape == (2, 3)
    assert (m.freq is None) == ("freq" not in m.active) and (m.emo is None) == ("emotion" not in m.active)
    out["logits"].sum().backward()
