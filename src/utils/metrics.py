"""Binary deepfake-detection metrics (label 1 = fake)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "ACC": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    if len(np.unique(y_true)) == 2:
        out["AUC"] = roc_auc_score(y_true, y_prob)
        out["AP"] = average_precision_score(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fnr = 1 - tpr
        i = np.nanargmin(np.abs(fnr - fpr))
        out["EER"] = float((fpr[i] + fnr[i]) / 2)
    else:
        out["AUC"] = out["AP"] = out["EER"] = float("nan")
    return {k: float(v) for k, v in out.items()}


def best_threshold(y_true, y_prob) -> float:
    """Threshold maximising Youden's J on a validation set (never tune on test)."""
    fpr, tpr, thr = roc_curve(np.asarray(y_true).astype(int), np.asarray(y_prob, dtype=float))
    return float(np.clip(thr[np.argmax(tpr - fpr)], 0.0, 1.0))


def to_markdown(rows: dict[str, dict], cols=("ACC", "AUC", "F1", "Precision", "Recall")) -> str:
    head = "| Model | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    body = ""
    for name, m in rows.items():
        cells = []
        for c in cols:
            v = m.get(c, float("nan"))
            cells.append(f"{v:.4f}" if c in ("F1", "Precision", "Recall") else f"{100 * v:.2f}%")
        body += f"| {name} | " + " | ".join(cells) + " |\n"
    return head + body
