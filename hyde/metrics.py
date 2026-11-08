"""Metrics. macro-F1 and weighted-F1 are co-primary.

Averages run over the classes PRESENT in y_true (benign included; -1 'unknown' included
when it appears as a true label). Binary metrics fold every non-benign label/prediction
(including -1) into 'attack'.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score

UNKNOWN = -1


def fast_f1(y: np.ndarray, p: np.ndarray, n_classes: int) -> tuple[float, float]:
    """(macro, weighted) F1 over classes present in y; labels must be 0..n_classes-1."""
    cm = np.bincount(y * n_classes + p, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    support = cm.sum(1).astype(float)
    pred = cm.sum(0).astype(float)
    denom = support + pred
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    present = support > 0
    if not present.any():
        return 0.0, 0.0
    return float(f1[present].mean()), float((f1 * support).sum() / support.sum())


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str],
                          benign: int = 0) -> dict:
    labels = sorted(set(y_true.tolist()))
    name = lambda c: "unknown" if c == UNKNOWN else class_names[c]
    out = {}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                     average=avg, zero_division=0)
        out[f"{avg}_f1"], out[f"{avg}_precision"], out[f"{avg}_recall"] = float(f), float(p), float(r)
    out["accuracy"] = float((y_true == y_pred).mean())
    attack_labels = [c for c in labels if c != benign]
    out["macro_f1_attack_only"] = float(f1_score(y_true, y_pred, labels=attack_labels, average="macro",
                                                 zero_division=0)) if attack_labels else 0.0
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out["per_class"] = {name(c): {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                                  "support": int(s[i])} for i, c in enumerate(labels)}
    out["binary"] = binary_report(y_true, y_pred, benign)
    return out


def binary_report(y_true: np.ndarray, y_pred: np.ndarray, benign: int = 0) -> dict:
    t, q = y_true != benign, y_pred != benign
    tp, fp = int((t & q).sum()), int((~t & q).sum())
    tn, fn = int((~t & ~q).sum()), int((t & ~q).sum())
    f1_att = 2 * tp / max(2 * tp + fp + fn, 1)
    f1_ben = 2 * tn / max(2 * tn + fn + fp, 1)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "f1_attack": f1_att,
            "recall_attack": tp / max(tp + fn, 1), "precision_attack": tp / max(tp + fp, 1),
            "fpr": fp / max(fp + tn, 1), "macro_f1": (f1_att + f1_ben) / 2}


def unknown_report(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray,
                   class_names: list[str], benign: int = 0) -> dict:
    """Open-set metrics. score = max similarity (higher = more 'known')."""
    is_unk, pred_unk = y_true == UNKNOWN, y_pred == UNKNOWN
    tp = int((is_unk & pred_unk).sum())
    rec = tp / max(int(is_unk.sum()), 1)
    prec = tp / max(int(pred_unk.sum()), 1)
    known_attack = (y_true != UNKNOWN) & (y_true != benign)
    auroc = None
    if is_unk.any() and known_attack.any():
        sel = is_unk | known_attack
        auroc = float(roc_auc_score(is_unk[sel].astype(int), -score[sel]))
    known = y_true != UNKNOWN
    labels = sorted(set(y_true[known].tolist()))
    kp = y_pred[known]
    return {
        "unknown_recall": rec, "unknown_precision": prec,
        "unknown_f1": 2 * prec * rec / max(prec + rec, 1e-12), "auroc": auroc,
        "known_macro_f1": float(f1_score(y_true[known], kp, labels=labels, average="macro", zero_division=0)),
        "known_weighted_f1": float(f1_score(y_true[known], kp, labels=labels, average="weighted", zero_division=0)),
        "benign_to_unknown_rate": float(((y_true == benign) & pred_unk).sum() / max((y_true == benign).sum(), 1)),
        "n_unknown": int(is_unk.sum()),
    }
