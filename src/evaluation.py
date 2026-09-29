import numpy as np
from sklearn.metrics import (precision_recall_curve, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)


def best_f1_threshold(y, s):
    """Threshold maximizing F1 (call on validation data only)."""
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[np.argmax(f)])


def classification_report(y, s, threshold):
    yh = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return {"threshold": float(threshold), "precision": float(precision_score(y, yh, zero_division=0)),
            "recall": float(recall_score(y, yh, zero_division=0)), "f1": float(f1_score(y, yh, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}
