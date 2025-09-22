# llm_mol_interp/utils/metrics.py
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from sklearn.metrics import roc_auc_score

def _valid_mask(y_true_col: np.ndarray) -> np.ndarray:
    # valid if label is 0 or 1 (ignore NaN / -1)
    return np.isfinite(y_true_col) & (y_true_col >= 0.0)

def masked_auc(y_true: np.ndarray, y_prob: np.ndarray):
    """
    y_true, y_prob: [N, T] with NaN (or -1) for missing labels in y_true.
    Returns (macro_auc, per_task_auc[T]) where invalid tasks get np.nan.
    """
    T = y_true.shape[1]
    per = []
    for t in range(T):
        m = _valid_mask(y_true[:, t])
        if m.sum() < 2:
            per.append(np.nan); continue
        yt = y_true[m, t]
        yp = y_prob[m, t]
        # must have both classes present
        if len(np.unique(yt)) < 2:
            per.append(np.nan); continue
        try:
            per.append(roc_auc_score(yt, yp))
        except Exception:
            per.append(np.nan)
    per = np.array(per, dtype=float)
    if np.all(np.isnan(per)):
        return np.nan, per
    return np.nanmean(per), per

def masked_aupr(y_true: np.ndarray, y_prob: np.ndarray):
    """
    Macro average of average_precision_score with same masking as above.
    """
    T = y_true.shape[1]
    per = []
    for t in range(T):
        m = _valid_mask(y_true[:, t])
        if m.sum() < 2:
            per.append(np.nan); continue
        yt = y_true[m, t]
        yp = y_prob[m, t]
        # need at least one positive
        if (yt == 1).sum() == 0:
            per.append(np.nan); continue
        try:
            per.append(average_precision_score(yt, yp))
        except Exception:
            per.append(np.nan)
    per = np.array(per, dtype=float)
    if np.all(np.isnan(per)):
        return np.nan, per
    return np.nanmean(per), per


"""
def masked_auc(y_true, y_pred):
    # y_true, y_pred: [N, T] numpy
    aucs = []
    for t in range(y_true.shape[1]):
        mask = y_true[:,t] >= 0
        if mask.sum() < 5:
            aucs.append(np.nan); continue
        y = y_true[mask, t]; p = y_pred[mask, t]
        if len(np.unique(y)) < 2:
            aucs.append(np.nan); continue
        aucs.append(roc_auc_score(y, p))
    return np.nanmean(aucs), aucs
    
"""
