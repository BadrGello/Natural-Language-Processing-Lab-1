"""Evaluation entry points for Parts 2.1, 2.2, and 2.3.

Two flavours are provided for every metric:
  - ``*_numpy``  : Pure-NumPy from-scratch implementations (Part 2.3 requirement).
  - ``sklearn_evaluation`` : scikit-learn reference; usable immediately by
    either student while the NumPy version is being finished.

Zero-division convention for per-class metrics: a class that has no predicted
positives (precision) or no true positives (recall) yields 0.0 for that metric.
Macro averages are computed as unweighted means across all K classes.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


# ---------------------------------------------------------------------------
# Part 2.3 — NumPy from-scratch implementations
# ---------------------------------------------------------------------------

def confusion_matrix_numpy(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> np.ndarray:
    """Compute a K×K confusion matrix using only NumPy.

    Convention: row = true class, column = predicted class.
    ``cm[i, j]`` is the number of samples whose true label is ``i`` and
    whose predicted label is ``j``.

    Parameters
    ----------
    y_true:
        1-D integer array of ground-truth class IDs (0 … K-1).
    y_pred:
        1-D integer array of predicted class IDs, same length as ``y_true``.
    num_classes:
        Total number of classes K.  All values in ``y_true`` and ``y_pred``
        must satisfy ``0 <= value < num_classes``.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(num_classes, num_classes)``.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    # Each (true, pred) pair increments exactly one cell.
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def metrics_from_confusion_matrix_numpy(cm: np.ndarray) -> dict:
    """Compute per-class and macro-averaged P, R, F1 from a confusion matrix.

    Derives TP, FP, FN directly from the matrix:
      - TP[k]  = cm[k, k]
      - FP[k]  = sum of column k  minus TP[k]   (predicted k but wrong)
      - FN[k]  = sum of row k     minus TP[k]   (true k but missed)

    Zero-division handling: any class with a zero denominator gets 0.0.

    Parameters
    ----------
    cm:
        K×K confusion matrix returned by :func:`confusion_matrix_numpy`.

    Returns
    -------
    dict with keys:
        ``precision``       — np.ndarray of length K (per-class)
        ``recall``          — np.ndarray of length K (per-class)
        ``f1``              — np.ndarray of length K (per-class)
        ``macro_precision`` — float
        ``macro_recall``    — float
        ``macro_f1``        — float
    """
    cm = np.asarray(cm, dtype=np.float64)
    k = cm.shape[0]

    tp = np.diag(cm)                          # shape (K,)
    fp = cm.sum(axis=0) - tp                  # false positives per class
    fn = cm.sum(axis=1) - tp                  # false negatives per class

    # Precision = TP / (TP + FP),  0 if denominator is 0
    denom_p = tp + fp
    precision = np.where(denom_p > 0, tp / denom_p, 0.0)

    # Recall = TP / (TP + FN),  0 if denominator is 0
    denom_r = tp + fn
    recall = np.where(denom_r > 0, tp / denom_r, 0.0)

    # F1 = 2 * P * R / (P + R),  0 if denominator is 0
    denom_f1 = precision + recall
    f1 = np.where(denom_f1 > 0, 2 * precision * recall / denom_f1, 0.0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }


# ---------------------------------------------------------------------------
# sklearn reference (usable immediately)
# ---------------------------------------------------------------------------

def sklearn_evaluation(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> dict:
    """Reference metrics via scikit-learn; usable while NumPy version is WIP.

    Returns the same keys as :func:`metrics_from_confusion_matrix_numpy` plus
    ``confusion_matrix`` so callers can compare matrices directly.
    """
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return {
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }
