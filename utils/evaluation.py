"""Evaluation entry points for Parts 2.1, 2.2, and 2.3.

Student 2 owns the NumPy implementations below. In the meantime, Student 3
can use sklearn_evaluation() for independently verifying predictions.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def confusion_matrix_numpy(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> np.ndarray:
    """TODO (Student 2): implement a K x K confusion matrix with NumPy only.

    Convention: row = true class, column = predicted class.
    """
    raise NotImplementedError("Student 2: implement confusion_matrix_numpy")


def metrics_from_confusion_matrix_numpy(cm: np.ndarray) -> dict:
    """TODO (Student 2): compute per-class/macro P, R, F1 using NumPy.

    Return keys: precision, recall, f1 (each array of length K) and
    macro_precision, macro_recall, macro_f1 (each float).
    Pick and document a zero-denominator convention (e.g., 0.0).
    """
    raise NotImplementedError("Student 2: implement metrics_from_confusion_matrix_numpy")


def sklearn_evaluation(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> dict:
    """Reference metrics; usable immediately while NumPy version is unfinished."""
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
