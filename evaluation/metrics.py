"""
evaluation/metrics.py
---------------------
Metric computation shared by both the classical and the quantum models.

For binary medical classification the important numbers are:

    * Accuracy        - fraction of all predictions that were correct
    * Precision       - of the positive predictions, how many were right
    * Recall/Sensitivity - of the actual positive cases, how many we caught
    * Specificity     - of the actual negative cases, how many we correctly ruled out
    * F1-score        - balanced combination of precision and recall

Everything is computed from the REAL predictions of the trained models.
No value is hardcoded anywhere in this project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None
) -> dict:
    """Compute the full set of metrics for a binary classification task.

    The class considered 'positive' is the second class (code 1).
    """
    cm = confusion_matrix(y_true, y_pred)

    # If the test set accidentally contains only one class, the matrix is 1x1.
    if cm.shape == (1, 1):
        tn, fp = 0, 0
        fn, tp = 0, 0
    else:
        tn, fp, fn, tp = cm.ravel().tolist()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    auc = float("nan")
    if y_prob is not None and len(set(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float("nan")

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "auc": float(auc),
        "confusion_matrix": cm,
    }
    return metrics


def metrics_dataframe(classical: dict, quantum: dict | None) -> pd.DataFrame:
    """Turn the metric dicts into a small comparison table."""
    rows = {"Metric": ["Accuracy", "Precision", "Recall (Sensitivity)", "Specificity", "F1-score", "AUC"]}
    rows["Classical ML"] = [
        f"{classical['metrics'][k]:.3f}" for k in ("accuracy", "precision", "recall", "specificity", "f1", "auc")
    ]
    if quantum is not None:
        rows["Hybrid Quantum ML"] = [
            f"{quantum['metrics'][k]:.3f}" for k in ("accuracy", "precision", "recall", "specificity", "f1", "auc")
        ]
    return pd.DataFrame(rows)


def classification_report_text(y_true, y_pred, class_names) -> str:
    """Short readable summary of predictions."""
    correct = int((np.asarray(y_true) == np.asarray(y_pred)).sum())
    total = len(y_true)
    table = pd.DataFrame(
        {
            "Class": class_names,
            "Count (actual)": [int((np.asarray(y_true) == i).sum()) for i in range(len(class_names))],
        }
    )
    return table, f"Correctly predicted {correct} out of {total} test samples ({correct / max(total, 1):.1%})."