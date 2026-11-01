"""Evaluation metrics: AUC, KS, Gini, calibration, confusion matrix."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def auc_roc(y_true, y_score) -> float:
    """Area under the ROC curve."""
    return float(roc_auc_score(y_true, y_score))


def gini(y_true, y_score) -> float:
    """Gini coefficient = 2 * AUC - 1."""
    return 2.0 * auc_roc(y_true, y_score) - 1.0


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov statistic: max separation of score CDFs.

    KS = max(|CDF_default(score) - CDF_nondefault(score)|).
    Computed on the held-out test set only.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    scores_default = y_score[y_true == 1]
    scores_nondefault = y_score[y_true == 0]
    if len(scores_default) == 0 or len(scores_nondefault) == 0:
        return 0.0
    all_scores = np.unique(np.concatenate([scores_default, scores_nondefault]))
    cdf_default = np.searchsorted(np.sort(scores_default), all_scores, side="right") / len(
        scores_default
    )
    cdf_nondefault = np.searchsorted(
        np.sort(scores_nondefault), all_scores, side="right"
    ) / len(scores_nondefault)
    return float(np.max(np.abs(cdf_default - cdf_nondefault)))


def roc_curve_data(y_true, y_score):
    """Return fpr, tpr, thresholds from roc_curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return fpr, tpr, thresholds


def calibration_by_decile(
    y_true, y_score, n_bins: int = 10
) -> pd.DataFrame:
    """Observed vs predicted default rate by score decile."""
    df = pd.DataFrame({"y": y_true, "score": y_score})
    df["decile"] = pd.qcut(df["score"].rank(method="first"), q=n_bins, labels=False)
    grouped = df.groupby("decile").agg(
        predicted=("score", "mean"),
        observed=("y", "mean"),
        n=("y", "size"),
    )
    return grouped.reset_index()  # type: ignore[no-any-return]


def brier_score(y_true, y_score) -> float:
    """Brier score (calibration + refinement)."""
    return float(np.mean((np.asarray(y_score) - np.asarray(y_true)) ** 2))


def confusion_at_threshold(
    y_true, y_score, threshold: float = 0.5
) -> dict:
    """Confusion matrix metrics at a business threshold."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
    }


def evaluate(y_true, y_score, threshold: float = 0.5) -> dict:
    """Aggregate evaluation report: AUC, KS, Gini, Brier, confusion matrix.

    All returned values are JSON-serializable primitives. Use
    ``calibration_by_decile()`` separately if a calibration frame is needed.
    """
    return {
        "auc": auc_roc(y_true, y_score),
        "ks": ks_statistic(y_true, y_score),
        "gini": gini(y_true, y_score),
        "brier": brier_score(y_true, y_score),
        "confusion": confusion_at_threshold(y_true, y_score, threshold),
    }
