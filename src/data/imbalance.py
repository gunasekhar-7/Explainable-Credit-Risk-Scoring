"""Class imbalance handling: class weights, scale_pos_weight, optional SMOTE toggle."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


def pos_fraction(y: pd.Series) -> float:
    """Fraction of positive (default) cases."""
    return float((y == 1).mean())


def neg_pos_ratio(y: pd.Series) -> float:
    """neg / pos ratio used for XGBoost scale_pos_weight."""
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return neg / pos if pos > 0 else 1.0


def logreg_class_weight(y: pd.Series) -> dict:
    """'balanced' class weights mapping {0: w0, 1: w1} for LogisticRegression."""
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights, strict=True)}


def xgb_scale_pos_weight(y: pd.Series) -> float:
    """scale_pos_weight = neg / pos for XGBoost."""
    return float(neg_pos_ratio(y))


def apply_smote(
    X: pd.DataFrame, y: pd.Series, random_seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Optional SMOTE oversampling of the minority class.

    WARNING: SMOTE distorts rate calibration. Not recommended for the production
    scorecard path; kept as a documented toggle for experimentation only.
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(  # noqa: TRY003
            "imblearn is required for SMOTE. Install imbalanced-learn or avoid SMOTE."
        ) from exc

    smote = SMOTE(random_state=random_seed)
    X_res, y_res = smote.fit_resample(X, y)
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)


def stratify_split_info(y: pd.Series) -> dict:
    """Return class counts and split sizes for logging/audit purposes."""
    return {
        "n_total": len(y),
        "n_pos": (y == 1).sum(),
        "n_neg": (y == 0).sum(),
        "pos_frac": pos_fraction(y),
        "scale_pos_weight": xgb_scale_pos_weight(y),
    }
