"""Logistic Regression scorecard on WoE features with scorecard scaling."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import (
    CORR_DROP_THRESHOLD,
    IV_USELESS,
    LOGREG_CLASS_WEIGHT,
    LOGREG_MAX_ITER,
    LOGREG_PENALTY,
    LOGREG_SOLVER,
    RISK_BAND_THRESHOLDS,
    SCORE_PDO,
    SCORE_REF_ODDS,
    SCORE_REF_POINTS,
)
from src.data.features import drop_correlated, select_features_by_iv
from src.security import assert_artifact_trusted, write_checksum

logger = logging.getLogger(__name__)

# A contribution this small is indistinguishable from the neutral 0.0 WoE that
# missing features receive, so it is not a reason for anything and is dropped
# from the explanation rather than padded in to fill a fixed-length list.
CONTRIBUTOR_EPS = 1e-9


def band_from_points(points: float, thresholds: list[float] | None = None) -> str:
    """Map scorecard points onto a risk band."""
    limits = thresholds if thresholds is not None else RISK_BAND_THRESHOLDS
    if points < limits[0]:
        return "High"
    if points < limits[1]:
        return "Medium"
    if points < limits[2]:
        return "Low-Medium"
    return "Low"


def scorecard_scaling() -> tuple[float, float]:
    """Return ``(offset, factor)`` for points = offset + factor * log-odds."""
    factor = SCORE_PDO / np.log(2.0)
    offset = SCORE_REF_POINTS - factor * np.log(SCORE_REF_ODDS)
    return float(offset), float(factor)


def risk_band_from_probability(
    probability: float, thresholds: list[float] | None = None
) -> str:
    """Map a default probability onto the scorecard's points-based risk bands.

    The bands are defined in points, so the cut points are converted into
    probability space with the same scaling the scorecard uses. That keeps every
    champion on one scale: an XGBoost applicant is banded as the score it would
    have earned, instead of reporting no band at all.
    """
    offset, factor = scorecard_scaling()
    p = float(min(max(probability, 1e-12), 1 - 1e-12))
    points = offset + factor * np.log(p / (1 - p))
    return band_from_points(points, thresholds)


class Scorecard:
    """WoE-based Logistic Regression scorecard.

    score() returns default probability, scorecard points, risk band, and
    top contributing features (coefficient x WoE) per applicant.
    """

    def __init__(self) -> None:
        self.model: LogisticRegression | None = None
        self.features: list[str] = []
        self.woe_maps: dict = {}
        self.offset: float = 0.0
        self.factor: float = 0.0
        self.risk_band_thresholds: list[float] = []

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        X_woe: pd.DataFrame,
        y: pd.Series,
        woe_maps: dict,
        min_iv: float = IV_USELESS,
        risk_band_thresholds: list[float] | None = None,
    ) -> Scorecard:
        # Feature selection: IV threshold + correlation pruning
        iv_features = select_features_by_iv(woe_maps, min_iv=min_iv)
        iv_features = [f for f in iv_features if f in X_woe.columns]
        X_sel = X_woe[iv_features]
        X_sel, _dropped = drop_correlated(X_sel, threshold=CORR_DROP_THRESHOLD)
        self.features = list(X_sel.columns)
        if not self.features:
            raise RuntimeError(  # noqa: TRY003
                "Scorecard fit produced zero features. Relax IV_USELESS "
                f"({IV_USELESS}) or revisit feature selection / binning."
            )

        self.model = LogisticRegression(
            penalty=LOGREG_PENALTY,
            class_weight=LOGREG_CLASS_WEIGHT,
            solver=LOGREG_SOLVER,
            max_iter=LOGREG_MAX_ITER,
        )
        self.model.fit(X_sel, y)

        self.woe_maps = woe_maps
        self.risk_band_thresholds = (
            risk_band_thresholds if risk_band_thresholds is not None
            else RISK_BAND_THRESHOLDS  # type: ignore[assignment]
        )

        # Scorecard scaling: points = offset + factor * log_odds
        # factor = PDO / ln(2); offset = ref_points - factor * ln(ref_odds)
        self.offset, self.factor = scorecard_scaling()
        return self

    # ----------------------------------------------------------------- predict
    def predict_proba(self, X_woe: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Scorecard not fitted. Call fit() first.")  # noqa: TRY003
        X_sel = X_woe[self.features]
        return self.model.predict_proba(X_sel)[:, 1]  # type: ignore[no-any-return]

    def _log_odds(self, X_woe: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Scorecard not fitted. Call fit() first.")  # noqa: TRY003
        X_sel = X_woe[self.features]
        return self.model.decision_function(X_sel)  # type: ignore[no-any-return]

    def score(
        self,
        X_woe: pd.DataFrame,
        bin_labels: pd.DataFrame | None = None,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """Return probability, points, risk band, and top contributors per row.

        Contributors are ranked by |coefficient x WoE| and stopped at `top_n` or
        at the first contribution below :data:`CONTRIBUTOR_EPS` -- whichever
        comes first. The list is never padded: a short list is the honest answer
        for an applicant whose features all sit in neutral bins.

        When `bin_labels` (from ``apply_woe_maps_with_labels``) is supplied, each
        contributor carries the bin the applicant fell into.
        """
        proba = self.predict_proba(X_woe)
        log_odds = self._log_odds(X_woe)
        points = self.offset + self.factor * log_odds
        risk_bands = [band_from_points(p, self.risk_band_thresholds) for p in points]

        # Vectorized reason codes: coefficient * WoE contribution per feature.
        X_sel = X_woe[self.features]
        contribs = X_sel.values * self.model.coef_[0]  # type: ignore[union-attr]  # (n_rows, n_features)
        contribs = np.nan_to_num(contribs, nan=0.0)  # guard against NaN WoE
        rankings = np.argsort(-np.abs(contribs), axis=1)  # descending magnitude

        label_rows = (
            [bin_labels.iloc[i] for i in range(len(X_sel))]
            if bin_labels is not None
            else [None] * len(X_sel)
        )

        contributors = []
        for i in range(len(X_sel)):
            top = []
            for f_idx in rankings[i]:
                v = float(contribs[i, f_idx])
                if abs(v) < CONTRIBUTOR_EPS:
                    # Ranked by magnitude, so everything after this is smaller.
                    break
                feature = self.features[f_idx]
                entry: dict = {
                    "feature": feature,
                    "direction": "increases_risk" if v > 0 else "decreases_risk",
                    "magnitude": abs(v),
                }
                labels = label_rows[i]
                if labels is not None and feature in labels.index:
                    label = labels[feature]
                    entry["bin_label"] = None if pd.isna(label) else str(label)
                top.append(entry)
                if len(top) >= top_n:
                    break
            contributors.append(top)

        return pd.DataFrame(  # type: ignore[no-any-return]
            {
                "default_probability": proba,
                "scorecard_points": points,
                "risk_band": risk_bands,
                "top_contributors": contributors,
            },
            index=X_woe.index,
        )

    # ----------------------------------------------------------------- persistence
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, directory / "scorecard_model.pkl")
        write_checksum(directory / "scorecard_model.pkl")
        payload = {
            "features": self.features,
            "woe_maps": self.woe_maps,
            "offset": self.offset,
            "factor": self.factor,
            "risk_band_thresholds": self.risk_band_thresholds,
        }
        with open(directory / "scorecard_meta.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> Scorecard:
        directory = Path(directory)
        model_path = directory / "scorecard_model.pkl"
        assert_artifact_trusted(model_path)
        obj = cls()
        obj.model = joblib.load(model_path)
        with open(directory / "scorecard_meta.json", encoding="utf-8") as fh:
            payload = json.load(fh)
        obj.features = payload["features"]
        obj.woe_maps = payload["woe_maps"]
        obj.offset = payload["offset"]
        obj.factor = payload["factor"]
        obj.risk_band_thresholds = payload["risk_band_thresholds"]
        return obj
