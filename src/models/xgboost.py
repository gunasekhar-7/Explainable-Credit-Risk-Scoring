"""XGBoost classifier pipeline on raw (non-WoE) features."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.config import XGB_EVAL_METRIC, XGB_OBJECTIVE
from src.data.imbalance import xgb_scale_pos_weight
from src.security import assert_artifact_trusted, write_checksum

logger = logging.getLogger(__name__)


class XGBoostModel:
    """XGBClassifier with scale_pos_weight and optional calibration."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_seed: int = 42,
        calibrate: bool = False,
    ) -> None:
        self.params = {
            "objective": XGB_OBJECTIVE,
            "eval_metric": XGB_EVAL_METRIC,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_seed,
            "n_jobs": 1,
        }
        self.calibrate = calibrate
        self.model: XGBClassifier | None = None
        self.calibrator = None
        self.features: list[str] = []
        self.scale_pos_weight: float | None = None
        self._encoders: dict[str, dict] = {}

    def _encode_object_cols(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        X = X.copy()
        for col in X.select_dtypes(include="object").columns:
            if fit:
                uniq = X[col].astype(str).unique()
                self._encoders[col] = {v: i for i, v in enumerate(uniq)}
            X[col] = X[col].astype(str).map(self._encoders.get(col, {})).fillna(-1).astype(float)
        return X

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> XGBoostModel:
        self.features = list(X.columns)
        X_enc = self._encode_object_cols(X, fit=True)
        scale_pos_weight = xgb_scale_pos_weight(y)
        self.scale_pos_weight = scale_pos_weight
        self.model = XGBClassifier(**self.params, scale_pos_weight=scale_pos_weight)

        fit_kwargs = {"verbose": False}
        if X_val is not None and y_val is not None:
            X_val_enc = self._encode_object_cols(X_val, fit=False)
            fit_kwargs["eval_set"] = [(X_val_enc, y_val)]  # type: ignore[assignment]
            # XGBoost 3.x Python fit() does not expose early_stopping_rounds /
            # callbacks; validation set is still recorded for monitoring.

        self.model.fit(X_enc, y, **fit_kwargs)

        if self.calibrate and X_val is not None and y_val is not None:
            from sklearn.calibration import CalibratedClassifierCV

            self.calibrator = CalibratedClassifierCV(
                self.model, method="isotonic", cv="prefit"
            )
            self.calibrator.fit(X_val, y_val)  # type: ignore[attr-defined]

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("XGBoost model not fitted. Call fit() first.")  # noqa: TRY003
        X_sel = X[self.features]
        X_sel = self._encode_object_cols(X_sel, fit=False)
        # Log novel-category rate for observability (unseen categories -> -1).
        for col in X_sel.select_dtypes(include="number").columns:
            if col in self._encoders:
                novel_frac = float((X_sel[col] == -1).mean())
                if novel_frac > 0:
                    logger.warning(
                        "Novel/unmapped categories in inference column %s: %.1f%% of rows",
                        col,
                        novel_frac * 100,
                    )
        if self.calibrator is not None:
            return self.calibrator.predict_proba(X_sel)[:, 1]
        return self.model.predict_proba(X_sel)[:, 1]

    # ----------------------------------------------------------------- persistence
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, directory / "xgboost_model.pkl")
        write_checksum(directory / "xgboost_model.pkl")
        payload = {
            "features": self.features,
            "encoders": self._encoders,
            "params": self.params,
            "calibrate": self.calibrate,
            "scale_pos_weight": self.scale_pos_weight,
        }
        with open(directory / "xgboost_meta.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> XGBoostModel:
        directory = Path(directory)
        model_path = directory / "xgboost_model.pkl"
        assert_artifact_trusted(model_path)
        with open(directory / "xgboost_meta.json", encoding="utf-8") as fh:
            payload = json.load(fh)
        p = payload["params"]
        obj = cls(
            n_estimators=p.get("n_estimators", 500),
            max_depth=p.get("max_depth", 6),
            learning_rate=p.get("learning_rate", 0.05),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.8),
            random_seed=p.get("random_state", 42),
        )
        obj.calibrate = payload["calibrate"]
        obj.features = payload["features"]
        obj._encoders = payload.get("encoders", {})
        obj.scale_pos_weight = payload.get("scale_pos_weight")
        obj.model = joblib.load(model_path)
        return obj
