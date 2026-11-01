"""SHAP explainability for XGBoost: global plots + local per-applicant explanations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap


class ShapExplainer:
    """Wrapper around shap.TreeExplainer for XGBoost models."""

    def __init__(
        self,
        model,
        feature_names: list[str] | None = None,
        background: pd.DataFrame | None = None,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.background = background
        self._explainer = None
        self._background_shap: np.ndarray | None = None

    @property
    def explainer(self):
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    def set_background(self, background: pd.DataFrame) -> ShapExplainer:
        """Store a background dataset and precompute its SHAP values for speed."""
        self.background = background.copy()
        self._background_shap = np.asarray(self.explainer.shap_values(self.background))
        return self

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.explainer.shap_values(X))

    def expected_value(self) -> float:
        ev = self.explainer.expected_value
        if isinstance(ev, list):
            return float(ev[1])
        return float(ev)

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Mean |SHAP| per feature (global feature importance)."""
        shap_vals = self.shap_values(X)
        mean_abs = np.abs(shap_vals).mean(axis=0)
        features = self.feature_names or [f"f{i}" for i in range(shap_vals.shape[1])]
        return pd.DataFrame(  # type: ignore[no-any-return]
            {"feature": features, "mean_abs_shap": mean_abs}
        ).sort_values("mean_abs_shap", ascending=False)

    def local_explanation(
        self, X: pd.DataFrame, index: int, top_n: int = 5
    ) -> dict:
        """Top contributors for a single applicant with direction and magnitude.

        SHAP is computed for the single row only (latency-safe); top-N cap applied
        at compute time per the production latency guard.
        """
        row = X.iloc[[index]]
        shap_vals = self.shap_values(row)[0]
        features = self.feature_names or [f"f{i}" for i in range(len(shap_vals))]
        base = self.expected_value()
        contribs = [
            {
                "feature": feat,
                "shap_value": float(val),
                "direction": "increases_risk" if val > 0 else "decreases_risk",
                "magnitude": abs(float(val)),
            }
            for feat, val in zip(features, shap_vals, strict=True)
        ]
        contribs.sort(key=lambda c: c["magnitude"], reverse=True)
        return {
            "index": int(index),
            "base_value": base,
            "shap_sum": float(shap_vals.sum()),
            "top_contributors": contribs[:top_n],
        }

    def batch_explanations(self, X: pd.DataFrame, top_n: int = 5) -> list[dict]:
        """Top contributors for every row in X, top-N capped per row.

        SHAP is evaluated once for the whole frame. Calling
        :meth:`local_explanation` per row re-enters the tree explainer for every
        applicant, which dominated batch latency.
        """
        if len(X) == 0:
            return []
        shap_vals = self.shap_values(X)
        features = self.feature_names or [f"f{i}" for i in range(shap_vals.shape[1])]
        base = self.expected_value()
        explanations: list[dict] = []
        for i in range(len(X)):
            row_vals = shap_vals[i]
            order = np.argsort(-np.abs(row_vals))[:top_n]
            explanations.append(
                {
                    "index": i,
                    "base_value": base,
                    "shap_sum": float(row_vals.sum()),
                    "top_contributors": [
                        {
                            "feature": features[j],
                            "shap_value": float(row_vals[j]),
                            "direction": (
                                "increases_risk" if row_vals[j] > 0 else "decreases_risk"
                            ),
                            "magnitude": abs(float(row_vals[j])),
                        }
                        for j in order
                    ],
                }
            )
        return explanations
