"""Inference assembly: load champion model + binning maps, predict, explain."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.api.schema import CreditApplication
from src.config import PROCESSED_DIR
from src.data.features import add_domain_features, apply_woe_maps_with_labels
from src.models.scorecard import CONTRIBUTOR_EPS, Scorecard, risk_band_from_probability
from src.models.xgboost import XGBoostModel

if TYPE_CHECKING:
    from src.explain.shap_explainer import ShapExplainer

_ARTIFACTS_DIR = PROCESSED_DIR  # persisted models & binning maps live here

# The applicant identifier is passed through, not modelled, so it is excluded
# from the completeness accounting.
_MODEL_INPUT_FIELDS: tuple[str, ...] = tuple(
    name for name in CreditApplication.model_fields if name != "SK_ID_CURR"
)

# Model features computed from request fields rather than sent directly.
_DERIVED_INPUTS: dict[str, tuple[str, ...]] = {
    "ANNUITY_INCOME_RATIO": ("AMT_ANNUITY", "AMT_INCOME_TOTAL"),
    "CREDIT_INCOME_RATIO": ("AMT_CREDIT", "AMT_INCOME_TOTAL"),
    "DAYS_EMPLOYED_PERCENT": ("DAYS_EMPLOYED", "DAYS_BIRTH"),
    "CHILDREN_RATIO": ("CNT_CHILDREN", "CNT_FAM_MEMBERS"),
}


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return False


def _data_quality(applicant: dict) -> dict:
    """Report how much of the record the caller actually supplied."""
    imputed = [f for f in _MODEL_INPUT_FIELDS if _is_missing(applicant.get(f))]
    return {
        "fields_total": len(_MODEL_INPUT_FIELDS),
        "fields_provided": len(_MODEL_INPUT_FIELDS) - len(imputed),
        "fields_imputed": len(imputed),
        "imputed_features": imputed,
        "complete": not imputed,
    }


class InferenceEngine:
    """Loads champion artifacts and produces /predict responses."""

    def __init__(self, artifacts_dir: str | Path = _ARTIFACTS_DIR) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.scorecard: Scorecard | None = None
        self.xgb: XGBoostModel | None = None
        self.explainer: ShapExplainer | None = None
        self.woe_maps: dict = {}
        self.xgb_features: list[str] = []
        self.champion: str = "xgboost"
        self.model_version: str = "1.0.0"
        self.background: pd.DataFrame | None = None
        self._loaded = False

    def load(self) -> InferenceEngine:
        manifest_path = self.artifacts_dir / "champion.json"
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as fh:
                manifest = json.load(fh)
            self.champion = manifest.get("champion", "xgboost")
            self.model_version = manifest.get("version", "1.0.0")

        scorecard_path = self.artifacts_dir / "scorecard"
        xgb_path = self.artifacts_dir / "xgboost"
        maps_path = self.artifacts_dir / "woe_maps.json"
        background_path = self.artifacts_dir / "background_sample.csv"

        # P3: lazy-load only the champion model at startup.
        # When champion is scorecard, skip SHAP explainer construction entirely.
        # When champion is xgboost, skip scorecard loading.
        if self.champion == "scorecard" and scorecard_path.exists():
            self.scorecard = Scorecard.load(scorecard_path)
        elif self.champion == "xgboost" and xgb_path.exists():
            self.xgb = XGBoostModel.load(xgb_path)
            self.xgb_features = self.xgb.features
            from src.explain.shap_explainer import ShapExplainer

            self.explainer = ShapExplainer(
                self.xgb.model, feature_names=self.xgb_features
            )
            if background_path.exists():
                self.background = pd.read_csv(background_path)
                X_bg = (
                    self.background[self.xgb_features].copy()
                    if self.xgb_features
                    else self.background.copy()
                )
                for col, mapping in self.xgb._encoders.items():
                    if col in X_bg.columns:
                        X_bg[col] = X_bg[col].astype(str).map(mapping).fillna(-1)
                X_bg = X_bg.apply(pd.to_numeric, errors="coerce").fillna(0.0)
                self.explainer.set_background(X_bg)

        if maps_path.exists():
            with open(maps_path, encoding="utf-8") as fh:
                self.woe_maps = json.load(fh)

        # P2: fail fast when no model artifacts are present rather than
        # silently defaulting to probability 0.0.
        if self.scorecard is None and self.xgb is None:
            raise RuntimeError("No model artifacts loaded; run scripts/train.py first")  # noqa: TRY003
        # A scorecard without its binning maps cannot score anyone. Without this
        # guard predict_one() would skip the scorecard branch and report a
        # 0.0 default probability -- i.e. approve every applicant at zero risk.
        if self.scorecard is not None and not self.woe_maps:
            raise RuntimeError(  # noqa: TRY003
                f"Scorecard champion requires {maps_path.name}; refusing to serve "
                "predictions without binning maps"
            )

        self._loaded = True
        return self

    def model_features(self) -> list[str]:
        """The feature names the loaded champion actually scores on."""
        if self.xgb is not None:
            return list(self.xgb_features)
        if self.scorecard is not None:
            return list(self.scorecard.features)
        return []

    def field_influence(self) -> tuple[list[str], list[str]]:
        """Split request fields into those that can move the score and those that cannot.

        A field influences the score if the model uses it directly, or if it
        feeds a derived ratio the model uses. Everything else is accepted for
        schema stability and reporting, but cannot change a prediction.
        """
        used: set[str] = set()
        for feature in self.model_features():
            if feature in _DERIVED_INPUTS:
                used.update(_DERIVED_INPUTS[feature])
            elif feature in _MODEL_INPUT_FIELDS:
                used.add(feature)
        influencing = [f for f in _MODEL_INPUT_FIELDS if f in used]
        ignored = [f for f in _MODEL_INPUT_FIELDS if f not in used]
        return influencing, ignored

    @staticmethod
    def _annotate_labels(contributors: list[dict], raw_row, model_row) -> list[dict]:
        """Attach the value each tree-model contributor refers to.

        ``local_explanation`` already sorts by magnitude, so the first contributor
        below the noise floor ends the list -- nothing after it is a reason.
        """
        annotated: list[dict] = []
        for contrib in contributors:
            if float(contrib["magnitude"]) < CONTRIBUTOR_EPS:
                break
            feature = contrib["feature"]
            label = raw_row.get(feature) if raw_row is not None else None
            if _is_missing(label):
                # Engineered or imputed input: show what the model actually saw.
                model_value = model_row.get(feature) if model_row is not None else None
                label = "n/a" if model_value is None else f"{float(model_value):.6g}"
            annotated.append({**contrib, "bin_label": str(label)})
        return annotated[:5]

    def predict_one(self, applicant: dict) -> dict:
        """Run inference for a single applicant record."""
        return self.predict_many([applicant])[0]

    def predict_many(self, applicants: list[dict]) -> list[dict]:
        """Run inference for a batch of applicants in one vectorized pass.

        Every stage -- binning, scorecard scoring, XGBoost scoring, SHAP -- is
        executed once over the whole frame instead of once per applicant. Rows
        are assembled back into per-applicant responses preserving input order.
        """
        if not applicants:
            return []
        if not self._loaded:
            self.load()
        # Training derives ratio features before binning, so serving must too:
        # otherwise a selected ratio (CHILDREN_RATIO) is always missing here and
        # silently contributes neutral WoE no matter what the caller supplies.
        X = add_domain_features(pd.DataFrame(applicants))

        # ---- Scorecard path (WoE features) ----
        scorecard_probas: list[float] | None = None
        scorecard_points: list[float | None] | None = None
        scorecard_bands: list[str] | None = None
        scorecard_contributors: list[list[dict]] | None = None

        if self.scorecard is not None and self.woe_maps:
            X_woe, woe_labels = apply_woe_maps_with_labels(X, self.woe_maps)
            scored = self.scorecard.score(X_woe, bin_labels=woe_labels)
            scorecard_probas = scored["default_probability"].tolist()
            scorecard_points = scored["scorecard_points"].tolist()
            scorecard_bands = scored["risk_band"].tolist()
            scorecard_contributors = scored["top_contributors"].tolist()

        # ---- XGBoost path (raw features) for SHAP explanation ----
        xgb_probas: list[float] | None = None
        shap_contributors: list[list[dict]] | None = None

        if self.xgb is not None and self.explainer is not None:
            if self.xgb_features:
                X_model = X.reindex(columns=self.xgb_features)
                for col, mapping in self.xgb._encoders.items():
                    if col in X_model.columns:
                        X_model[col] = X_model[col].astype(str).map(mapping).fillna(-1)
                X_model = X_model.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            else:
                X_model = X.copy()
            xgb_probas = [float(p) for p in self.xgb.predict_proba(X_model)]
            shap_contributors = [
                self._annotate_labels(
                    explanation["top_contributors"], X.iloc[i], X_model.iloc[i]
                )
                for i, explanation in enumerate(
                    self.explainer.batch_explanations(X_model, top_n=5)
                )
            ]

        results: list[dict] = []
        for i, applicant in enumerate(applicants):
            # ---- Route probability + explanation through the champion model ----
            if self.champion == "scorecard" and scorecard_probas is not None:
                default_probability = scorecard_probas[i]
                top_contributors = scorecard_contributors[i] if scorecard_contributors else []
            elif self.champion == "xgboost" and xgb_probas is not None:
                default_probability = xgb_probas[i]
                top_contributors = shap_contributors[i] if shap_contributors else []
            elif scorecard_probas is not None:
                default_probability = scorecard_probas[i]
                top_contributors = scorecard_contributors[i] if scorecard_contributors else []
            elif xgb_probas is not None:
                default_probability = xgb_probas[i]
                top_contributors = shap_contributors[i] if shap_contributors else []
            else:
                # Never invent a probability: a missing 0.0 reads as "no risk".
                raise RuntimeError(  # noqa: TRY003
                    "No champion model produced a score for this applicant"
                )

            # The scorecard bands on points; a champion without points (XGBoost) is
            # banded on the same scale via its probability, so risk_band is never null.
            if scorecard_bands is not None:
                risk_band = scorecard_bands[i]
            else:
                thresholds = self.scorecard.risk_band_thresholds if self.scorecard else None
                risk_band = risk_band_from_probability(default_probability, thresholds)

            results.append(
                {
                    "SK_ID_CURR": applicant.get("SK_ID_CURR"),
                    "champion": self.champion,
                    "model_version": self.model_version,
                    "default_probability": default_probability,
                    "scorecard_points": scorecard_points[i] if scorecard_points else None,
                    "risk_band": risk_band,
                    "top_contributors": top_contributors[:5],
                    "data_quality": _data_quality(applicant),
                }
            )
        return results
