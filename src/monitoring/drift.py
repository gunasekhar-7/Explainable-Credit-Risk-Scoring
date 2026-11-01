"""Drift monitoring: Population Stability Index (PSI), KS drift, feature distribution tracking."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR

logger = logging.getLogger(__name__)

# Default thresholds (industry standard)
PSI_STABLE = 0.10
PSI_WARN = 0.25
KS_STABLE = 0.10
KS_WARN = 0.20


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index between two distributions.

    PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))

    Returns PSI value; interpret as:
      < 0.10  : stable (no drift)
      0.10-0.25: warning (moderate drift)
      > 0.25   : alert (significant drift)
    """
    # Quantile-based binning on expected to get equal-count bins
    _, bin_edges = pd.qcut(expected, q=bins, retbins=True, duplicates="drop")
    bin_edges = np.asarray(bin_edges, dtype=float)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    expected_binned = pd.cut(expected, bins=bin_edges, include_lowest=True, duplicates="drop")
    actual_binned = pd.cut(actual, bins=bin_edges, include_lowest=True, duplicates="drop")

    expected_counts = expected_binned.value_counts().sort_index()
    actual_counts = actual_binned.value_counts().sort_index()

    # Align indices
    expected_counts = expected_counts.reindex(expected_counts.index.union(actual_counts.index), fill_value=0)
    actual_counts = actual_counts.reindex(expected_counts.index.union(actual_counts.index), fill_value=0)

    expected_pct = expected_counts / max(len(expected), 1)
    actual_pct = actual_counts / max(len(actual), 1)

    # Avoid log(0) and division by zero
    expected_pct = expected_pct.clip(lower=1e-6)
    actual_pct = actual_pct.clip(lower=1e-6)

    psi_val = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return psi_val


def ks_drift(expected: np.ndarray, actual: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov drift statistic between two distributions.

    Returns KS value; interpret as:
      < 0.10  : stable
      0.10-0.20: warning
      > 0.20   : alert
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    all_vals = np.unique(np.concatenate([expected, actual]))

    ecdf_exp = np.searchsorted(np.sort(expected), all_vals, side="right") / max(len(expected), 1)
    ecdf_act = np.searchsorted(np.sort(actual), all_vals, side="right") / max(len(actual), 1)

    return float(np.max(np.abs(ecdf_exp - ecdf_act)))


def compute_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str] | None = None,
    psi_bins: int = 10,
) -> dict[str, dict[str, float]]:
    """
    Compute PSI and KS drift for each feature.

    Returns dict: {feature: {"psi": float, "ks": float, "status": "stable|warning|alert"}}
    """
    if features is None:
        features = [c for c in reference.columns if c in current.columns]

    results: dict[str, dict[str, float]] = {}
    for feat in features:
        if feat not in reference.columns or feat not in current.columns:
            continue
        ref_vals = pd.to_numeric(reference[feat], errors="coerce").dropna().values
        cur_vals = pd.to_numeric(current[feat], errors="coerce").dropna().values
        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue

        psi_val = psi(ref_vals, cur_vals, bins=psi_bins)
        ks_val = ks_drift(ref_vals, cur_vals)

        if psi_val > PSI_WARN or ks_val > KS_WARN:
            status = "alert"
        elif psi_val > PSI_STABLE or ks_val > KS_STABLE:
            status = "warning"
        else:
            status = "stable"

        results[feat] = {"psi": psi_val, "ks": ks_val, "status": status}

    return results


def compute_score_drift(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
) -> dict[str, float | str]:
    """
    Compute drift on model scores (probabilities or points).

    Returns {"psi": float, "ks": float, "status": str}
    """
    psi_val = psi(reference_scores, current_scores)
    ks_val = ks_drift(reference_scores, current_scores)

    if psi_val > PSI_WARN or ks_val > KS_WARN:
        status = "alert"
    elif psi_val > PSI_STABLE or ks_val > KS_STABLE:
        status = "warning"
    else:
        status = "stable"

    return {"psi": psi_val, "ks": ks_val, "status": status}


class DriftMonitor:
    """
    Persistent drift monitor that stores reference distributions and computes drift on new batches.

    Usage:
        monitor = DriftMonitor(reference_data=X_train, features=["EXT_SOURCE_1", "CHILDREN_RATIO"])
        monitor.save()
        # Later, on new batch:
        monitor = DriftMonitor.load()
        report = monitor.check(new_batch)
    """

    def __init__(
        self,
        reference_data: pd.DataFrame | None = None,
        features: list[str] | None = None,
        psi_bins: int = 10,
        reference_scores: np.ndarray | None = None,
    ) -> None:
        self.features = features or []
        self.psi_bins = psi_bins
        self.reference_data = reference_data.copy() if reference_data is not None else None
        self.reference_scores = reference_scores.copy() if reference_scores is not None else None

    def save(self, path: str | Path = PROCESSED_DIR / "drift_reference.json") -> None:
        """Persist reference data and configuration to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "features": self.features,
            "psi_bins": self.psi_bins,
        }
        if self.reference_data is not None:
            # Store as list of records (JSON-serializable)
            payload["reference_data"] = self.reference_data[self.features].to_dict("records")
        if self.reference_scores is not None:
            payload["reference_scores"] = self.reference_scores.tolist()

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: str | Path = PROCESSED_DIR / "drift_reference.json") -> DriftMonitor:
        """Load monitor from persisted JSON."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError("Reference not found")

        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)

        ref_data = None
        if "reference_data" in payload:
            ref_data = pd.DataFrame(payload["reference_data"])

        ref_scores = None
        if "reference_scores" in payload:
            ref_scores = np.array(payload["reference_scores"])

        return cls(
            reference_data=ref_data,
            features=payload.get("features", []),
            psi_bins=payload.get("psi_bins", 10),
            reference_scores=ref_scores,
        )

    def check(
        self,
        current_data: pd.DataFrame,
        current_scores: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Compute drift report against stored reference.

        Returns full report with per-feature and score drift.
        """
        if self.reference_data is None:
            raise RuntimeError("No reference data")

        feature_drift = compute_feature_drift(
            self.reference_data, current_data, self.features, self.psi_bins
        )

        report: dict[str, Any] = {
            "feature_drift": feature_drift,
            "overall_status": "stable",
        }

        # Aggregate status: worst of all features
        statuses = [v["status"] for v in feature_drift.values()]
        if "alert" in statuses:
            report["overall_status"] = "alert"
        elif "warning" in statuses:
            report["overall_status"] = "warning"

        if current_scores is not None and self.reference_scores is not None:
            score_drift = compute_score_drift(self.reference_scores, current_scores)
            report["score_drift"] = score_drift
            if score_drift["status"] == "alert":
                report["overall_status"] = "alert"
            elif score_drift["status"] == "warning" and report["overall_status"] == "stable":
                report["overall_status"] = "warning"

        logger.info(f"Drift check: overall_status={report['overall_status']}, features={len(feature_drift)}")
        return report

    def check_and_alert(
        self,
        current_data: pd.DataFrame,
        current_scores: np.ndarray | None = None,
        alert_callback: callable | None = None,
    ) -> dict[str, Any]:
        """Run check and invoke callback if status is warning or alert."""
        report = self.check(current_data, current_scores)
        if report["overall_status"] in ("warning", "alert") and alert_callback:
            alert_callback(report)
        return report