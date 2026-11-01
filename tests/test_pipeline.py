"""Smoke tests for the credit risk pipeline (load / train / predict)."""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tempfile

import numpy as np
import pandas as pd

from src.api.predict import InferenceEngine
from src.config import ID_COL, TARGET_COL
from src.data.cleaning import clean
from src.data.features import add_domain_features, apply_woe_maps, build_woe_maps
from src.data.imbalance import pos_fraction, xgb_scale_pos_weight
from src.data.load import load_application_train, split_xy
from src.evaluation.metrics import auc_roc, ks_statistic
from src.models.scorecard import Scorecard
from src.models.xgboost import XGBoostModel


def test_load_synthetic_sample() -> None:
    df = load_application_train()
    assert not df.empty
    assert TARGET_COL in df.columns


def test_split_xy() -> None:
    df = load_application_train()
    X, y, _ids = split_xy(df)
    assert y is not None
    assert TARGET_COL not in X.columns
    assert len(X) == len(y)


def test_cleaning_runs() -> None:
    df = load_application_train()
    X, _, _ = split_xy(df)
    cleaned = clean(X)
    assert not cleaned.isna().any().any()


def test_woe_iv_maps() -> None:
    df = load_application_train()
    X, y, _ = split_xy(df)
    X_clean = clean(X)
    numeric_cols = X_clean.select_dtypes(include="number").columns.tolist()[:5]
    categorical_cols = [c for c in X_clean.columns if c not in numeric_cols][:3]
    maps = build_woe_maps(X_clean, y, numeric_cols, categorical_cols)
    # Categorical features always produce a map.
    assert set(categorical_cols) <= set(maps)
    # A numeric feature is either mapped with usable structure, or dropped as
    # degenerate because its WoE could not be made monotonic within the bin
    # floor (see build_woe_maps). Both outcomes are valid; a shipped map is not.
    for col in numeric_cols:
        if col not in maps:
            continue
        spec = maps[col]
        assert spec["type"] == "numeric"
        assert spec["woe_map"], f"{col} shipped an empty WoE map"
        assert spec["monotonic"] is True, f"{col} shipped non-monotonic WoE"
    assert len(maps) <= len(numeric_cols) + len(categorical_cols)
    for spec in maps.values():
        assert "iv" in spec and "woe_map" in spec


def test_apply_woe_maps_returns_woe_frame() -> None:
    df = load_application_train()
    X, y, _ = split_xy(df)
    X_clean = clean(X)
    numeric_cols = X_clean.select_dtypes(include="number").columns.tolist()[:5]
    categorical_cols = [c for c in X_clean.columns if c not in numeric_cols][:3]
    maps = build_woe_maps(X_clean, y, numeric_cols, categorical_cols)
    X_woe = apply_woe_maps(X_clean, maps)
    assert X_woe.shape[1] == len(maps)


def test_woe_monotonicity_enforced() -> None:
    """Every numeric WoE map must be monotonic (plan requirement)."""
    df = load_application_train()
    X, y, _ = split_xy(df)
    X_clean = clean(X)
    X_clean = add_domain_features(X_clean)
    numeric_cols = X_clean.select_dtypes(include="number").columns.tolist()[:6]
    categorical_cols = [c for c in X_clean.columns if c not in numeric_cols][:3]
    maps = build_woe_maps(X_clean, y, numeric_cols, categorical_cols, enforce_monotonic=True)
    shipped = 0
    for col in numeric_cols:
        if col not in maps:
            # Degenerate binning: the feature is excluded instead of being
            # shipped with non-monotonic WoE.
            continue
        spec = maps[col]
        assert spec["monotonic"] is True, f"{col} WoE not monotonic"
        shipped += 1
    assert shipped > 0, "monotonicity enforcement dropped every numeric feature"


def test_leakage_audit() -> None:
    """Target and ID columns must not appear in the feature frame."""
    df = load_application_train()
    X, _y, _ids = split_xy(df)
    assert TARGET_COL not in X.columns
    assert ID_COL not in X.columns
    # No column name should contain post-default hints.
    banned = ["default", "days_last", "days_termination", "close_date"]
    for col in X.columns:
        low = col.lower()
        assert not any(b in low for b in banned), f"possible leakage feature: {col}"


def test_imbalance_metrics() -> None:
    df = load_application_train()
    _, y, _ = split_xy(df)
    pf = pos_fraction(y)
    assert 0 < pf < 1
    spw = xgb_scale_pos_weight(y)
    assert spw > 0


def test_scorecard_fit_and_score() -> None:
    df = load_application_train()
    X, y, _ = split_xy(df)
    X_clean = clean(X)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=0.2, stratify=y, random_state=42
    )
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]
    maps = build_woe_maps(X_train, y_train, numeric_cols, categorical_cols)
    iv_features = [c for c, s in maps.items() if s["iv"] >= 0.02]
    X_tr = X_train[iv_features]
    X_te = X_test[iv_features]
    X_tr_woe = apply_woe_maps(X_tr, maps)
    X_te_woe = apply_woe_maps(X_te, maps)

    scorecard = Scorecard()
    scorecard.fit(X_tr_woe, y_train, maps)
    scored = scorecard.score(X_te_woe)
    assert "default_probability" in scored.columns
    assert "scorecard_points" in scored.columns
    assert "risk_band" in scored.columns
    auc = auc_roc(y_test.values, scored["default_probability"].values)
    assert auc > 0.45  # weak baseline on tiny synthetic holdout


def test_xgboost_fit_and_predict() -> None:
    df = load_application_train()
    X, y, _ = split_xy(df)
    X_clean = clean(X)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=0.2, stratify=y, random_state=42
    )
    xgb = XGBoostModel(n_estimators=20, random_seed=42)
    xgb.fit(X_train, y_train)
    proba = xgb.predict_proba(X_test)
    assert len(proba) == len(X_test)
    auc = auc_roc(y_test.values, proba)
    assert auc > 0.5
    ks = ks_statistic(y_test.values, proba)
    assert 0 <= ks <= 1


def test_ks_bounds() -> None:
    y = pd.Series([0, 0, 1, 1])
    s = pd.Series([0.1, 0.2, 0.8, 0.9])
    ks = ks_statistic(y, s)
    assert 0 <= ks <= 1


def test_inference_engine_load_raises_on_missing_artifacts() -> None:
    """InferenceEngine.load() must raise RuntimeError when no model artifacts exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = InferenceEngine(artifacts_dir=tmpdir)
        try:
            engine.load()
        except RuntimeError as exc:
            assert "No model artifacts" in str(exc)
            return
    raise AssertionError("Expected RuntimeError when no artifacts exist")  # noqa: TRY003


def test_build_woe_maps_excludes_degenerate_features() -> None:
    """Degenerate numeric features must be excluded from the maps dict entirely."""
    import src.data.features as features

    df = load_application_train()
    X, y, _ = split_xy(df)
    X_clean = clean(X)
    X_clean = add_domain_features(X_clean)
    numeric_cols = X_clean.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X_clean.columns if c not in numeric_cols]

    original_enforce = features.enforce_monotonic_woe

    def force_degenerate(
        woe_df: pd.DataFrame,
        bin_edges: np.ndarray,
        min_bins: int = 3,
        iv_threshold: float = 0.0,
    ) -> tuple[pd.DataFrame, np.ndarray, bool]:
        woe_df2, edges2, _ = original_enforce(
            woe_df, bin_edges, min_bins=min_bins, iv_threshold=iv_threshold
        )
        return woe_df2, edges2, True

    with mock.patch.object(features, "woe_monotonic", return_value=False), mock.patch.object(
        features, "enforce_monotonic_woe", side_effect=force_degenerate
    ):
        maps = build_woe_maps(
            X_clean, y, numeric_cols, categorical_cols, enforce_monotonic=True
        )

    assert set(maps.keys()) == set(categorical_cols)
    for col in numeric_cols:
        assert col not in maps


def test_scorecard_score_nan_guard() -> None:
    """NaN WoE values must not corrupt contributor ranking or magnitudes."""
    df = load_application_train()
    X, y, _ = split_xy(df)
    X_clean = clean(X)
    X_clean = add_domain_features(X_clean)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, _y_test = train_test_split(
        X_clean, y, test_size=0.2, stratify=y, random_state=42
    )
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]
    maps = build_woe_maps(X_train, y_train, numeric_cols, categorical_cols)
    iv_features = [c for c, s in maps.items() if s["iv"] >= 0.02]
    iv_features = [
        f for f in iv_features if f in X_train.columns and f in numeric_cols
    ]
    X_tr = X_train[iv_features]
    X_te = X_test[iv_features]
    X_tr_woe = apply_woe_maps(X_tr, maps)
    X_te_woe = apply_woe_maps(X_te, maps)

    scorecard = Scorecard()
    scorecard.fit(X_tr_woe, y_train, maps)

    X_woe_nan = X_te_woe.copy()
    if len(scorecard.features) > 0:
        X_woe_nan.iloc[0, 0] = np.nan

    scored = scorecard.score(X_woe_nan)
    assert "top_contributors" in scored.columns
    for row in scored["top_contributors"]:
        for contrib in row:
            assert contrib["direction"] in ("increases_risk", "decreases_risk")
            assert contrib["magnitude"] == contrib["magnitude"]  # not NaN
            assert abs(contrib["magnitude"]) < float("inf")


def test_xgboost_scale_pos_weight_roundtrip() -> None:
    """scale_pos_weight must be initialized None, persisted, and restored on load."""
    fresh = XGBoostModel()
    assert fresh.scale_pos_weight is None

    df = load_application_train()
    X, y, _ = split_xy(df)
    X_clean = clean(X)
    from sklearn.model_selection import train_test_split

    X_train, _, y_train, _ = train_test_split(
        X_clean, y, test_size=0.2, stratify=y, random_state=42
    )
    xgb = XGBoostModel(n_estimators=10, random_seed=42)
    xgb.fit(X_train, y_train)
    spw = xgb.scale_pos_weight
    assert spw is not None and spw > 0

    with tempfile.TemporaryDirectory() as tmpdir:
        xgb.save(tmpdir)
        loaded = XGBoostModel.load(tmpdir)
        assert loaded.scale_pos_weight == spw


def test_ready_endpoint_state() -> None:
    """``/ready`` reports 503 before load and ready after the engine is loaded."""
    from fastapi import HTTPException

    import src.api.app as app_module

    app_module._engine = None
    try:
        app_module.ready()
        raise AssertionError("Expected HTTPException 503 before load")  # noqa: TRY003
    except HTTPException as exc:
        assert exc.status_code == 503

    fake_engine = types.SimpleNamespace(_loaded=True)
    app_module._engine = fake_engine
    assert app_module.ready() == {"status": "ready"}
    app_module._engine = None


def test_model_info_model_type_reflects_champion() -> None:
    """``/model-info`` model_type must reflect the actual champion, not a hardcoded string."""
    from unittest import mock

    from fastapi import HTTPException

    import src.api.app as app_module

    # Engine unloaded and get_engine() raises RuntimeError -> 503
    app_module._engine = None
    with mock.patch.object(
        app_module, "get_engine", side_effect=RuntimeError("No model artifacts loaded")
    ):
        try:
            app_module.model_info()
            raise AssertionError("Expected HTTPException 503 when engine unloaded")  # noqa: TRY003
        except HTTPException as exc:
            assert exc.status_code == 503
    app_module._engine = None

    # With an engine loaded, model_type reflects the actual champion.
    fake_engine = types.SimpleNamespace(
        _loaded=True,
        champion="scorecard",
        xgb=None,
        scorecard=None,
        model_features=lambda: ["EXT_SOURCE_1"],
        field_influence=lambda: (["EXT_SOURCE_1"], ["AMT_CREDIT"]),
    )
    app_module._engine = fake_engine
    with mock.patch.object(app_module, "_champion_metrics", return_value={}):
        info = app_module.model_info(identity="ip:test")
    assert info.model_type == "scorecard"
    assert info.features == ["EXT_SOURCE_1"]
    assert info.influencing_fields == ["EXT_SOURCE_1"]
    assert info.ignored_fields == ["AMT_CREDIT"]
    assert info.challenger == "xgboost"
    app_module._engine = None


def test_scorecard_empty_feature_guard() -> None:
    """Scorecard.fit() must raise RuntimeError when IV selection yields zero features."""
    import pandas as pd

    X = pd.DataFrame({"a": [0.1, 0.2, 0.3, 0.4, 0.5] * 4, "b": [1, 2, 3, 4, 5] * 4})
    y = pd.Series([0, 0, 1, 1, 0] * 4)
    maps = {"a": {"iv": 0.001}, "b": {"iv": 0.001}}  # all below IV_USELESS=0.02
    scorecard = Scorecard()
    try:
        scorecard.fit(X, y, maps)
        raise AssertionError("Expected RuntimeError for zero features")  # noqa: TRY003
    except RuntimeError as exc:
        assert "zero features" in str(exc).lower()


def test_checksum_sidecar_written_and_verified() -> None:
    """Model artifacts must carry a SHA-256 sidecar and load must verify it."""
    import tempfile
    from pathlib import Path

    from src.security import verify_checksum, write_checksum

    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "model.pkl"
        p.write_bytes(b"model-bytes")
        write_checksum(p)
        assert (Path(tmpdir) / "model.pkl.sha256").exists()
        assert verify_checksum(p) is True

        # Tamper -> verification fails
        (Path(tmpdir) / "model.pkl.sha256").write_text("deadbeef")
        assert verify_checksum(p) is False


def test_load_missing_artifact_raises_file_not_found() -> None:
    """Loading from a directory with no model artifact must raise FileNotFoundError."""
    import tempfile

    from src.models.scorecard import Scorecard

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            Scorecard.load(tmpdir)
            raise AssertionError("Expected FileNotFoundError for missing artifact")  # noqa: TRY003
        except FileNotFoundError:
            pass


def test_derived_features_are_computed_at_inference() -> None:
    """A ratio the model was trained on must be derived at serving time too.

    Training calls add_domain_features before binning. If inference skips it,
    CHILDREN_RATIO is permanently absent, scores identical applicants who differ
    only in family composition, and the deployed model is weaker than the one
    that was validated.
    """
    from src.data.features import add_domain_features

    frame = pd.DataFrame(
        [
            {"CNT_CHILDREN": 0, "CNT_FAM_MEMBERS": 2},
            {"CNT_CHILDREN": None, "CNT_FAM_MEMBERS": None},
            {"CNT_FAM_MEMBERS": 0},
        ]
    )
    derived = add_domain_features(frame)
    assert derived["CHILDREN_RATIO"].tolist()[0] == 0.0
    # Missing operands must degrade to NaN, never raise.
    assert pd.isna(derived["CHILDREN_RATIO"].iloc[1])
    assert pd.isna(derived["CHILDREN_RATIO"].iloc[2])

    from src.config import PROCESSED_DIR

    if not (PROCESSED_DIR / "champion.json").exists():
        print("      (skipped end-to-end check: no trained artifacts)")
        return

    engine = InferenceEngine().load()
    if "CHILDREN_RATIO" not in engine.model_features():
        print("      (champion does not use CHILDREN_RATIO; nothing to assert)")
        return

    base = {"EXT_SOURCE_1": 0.5, "OCCUPATION_TYPE": None, "NAME_INCOME_TYPE": None}
    low = engine.predict_one({**base, "CNT_CHILDREN": 0.0, "CNT_FAM_MEMBERS": 2.0})
    high = engine.predict_one({**base, "CNT_CHILDREN": 4.0, "CNT_FAM_MEMBERS": 6.0})
    assert low["default_probability"] != high["default_probability"], (
        "family composition did not move the score: CHILDREN_RATIO is inert",
    )


def test_drift_psi_ks() -> None:
    """PSI and KS drift metrics compute correctly on known distributions."""
    import numpy as np

    from src.monitoring.drift import ks_drift, psi

    # Identical distributions -> PSI ~ 0, KS = 0
    np.random.seed(42)
    ref = np.random.normal(0, 1, 1000)
    cur = np.random.normal(0, 1, 1000)
    assert psi(ref, cur) < 0.05
    assert ks_drift(ref, cur) < 0.05

    # Shifted distribution -> PSI > 0.1, KS > 0.1
    cur_shifted = np.random.normal(1, 1, 1000)
    assert psi(ref, cur_shifted) > 0.1
    assert ks_drift(ref, cur_shifted) > 0.1


def test_drift_monitor_save_load() -> None:
    """DriftMonitor saves and loads reference data correctly."""
    import tempfile

    import numpy as np
    import pandas as pd

    from src.monitoring.drift import DriftMonitor

    np.random.seed(42)
    ref_data = pd.DataFrame({
        "feature1": np.random.normal(0, 1, 100),
        "feature2": np.random.normal(5, 2, 100),
    })
    ref_scores = np.random.uniform(0, 1, 100)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "drift_ref.json"
        monitor = DriftMonitor(
            reference_data=ref_data,
            features=["feature1", "feature2"],
            reference_scores=ref_scores,
        )
        monitor.save(path)

        loaded = DriftMonitor.load(path)
        assert loaded.features == ["feature1", "feature2"]
        assert loaded.reference_data is not None
        assert len(loaded.reference_data) == 100
        assert loaded.reference_scores is not None
        assert len(loaded.reference_scores) == 100


def test_drift_monitor_check() -> None:
    """DriftMonitor.check computes drift report correctly."""
    import tempfile

    import numpy as np
    import pandas as pd

    from src.monitoring.drift import DriftMonitor

    np.random.seed(42)
    # Generate larger reference sample for more stable statistics
    ref_feature1 = np.random.normal(0, 1, 1000)
    ref_data = pd.DataFrame({"feature1": ref_feature1})
    ref_scores = np.random.uniform(0, 1, 1000)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "drift_ref.json"
        monitor = DriftMonitor(
            reference_data=ref_data,
            features=["feature1"],
            reference_scores=ref_scores,
        )
        monitor.save(path)

        # Test with identical data (should be stable - PSI=0, KS=0)
        report = monitor.check(ref_data.copy(), ref_scores.copy())
        assert report["overall_status"] == "stable"
        assert "feature_drift" in report
        assert "score_drift" in report
        assert report["feature_drift"]["feature1"]["status"] == "stable"
        assert report["score_drift"]["status"] == "stable"

        # Test with similar but different sample from same distribution (may be stable or warning)
        np.random.seed(123)
        current_data = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 1000),
        })
        current_scores = np.random.uniform(0, 1, 1000)
        report2 = monitor.check(current_data, current_scores)
        assert report2["overall_status"] in ("stable", "warning")
        assert "feature_drift" in report2
        assert "score_drift" in report2

        # Test with shifted distribution (should be alert)
        np.random.seed(456)
        current_data_shifted = pd.DataFrame({
            "feature1": np.random.normal(3, 1, 1000),
        })
        report_shifted = monitor.check(current_data_shifted, current_scores)
        assert report_shifted["overall_status"] == "alert"
        assert report_shifted["feature_drift"]["feature1"]["status"] == "alert"


def test_api_key_auth_and_rate_limit() -> None:
    """Open mode allows; a configured key is required; budgets reject excess."""
    import os

    from fastapi import HTTPException
    from starlette.requests import Request

    from src.api import auth

    def make_request() -> Request:
        return Request({"type": "http", "client": ("127.0.0.1", 5000), "headers": []})

    def expect_401(key: str | None) -> None:
        try:
            auth.require_api_key(make_request(), key)
        except HTTPException as exc:
            assert exc.status_code == 401
            return
        raise AssertionError(f"expected 401 for {key!r}")  # noqa: TRY003

    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("API_KEYS", None)
        assert auth.auth_mode() == auth.AUTH_MODE_OPEN
        assert auth.require_api_key(make_request(), None).startswith("ip:")

    with mock.patch.dict(os.environ, {"API_KEYS": "key-one, key-two"}, clear=False):
        assert auth.auth_mode() == auth.AUTH_MODE_API_KEY
        for missing in (None, "wrong", "key-one "):
            expect_401(missing)
        identity = auth.require_api_key(make_request(), "key-two")
        assert identity.startswith("key:")
        assert "key-two" not in identity  # the raw credential is never the identity

    # Budget: the first N are allowed, the next is rejected with Retry-After.
    limiter = auth.RateLimiter(limit=3, window_seconds=60.0)
    assert [limiter.check("ip:1.2.3.4") for _ in range(3)] == [None, None, None]
    retry = limiter.check("ip:1.2.3.4")
    assert retry is not None and retry > 0
    # A different identity has its own budget.
    assert limiter.check("ip:5.6.7.8") is None
    # Rows are charged as a block, so an oversized batch is refused outright.
    assert limiter.check("ip:9.9.9.9", cost=4) is not None


def test_batch_scoring_matches_single_prediction() -> None:
    """Batch scoring is vectorized, so it must agree with /predict row for row."""
    from src.config import PROCESSED_DIR

    if not (PROCESSED_DIR / "champion.json").exists():
        print("      (skipped: no trained artifacts in data/processed)")
        return

    engine = InferenceEngine().load()
    raw = load_application_train().drop(columns=[TARGET_COL], errors="ignore").head(6)
    rows = raw.astype(object).where(raw.notna(), None).to_dict("records")

    batch = engine.predict_many(rows)
    singles = [engine.predict_one(row) for row in rows]
    assert len(batch) == len(singles)
    for batched, single in zip(batch, singles, strict=True):
        # WoE inputs are bit-identical between the two paths, but the linear
        # model's BLAS reduction order depends on how many rows are in the
        # matrix, so the last bit of the log-odds can differ. Compare the
        # derived floats with a tolerance and everything else exactly.
        assert abs(batched["default_probability"] - single["default_probability"]) <= 1e-12
        assert batched["scorecard_points"] is None or (
            abs(batched["scorecard_points"] - single["scorecard_points"]) <= 1e-9
        )
        for key in (ID_COL, "champion", "model_version", "risk_band", "data_quality"):
            assert batched[key] == single[key], f"batch/{key} disagreed with single inference"
        assert batched["top_contributors"] == single["top_contributors"]
    assert [b.get(ID_COL) for b in batch] == [r.get(ID_COL) for r in rows], "row order lost"
    assert engine.predict_many([]) == []

    # The explainability contract holds on the vectorized path too.
    for result in batch:
        assert result["risk_band"] is not None
        assert all(c["magnitude"] > 0 for c in result["top_contributors"])
        assert all(c.get("bin_label") is not None for c in result["top_contributors"])

    # A caller who supplies nothing gets a score, and it says so.
    empty = engine.predict_one({})
    quality = empty["data_quality"]
    assert quality["fields_provided"] == 0
    assert quality["fields_imputed"] == quality["fields_total"]
    assert empty["default_probability"] >= 0.0
    assert empty["top_contributors"] == []


if __name__ == "__main__":
    import traceback

    tests = [
        test_load_synthetic_sample,
        test_split_xy,
        test_cleaning_runs,
        test_woe_iv_maps,
        test_apply_woe_maps_returns_woe_frame,
        test_woe_monotonicity_enforced,
        test_leakage_audit,
        test_imbalance_metrics,
        test_scorecard_fit_and_score,
        test_xgboost_fit_and_predict,
        test_ks_bounds,
        test_inference_engine_load_raises_on_missing_artifacts,
        test_build_woe_maps_excludes_degenerate_features,
        test_scorecard_score_nan_guard,
        test_xgboost_scale_pos_weight_roundtrip,
        test_ready_endpoint_state,
        test_model_info_model_type_reflects_champion,
        test_scorecard_empty_feature_guard,
        test_checksum_sidecar_written_and_verified,
        test_load_missing_artifact_raises_file_not_found,
        test_batch_scoring_matches_single_prediction,
        test_derived_features_are_computed_at_inference,
        test_api_key_auth_and_rate_limit,
        test_drift_psi_ks,
        test_drift_monitor_save_load,
        test_drift_monitor_check,
    ]
    passed = 0
    failed: list[str] = []
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
            passed += 1
        except Exception as exc:
            failed.append(test.__name__)
            print(f"FAIL {test.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} tests passed")
    # CI runs this file directly, so the exit status is the only failure signal.
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)
