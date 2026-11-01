"""CLI entry point: python scripts/train.py

load -> clean -> domain features -> WoE/IV -> train scorecard + XGBoost ->
evaluate -> select champion (best KS, interpretability constraint) -> persist
artifacts + MLflow logging + champion.json manifest.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.model_selection import train_test_split

from src.config import (
    PROCESSED_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    TEST_SIZE,
    VAL_SIZE,
)
from src.data.cleaning import clean
from src.data.features import (
    add_domain_features,
    apply_woe_maps,
    build_woe_maps,
)
from src.data.imbalance import stratify_split_info
from src.data.load import load_application_train, split_xy
from src.evaluation.metrics import evaluate
from src.mlflow_tracking import RunLogger, init_tracking
from src.models.scorecard import Scorecard
from src.models.xgboost import XGBoostModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("train")

CHAMPION_MANIFEST = PROCESSED_DIR / "champion.json"


def train_and_evaluate(csv_path: str | None = None) -> dict:
    # ---- load ----
    df = load_application_train(csv_path)
    X_raw, y, _ids = split_xy(df)
    log.info(f"Loaded {len(df)} rows; target positive fraction: {(y==1).mean():.3f}")
    log.info(stratify_split_info(y))

    # ---- clean + domain features ----
    X_clean = clean(X_raw)
    X_clean = add_domain_features(X_clean)

    # ---- split (stratified 70/15/15) ----
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_clean, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    val_size_adj = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size_adj, stratify=y_trainval, random_state=RANDOM_SEED
    )

    # ---- WoE/IV binning on train (monotonicity enforced) ----
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]
    maps_path = PROCESSED_DIR / "woe_maps.json"
    maps_path.parent.mkdir(parents=True, exist_ok=True)
    woe_maps = build_woe_maps(
        X_train, y_train, numeric_cols, categorical_cols, maps_path=maps_path
    )

    # ---- persist background sample for SHAP precompute ----
    bg = X_train.sample(n=min(50, len(X_train)), random_state=RANDOM_SEED)
    bg.to_csv(PROCESSED_DIR / "background_sample.csv", index=False)

    # ---- feature selection for scorecard (handled inside scorecard.fit) ----
    X_train_sel = X_train
    X_test_sel = X_test

    X_train_woe = apply_woe_maps(X_train_sel, woe_maps)
    X_test_woe = apply_woe_maps(X_test_sel, woe_maps)

    # ---- Logistic Regression scorecard ----
    scorecard = Scorecard()
    scorecard.fit(X_train_woe, y_train, woe_maps)
    test_score = scorecard.score(X_test_woe)
    score_metrics = evaluate(y_test.values, test_score["default_probability"].values)
    if score_metrics["auc"] < 0.55:
        raise RuntimeError(  # noqa: TRY003
            f"Scorecard validation AUC {score_metrics['auc']:.4f} is below the 0.55 threshold. "
            "Relax monotonicity enforcement or revisit feature selection."
        )
    log.info(
        f"[Scorecard] AUC={score_metrics['auc']:.4f} KS={score_metrics['ks']:.4f} "
        f"Gini={score_metrics['gini']:.4f}"
    )

    # ---- XGBoost ----
    xgb = XGBoostModel(random_seed=RANDOM_SEED)
    xgb.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    xgb_proba = xgb.predict_proba(X_test)
    xgb_metrics = evaluate(y_test.values, xgb_proba)
    log.info(
        f"[XGBoost] AUC={xgb_metrics['auc']:.4f} KS={xgb_metrics['ks']:.4f} "
        f"Gini={xgb_metrics['gini']:.4f}"
    )

    # ---- D4: comparison table (AUC/KS/Gini/Brier per model) ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = RESULTS_DIR / "comparison.csv"
    with open(comparison_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["model", "auc", "ks", "gini", "brier"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "model": "scorecard",
                "auc": score_metrics["auc"],
                "ks": score_metrics["ks"],
                "gini": score_metrics["gini"],
                "brier": score_metrics["brier"],
            }
        )
        writer.writerow(
            {
                "model": "xgboost",
                "auc": xgb_metrics["auc"],
                "ks": xgb_metrics["ks"],
                "gini": xgb_metrics["gini"],
                "brier": xgb_metrics["brier"],
            }
        )
    log.info(f"[Comparison] wrote {comparison_path}")

    # ---- champion selection: best KS, interpretability constraint ----
    # Prefer scorecard if its KS is within 5 points of XGBoost (regulator-friendly).
    ks_gap = xgb_metrics["ks"] - score_metrics["ks"]
    champion = "scorecard" if ks_gap <= 0.05 else "xgboost"
    log.info(f"[Champion] {champion} (KS gap={ks_gap:.4f})")

    # ---- persist artifacts ----
    scorecard_path = PROCESSED_DIR / "scorecard"
    xgb_path = PROCESSED_DIR / "xgboost"
    scorecard.save(scorecard_path)
    xgb.save(xgb_path)

    # Champion metrics for /model-info (D6) + flat fields retained for compatibility.
    champion_metrics = score_metrics if champion == "scorecard" else xgb_metrics
    manifest = {
        "champion": champion,
        "version": "1.0.0",
        "metrics": {
            "auc": champion_metrics["auc"],
            "ks": champion_metrics["ks"],
            "gini": champion_metrics["gini"],
            "brier": champion_metrics["brier"],
        },
        "scorecard_auc": score_metrics["auc"],
        "scorecard_ks": score_metrics["ks"],
        "xgb_auc": xgb_metrics["auc"],
        "xgb_ks": xgb_metrics["ks"],
        "ks_gap": ks_gap,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "dataset": "synthetic_sample",
    }
    CHAMPION_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(CHAMPION_MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    # ---- MLflow logging ----
    init_tracking()
    stage_tag = "champion" if champion == "xgboost" else "challenger"
    with RunLogger(
        run_name="hybrid_training",
        tags={
            "dataset": "synthetic_sample",
            "champion": champion,
            "model_stage": stage_tag,
        },
    ) as run_logger:
        run_logger.log_params(
            {
                "model_types": "logistic_regression_scorecard,xgboost",
                "n_train": len(X_train),
                "n_test": len(X_test),
                "champion": champion,
                "scale_pos_weight": getattr(xgb, "scale_pos_weight", None),
            }
        )
        run_logger.log_metrics(
            {
                "scorecard_auc": score_metrics["auc"],
                "scorecard_ks": score_metrics["ks"],
                "xgb_auc": xgb_metrics["auc"],
                "xgb_ks": xgb_metrics["ks"],
                "champion_ks": max(score_metrics["ks"], xgb_metrics["ks"]),
            }
        )
        run_logger.log_artifact(maps_path, artifact_path="binning_maps")
        run_logger.log_artifact(CHAMPION_MANIFEST, artifact_path="manifest")
        run_logger.log_model_sklearn(scorecard.model, artifact_path="scorecard_model")
        run_logger.log_model_xgboost(xgb.model, artifact_path="xgb_model")

    return {
        "scorecard_metrics": score_metrics,
        "xgb_metrics": xgb_metrics,
        "champion": champion,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train credit risk pipeline")
    parser.add_argument("--csv", default=None, help="Path to application_train.csv")
    args = parser.parse_args()
    train_and_evaluate(args.csv)


if __name__ == "__main__":
    main()
