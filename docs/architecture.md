# Data Flow & Architecture

```
data/raw/application_train.csv
        |
        v
src/data/load.py  -->  src/data/cleaning.py  -->  src/data/imbalance.py
        |                                    |
        v                                    v
src/data/features.py (WoE/IV binning)  -->  stratified train/val/test split
        |
        +--> src/models/scorecard.py   (Logistic Regression on WoE features)
        |
        +--> src/models/xgboost.py     (XGBClassifier on raw features)
        |
        +--> src/evaluation/metrics.py (AUC, KS, Gini, calibration)
        |
        +--> src/explain/shap_explainer.py (TreeExplainer, local/waterfall)
        |
        +--> src/mlflow_tracking.py    (experiment logging to mlruns/)
        |
        +--> src/api/app.py            (FastAPI /predict, /health, /model-info)
```

## Components

- `src/config.py` — paths, target column (`TARGET`), ID column (`SK_ID_CURR`), seed, WoE bin count, IV thresholds.
- `src/data/load.py` — Kaggle download + local CSV load; Lending Club fallback documented.
- `src/data/cleaning.py` — missing-value strategy ("MISSING" category flag, median impute <40% missing, drop >60% missing), winsorize at 1st/99th percentile, type fixes.
- `src/data/imbalance.py` — class weights / `scale_pos_weight`; optional SMOTE toggle (not recommended for production).
- `src/data/features.py` — quantile binning with monotonicity check, rare-category grouping, WoE/IV computation, maps persisted to JSON.
- `src/models/scorecard.py` — Logistic Regression (`class_weight='balanced'`), scorecard scaling (points = offset + factor × log-odds), score() returns points + risk band + reason codes.
- `src/models/xgboost.py` — `XGBClassifier` with `scale_pos_weight`, early stopping, optional Platt/isotonic calibration.
- `src/evaluation/metrics.py` — AUC-ROC, KS statistic, Gini, calibration by decile, confusion matrix at business threshold.
- `src/explain/shap_explainer.py` — `shap.TreeExplainer`, global summary/bar/dependence plots, local waterfall per applicant.
- `src/mlflow_tracking.py` — local MLflow file store at `mlruns/`; logs params, metrics, model artifacts, WoE maps; tags dataset version/git commit/model_stage.
- `src/api/` — FastAPI app: `GET /health`, `POST /predict` (probability, scorecard points, risk band, top contributors), `POST /predict/batch`, `GET /model-info`.

## Regulatory notes

- No target-leakage features: only fields known at application time; post-default fields excluded and documented.
- Rate calibration preserved: class weights / `scale_pos_weight`, no SMOTE in production path; KS and AUC computed on held-out test set only.
- Adverse-action reason codes: top contributors by |SHAP| (XGBoost) or coefficient × WoE (scorecard), mapped to business language.
- Model governance: MLflow run tags (dataset version, git commit, model_stage candidate/challenger/champion).
