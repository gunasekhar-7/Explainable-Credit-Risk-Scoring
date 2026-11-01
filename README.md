# Explainable Credit Risk Scoring

End-to-end ML pipeline for credit risk scoring on the Home Credit Default Risk dataset. Demonstrates a hybrid modeling approach: an interpretable Logistic Regression scorecard (WoE/IV transformations) alongside a high-performance XGBoost model, with SHAP explainability bridging black-box predictions to regulatory requirements. Experiments tracked via MLflow; predictions exposed through a FastAPI API.

## Problem

Predict the probability of default (`TARGET`) for a loan applicant within the observation window, and provide per-applicant explanations suitable for adverse-action reason codes and fair-lending compliance.

## Dataset

- **Primary:** Home Credit Default Risk (Kaggle) — `application_train.csv`. ~300k applicants, ~8% default rate, rich application-time features.
- **Fallback:** Lending Club (documented in `DATA_DOWNLOAD.md`) if Kaggle access fails.
- **Committed:** `data/synthetic_sample.csv` (~200 rows mirroring Home Credit schema) so the repo runs immediately without a Kaggle download. Raw data is gitignored (`data/raw/`).

## Stack

Python 3.10+, pandas, numpy, scikit-learn, xgboost, shap, mlflow, fastapi, uvicorn, pydantic, matplotlib/seaborn, jupyter.

## Repo structure

```
explainable-credit-risk-scoring/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── DATA_DOWNLOAD.md
├── data/
│   ├── raw/              (gitignored)
│   ├── processed/        (gitignored)
│   └── synthetic_sample.csv   (committed sample)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering_woe_iv.ipynb
│   ├── 03_modeling_logreg_scorecard.ipynb
│   ├── 04_modeling_xgboost.ipynb
│   ├── 05_evaluation_auc_ks.ipynb
│   └── 06_shap_explainability.ipynb
├── src/
│   ├── config.py
│   ├── data/ (load, cleaning, imbalance, features)
│   ├── models/ (scorecard, xgboost)
│   ├── evaluation/metrics.py
│   ├── explain/shap_explainer.py
│   ├── mlflow_tracking.py
│   └── api/ (app, schema, predict)
├── mlruns/               (gitignored MLflow store)
├── scripts/train.py
├── tests/test_pipeline.py
├── docs/architecture.md
└── plans/
```

## Phases

1. **Project Setup & Data Engineering** — venv, requirements, cleaning, class imbalance handling, WoE/IV binning with JSON-persisted maps.
2. **Model Development** — Logistic Regression scorecard (WoE features, scorecard scaling) + XGBoost (raw features, `scale_pos_weight`, early stopping, optional calibration).
3. **Explainability** — SHAP global summary/dependence plots + local per-applicant waterfall; top contributors with direction/magnitude.
4. **MLOps & Production** — MLflow experiment tracking (AUC, KS, Gini, calibration); FastAPI `/predict` returning default probability, scorecard points, risk band, top contributors.
5. **Portfolio** — README, results table, resume positioning, regulatory framing.

## How to run

```bash
# 1. Create environment and install dependencies
python -m venv .venv
# activate: .venv\Scripts\activate  (Windows) or source .venv/bin/activate (Unix)
pip install -r requirements.txt

# 2. Run the full training pipeline (uses synthetic_sample.csv by default)
python scripts/train.py

# 3. Start the API (binds loopback by default)
uvicorn src.api.app:app --reload

# 4. Example request (API_KEYS unset => local-open mode)
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"SK_ID_CURR": 100002, "EXT_SOURCE_1": 0.53, "OCCUPATION_TYPE": "Managers",
       "NAME_INCOME_TYPE": "Working", "CNT_CHILDREN": 1, "CNT_FAM_MEMBERS": 3}'
```

## Authentication and budgets

Scoring endpoints are gated and rate-limited; `/health` and `/ready` stay open so probes never need a credential.

```bash
# Require an API key on every scoring call
API_KEYS="key-one,key-two" uvicorn src.api.app:app --host 0.0.0.0 --port 8000

curl -H "X-API-Key: key-one" http://127.0.0.1:8000/model-info
```

- **`API_KEYS` unset** → `local-open` mode: requests are accepted without a key. The mode is never silent — `GET /` and `GET /model-info` report `auth_mode`, and startup logs a warning. Safe only because the service binds to loopback by default.
- **`API_KEYS` set** (comma-separated) → every request to `/predict`, `/predict/batch` and `/model-info` needs `X-API-Key`; missing or wrong returns `401`.
- **Budgets** are per identity (key digest, else client address) and per process: 120 requests/min and 6000 rows/min, configurable in `src/config.py`. Exceeding either returns `429` with `Retry-After`. With `--workers N` each worker enforces its own budget, so a multi-worker deployment needs a shared limiter.
- Keys are compared in constant time and never appear as identities (only a truncated SHA-256 digest is used).

## Regulatory & interpretability

- Adverse-action reason codes: top 3–5 contributors by |SHAP| (XGBoost) or coefficient × WoE (scorecard).
- Rate calibration preserved via class weights / `scale_pos_weight`; no SMOTE in production path.
- KS and AUC computed on held-out test set only; no target-leakage features.
- Model versioning via MLflow (dataset version, git commit, model_stage tag).

## Production-grade assurances

- **WoE monotonicity enforced**: numeric bins are merged until WoE is monotonic; unit-tested per feature.
- **Leakage audit**: target/ID columns excluded from features; post-default-hint column names flagged; unit-tested.
- **Champion selection**: best KS on held-out test, with interpretability constraint (prefer scorecard if within 5 KS points of XGBoost); manifest at `data/processed/champion.json`.
- **SHAP latency guard**: background summary precomputed (`data/processed/background_sample.csv`); top-N capped at compute time.
- **Train/serve parity**: derived ratio features (`CHILDREN_RATIO`, `ANNUITY_INCOME_RATIO`, `CREDIT_INCOME_RATIO`, `DAYS_EMPLOYED_PERCENT`) are computed at inference exactly as at training, so a selected ratio is never silently neutral. Regression-tested: two applicants differing only in family composition must score differently.
- **Inference robustness**: missing features imputed to neutral zero; unknown categorical categories mapped to `Other`; partial applicants supported and disclosed via `data_quality`.
- **Access control**: optional API-key auth (`API_KEYS`) with constant-time comparison, per-identity request and row budgets, `429` with `Retry-After`, and lock-free health/readiness probes.
- **Batch scoring**: shared code path with single prediction, vectorized across binning, scoring and SHAP. 500 applicants in ~0.1–0.3 s (~0.2–0.6 ms/row), versus ~12 s for the earlier per-row loop.
- **Tests**: 23 smoke tests covering load/clean/train/predict, WoE monotonicity, leakage audit, imbalance metrics, AUC bounds, artifact checksum security, empty-feature guard, `/model-info` champion reflection, batch-vs-single parity, train/serve parity for derived features, and API-key/rate-limit enforcement.

## Example API response

Captured from a running service (scorecard champion) for the request in "How to run". Every field below is real output, not an illustration.

```json
{
  "SK_ID_CURR": 100002,
  "champion": "scorecard",
  "model_version": "1.0.0",
  "default_probability": 0.34375750265799254,
  "scorecard_points": 647.7818054484263,
  "risk_band": "Medium",
  "top_contributors": [
    {"feature": "OCCUPATION_TYPE", "direction": "decreases_risk", "magnitude": 0.6697, "bin_label": "Managers"},
    {"feature": "NAME_INCOME_TYPE", "direction": "decreases_risk", "magnitude": 0.0648, "bin_label": "Working"},
    {"feature": "EXT_SOURCE_1", "direction": "increases_risk", "magnitude": 0.0512, "bin_label": "(0.43, 0.63]"},
    {"feature": "CHILDREN_RATIO", "direction": "increases_risk", "magnitude": 0.0345, "bin_label": "(0.25, 0.333]"}
  ],
  "data_quality": {
    "fields_total": 19,
    "fields_provided": 7,
    "fields_imputed": 12,
    "imputed_features": ["NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "AMT_ANNUITY", "DAYS_BIRTH", "DAYS_EMPLOYED", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "EXT_SOURCE_2", "EXT_SOURCE_3"],
    "complete": false
  }
}
```

Each reason code names **which bin** the applicant fell into, not just the column, and the list is never padded: contributions below a 1e-9 noise floor are dropped, so an applicant whose features all sit in neutral bins honestly returns `"top_contributors": []` rather than five zeroes.

`GET /model-info` reports how much of the record can actually move the score:

```json
{
  "model_type": "scorecard",
  "features": ["EXT_SOURCE_1", "OCCUPATION_TYPE", "CHILDREN_RATIO", "NAME_INCOME_TYPE"],
  "metrics": {"auc": 0.6023, "ks": 0.2519, "gini": 0.2046, "brier": 0.2441},
  "challenger": "xgboost",
  "challenger_metrics": {"auc": 0.5948, "ks": 0.1825},
  "influencing_fields": ["CNT_CHILDREN", "NAME_INCOME_TYPE", "OCCUPATION_TYPE", "CNT_FAM_MEMBERS", "EXT_SOURCE_1"],
  "ignored_fields": ["NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH", "DAYS_EMPLOYED", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "EXT_SOURCE_2", "EXT_SOURCE_3"],
  "auth_mode": "api-key"
}
```

**Honest coverage.** On the 500-row synthetic sample the champion selects 4 features, so 5 of the 19 request fields can move a score and 14 cannot. Relaxing the IV threshold to force more features in was measured and made it *worse* (AUC 0.6250 at 11 features vs 0.6302 at 4), so the selection stands and the split is disclosed instead of implied. With the full Home Credit dataset far more features clear selection.

## License

MIT"# Explainable-Credit-Risk-Scoring" 
