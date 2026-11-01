# Explainable Credit Risk Scoring — Technical Roadmap

**Goal:** End-to-end ML project for a professional portfolio demonstrating fintech/banking domain expertise and technical proficiency in both interpretable (scorecard) and high-performance (XGBoost) modeling, with an explainability layer bridging black-box predictions to regulatory requirements.

**Primary dataset:** Home Credit Default Risk (Kaggle). Richer and more portfolio-worthy than Lending Club. **Fallback:** Lending Club (cleaner, smaller, easier to host on GitHub if Home Credit's size/restrictions are a concern). Decision resolved: start with Home Credit; document Lending Club as a documented alternative in README.

**Target:** binary default flag (`TARGET` in Home Credit) — probability of default within the observation window.

**Stack:** Python 3.10+, pandas, numpy, scikit-learn, xgboost, shap, mlflow, fastapi, uvicorn, pydantic, matplotlib/seaborn, jupyter. Environment via `requirements.txt` + `python -m venv` (or conda).

---

## Repo structure (create these directories/files)

```
explainable-credit-risk-scoring/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── data/                       # gitignored (raw data not committed)
│   ├── raw/
│   ├── processed/
│   └── synthetic_sample.csv    # small committed sample so repo is runnable without Kaggle download
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering_woe_iv.ipynb
│   ├── 03_modeling_logreg_scorecard.ipynb
│   ├── 04_modeling_xgboost.ipynb
│   ├── 05_evaluation_auc_ks.ipynb
│   └── 06_shap_explainability.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py               # paths, constants, model hyperparameters
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load.py             # Kaggle download / local CSV load
│   │   ├── cleaning.py         # missing-value strategy, outlier caps, type fixes
│   │   ├── imbalance.py        # class weights / scale_pos_weight / SMOTE toggle
│   │   └── features.py         # WoE/IV transforms, binning, credit-specific features
│   ├── models/
│   │   ├── __init__.py
│   │   ├── scorecard.py        # Logistic Regression on WoE features + scorecard scaling
│   │   └── xgboost.py          # XGBClassifier pipeline
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py          # AUC, KS, calibration, confusion matrix at business thresholds
│   ├── explain/
│   │   ├── __init__.py
│   │   └── shap_explainer.py   # SHAP TreeExplainer, local explanation per applicant
│   ├── mlflow_tracking.py      # experiment logging wrapper
│   └── api/
│       ├── __init__.py
│       ├── app.py              # FastAPI app
│       ├── schema.py           # Pydantic request/response models
│       └── predict.py          # inference + explanation assembly
├── mlruns/                     # gitignored (MLflow local store)
├── tests/
│   └── test_pipeline.py        # smoke tests for load/train/predict
├── scripts/
│   └── train.py                # CLI entry: python scripts/train.py
├── docs/
│   └── architecture.md         # data flow diagram + regulatory notes
└── plans/
    └── 1793380764261-explainable-credit-risk-scoring-roadmap.md
```

---

## Phase 1 — Project Setup & Data Engineering

### 1.1 Setup
- Create venv, `requirements.txt` pinned versions (xgboost, shap, mlflow, fastapi, scikit-learn, pandas).
- `.gitignore`: `data/raw/`, `mlruns/`, `.ipynb_checkpoints`, `__pycache__`, `.env`, `*.pkl`.
- Commit a small `data/synthetic_sample.csv` (100–500 rows mirroring Home Credit schema) so the repo is runnable immediately without a Kaggle download; document how to replace it with full data.
- `src/config.py`: centralize paths, target column name, ID columns (SK_ID_CURR), random seed, WoE bin count, IV thresholds.

### 1.2 Data loading & cleaning
- Load `application_train.csv` (+ `application_test.csv`, `bureau.csv`, `previous_application.csv`, etc. if merging; Phase 1 MVP uses application_train only).
- Drop ID column from features; separate target.
- Missing values: credit-specific strategy — do NOT blanket impute. Flag missingness as its own category ("MISSING") for categorical/binned variables; median impute only for strongly numeric variables with <40% missing; drop columns >60% missing.
- Outlier caps: winsorize extreme values at 1st/99th percentile for income, credit amount, annuity (credit-risk convention).
- Type fixes: convert flag columns (FLAG_OWN_CAR, etc.) to int/binary; days-based features (DAYS_BIRTH, DAYS_EMPLOYED) are negative — keep sign, create ABS derivatives where meaningful.

### 1.3 Class imbalance
- Default rate is low (~8% in Home Credit). Measure `pos_frac`.
- Do NOT resample the training set blindly for a scorecard — preserves rate calibration needed for risk bands.
- Handle imbalance via: (a) `class_weight='balanced'` in Logistic Regression, (b) `scale_pos_weight = neg/pos` in XGBoost, (c) optional SMOTE as a toggle in `imbalance.py` with clear documentation that SMOTE distorts rate calibration (not recommended for production scorecards).
- Stratified train/validation/test split (70/15/15) by target; keep test set completely held out until final evaluation.

### 1.4 Credit-specific feature engineering
- Domain features: `ANNUITY_INCOME_RATIO = AMT_ANNUITY / AMT_INCOME_TOTAL`, `CREDIT_INCOME_RATIO`, `DAYS_EMPLOYED_PERCENT = DAYS_EMPLOYED / DAYS_BIRTH`, `CHILDREN_RATIO`, `OWN_CAR_AGE` if external data merged.
- External-data merge (optional stage 2): bureau (previous loans), previous_application (prior credit applications) — aggregate counts/sums by SK_ID_CURR.

### 1.5 WoE & IV transformations (core deliverable)
- **Binning:** For each numeric feature, create quantile-based bins (e.g., 10 bins) with monotonicity check; for categorical, group rare categories (<5%) into "Other".
- **WoE per bin:** `WoE = ln(%non_events_in_bin / %events_in_bin)` with smoothing (add small constant to zero counts to avoid `ln(0)`).
- **IV per feature:** `IV = sum((%non_events - %events) * WoE)` across bins.
- **IV thresholds:** <0.02 useless, 0.02–0.1 weak, 0.1–0.3 medium, >0.3 strong (skeptical of >0.5 — possible leakage).
- Replace original features with WoE values for the Logistic Regression scorecard. Keep raw features for XGBoost.
- Save binning maps + WoE mappings to JSON (needed at inference to transform new applicants identically).

**Validation:** unit-check WoE monotonicity per feature; IV ranking stability across train vs validation split; no feature with IV > 0.5 without leakage review.

---

## Phase 2 — Model Development (Hybrid Approach)

### 2.1 Logistic Regression scorecard (WoE/IV)
- Input: WoE-transformed features selected by IV threshold (e.g., IV > 0.02) and correlation check (drop one of any pair |rho| > 0.6).
- Model: `LogisticRegression(penalty='l2', class_weight='balanced', solver='lbfgs', max_iter=1000)`.
- **Scorecard scaling:** convert log-odds to points. Choose reference odds and points-to-double-odds (PDO), e.g., 600 points at odds 1:10, PDO=20. Formula: `points = offset + factor * log_odds`. Produce `src/models/scorecard.py` with `score()` returning points + risk band.
- Output: default probability + scorecard points + reason codes (top contributing features by coefficient × WoE).

### 2.2 XGBoost high-performance model
- Input: raw features + engineered domain features (NOT WoE — tree models handle raw scales; WoE optional).
- Model: `XGBClassifier(objective='binary:logistic', eval_metric='auc', scale_pos_weight=neg/pos, n_estimators, max_depth, learning_rate, subsample, colsample_bytree, random_seed)`.
- Early stopping on validation AUC; tune via small grid/random search over `max_depth`, `learning_rate`, `n_estimators`, `subsample`.
- Calibrate probabilities (Platt scaling / isotonic) if predicted default rates deviate from observed rates — important for banking use.

### 2.3 Comparative evaluation (AUC + KS)
- Metrics (in `src/evaluation/metrics.py`):
  - **AUC-ROC:** discrimination across all thresholds.
  - **KS statistic:** max separation between cumulative distributions of scores for defaulted vs non-defaulted applicants. `KS = max(|CDF_default(score) - CDF_nondefault(score)|)`. Industry benchmark: KS > 40 is strong for credit.
  - Also report: Gini = 2×AUC − 1, calibration (observed vs predicted default rate by decile), confusion matrix at a business threshold (e.g., score cutoff tied to a target approval rate).
- Comparison table: Logistic Regression (scorecard) vs XGBoost — AUC, KS, calibration, interpretability, inference latency.
- **Expected outcome:** XGBoost higher AUC/KS; Logistic Regression lower but fully interpretable and regulator-friendly. This trade-off is the portfolio story.

**Validation:** both models beat a naive baseline (always predict majority class / AUC=0.5); KS computed on held-out test only; no target leakage (exclude future-information features like `DAYS_LAST_PHONE_CHANGE` if post-default).

---

## Phase 3 — Explainability & Interpretability Layer

### 3.1 SHAP for XGBoost
- `src/explain/shap_explainer.py`: `shap.TreeExplainer(xgb_model)`.
- **Global:** summary plot (beeswarm), bar plot of mean |SHAP|, dependence plots for top features. Save as PNGs to `notebooks/` or `docs/`.
- **Local (per-applicant):** `shap_values = explainer.shap_values(applicant_row)`; produce a waterfall/force plot per applicant showing which features push the prediction toward/away from default.
- Map SHAP values back to business language: "AMT_INCOME_TOTAL decreased risk by X points", "CNT_CHILDREN increased risk by Y points".
- **Link to scorecard:** show that Logistic Regression coefficient × WoE ≈ directional SHAP contribution for the same feature, demonstrating consistency between the two models (nice portfolio talking point).

### 3.2 Regulatory framing
- Document: adverse-action reason codes (top 4 features by |SHAP| or coefficient contribution), model governance (versioning, sign-off), and how explanations satisfy "right to explanation" / fair-lending expectations.
- Output format per applicant: default probability, scorecard points, risk band, top 3–5 contributing features with direction and magnitude.

---

## Phase 4 — MLOps & Productionization

### 4.1 MLflow experiment tracking
- `src/mlflow_tracking.py`: initialize local MLflow (`mlflow.set_tracking_uri('mlruns')` or file store).
- Log per run: parameters (model type, hyperparameters, IV threshold, bin count), metrics (AUC, KS, Gini, calibration error), model artifact (`mlflow.sklearn.log_model` / `mlflow.xgboost.log_model`), WoE binning maps as artifacts.
- Tag runs with dataset version, git commit, and `model_stage` (candidate/challenger/champion).
- Select champion: model with best KS on test subject to interpretability constraint; register in MLflow model registry (optional) or promote via `mlruns` structure.

### 4.2 FastAPI production API
- `src/api/schema.py`: Pydantic `CreditApplication` (request fields mirroring model input) and `PredictionResponse` (default_probability, scorecard_points, risk_band, top_contributors: list of {feature, direction, magnitude, shap_value}).
- `src/api/predict.py`: load champion model + binning maps; transform incoming applicant (apply same WoE binning for scorecard path), run inference, compute SHAP explanation, assemble response.
- `src/api/app.py`: FastAPI app with endpoints:
  - `GET /health`
  - `POST /predict` → PredictionResponse
  - `POST /predict/batch` → list of responses (optional)
  - `GET /model-info` (model version, metrics, features)
- Run locally: `uvicorn src.api.app:app --reload`; document Dockerfile option.

**Validation:** API returns 200 with well-formed response for a known applicant; default probability matches batch pipeline output within tolerance; explanation includes exactly the top-N contributors.

---

## Phase 5 — Portfolio & Resume Optimization

### 5.1 GitHub documentation
- `README.md` with: project title, one-line value prop, dataset + problem statement, architecture diagram, phase summary, key results (AUC/KS numbers), how to run (`pip install -r requirements.txt`, `python scripts/train.py`, `uvicorn ...`), example API request/response, and a "Regulatory & interpretability" section.
- Commit notebooks with executed outputs (or use `nbstripout` to keep diffs clean) and a `results/` folder with AUC/KS comparison table and SHAP plots.
- Include `data/synthetic_sample.csv` and a `DATA_DOWNLOAD.md` explaining Kaggle credentials for the full Home Credit set.
- Tags/topics: `credit-risk`, `explainable-ai`, `shap`, `xgboost`, `mlflow`, `fastapi`, `fintech`, `mlops`.

### 5.2 Resume positioning
- **Domain awareness:** mention regulatory compliance (adverse-action reason codes, model governance, rate calibration, fair-lending-friendly scorecard), WoE/IV industry-standard transformations, and the interpretability-vs-performance trade-off decision framework.
- **Technical depth:** name the stack (XGBoost, Logistic Regression, SHAP, MLflow, FastAPI), metrics (AUC, KS, Gini, calibration), and the hybrid modeling approach with concrete numbers.
- Sample resume bullet: "Built an end-to-end explainable credit risk scoring pipeline on Home Credit data; deployed a hybrid Logistic-Regression scorecard (WoE/IV) and XGBoost model with SHAP explanations; tracked experiments via MLflow and exposed predictions through a FastAPI API returning default probability plus top contributing features per applicant."

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Target leakage (features containing post-default info) | Feature audit; exclude any field unknown at application time; document excluded features |
| Class imbalance distorting rate calibration | Use class weights / scale_pos_weight; evaluate calibration; avoid SMOTE for production path |
| Home Credit size/restrictions on GitHub | Commit synthetic sample only; document full-data download via Kaggle API |
| SHAP latency at inference | TreeExplainer is fast; precompute background summary; cap explanation to top-N features |
| Non-monotonic WoE bins breaking scorecard logic | Enforce monotonic binning or merge bins; unit-test monotonicity |

## Success criteria (definition of done)
1. Phase 1: clean dataset, WoE/IV tables with IV ranking, no leakage flags.
2. Phase 2: Logistic Regression scorecard and XGBoost both trained; AUC/KS comparison table produced on held-out test.
3. Phase 3: SHAP global + local explanations generated; per-applicant contributor output works.
4. Phase 4: MLflow logs ≥1 experiment per model; FastAPI `/predict` returns probability + contributors; smoke test passes.
5. Phase 5: README + resume bullet drafted; repo public-ready (no secrets, synthetic data only committed).

---

## Open decision (resolved by recommendation)
- **Dataset:** Home Credit primary (recommended), Lending Club documented fallback. If the implementer cannot access Kaggle, the synthetic sample allows the full pipeline to run end-to-end.
