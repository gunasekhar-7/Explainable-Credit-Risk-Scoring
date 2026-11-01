from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_SAMPLE = DATA_DIR / "synthetic_sample.csv"
MLRUNS_DIR = ROOT / "mlruns"
NOTEBOOKS_DIR = ROOT / "notebooks"
RESULTS_DIR = ROOT / "results"

# ---------------------------------------------------------------------------
# Dataset / target
# ---------------------------------------------------------------------------
TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"

# Primary: Home Credit Default Risk (Kaggle). Fallback: Lending Club.
KAGGLE_COMPETITION = "home-credit-default-risk"
KAGGLE_DATASET = "home-credit-default-risk"

# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_SEED = 42
STRATIFY = True

# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
MISSING_CATEGORY = "MISSING"
MISSING_THRESHOLD_DROP = 0.60      # drop columns with >60% missing
MISSING_MEDIAN_MAX = 0.40          # median impute only if <40% missing
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99

# ---------------------------------------------------------------------------
# WoE / IV
# ---------------------------------------------------------------------------
WOE_BIN_COUNT = 10
RARE_CATEGORY_THRESHOLD = 0.05     # group categories below 5% into "Other"
WOE_SMOOTHING = 1e-6

IV_USELESS = 0.02
IV_WEAK = 0.1
IV_MEDIUM = 0.3
IV_STRONG = 0.5                    # >0.5 triggers leakage review

CORR_DROP_THRESHOLD = 0.60         # drop one of any pair |rho| > 0.6

# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
LOGREG_PENALTY = "l2"
LOGREG_SOLVER = "lbfgs"
LOGREG_MAX_ITER = 1000
LOGREG_CLASS_WEIGHT = "balanced"

XGB_OBJECTIVE = "binary:logistic"
XGB_EVAL_METRIC = "auc"

# Scorecard scaling
SCORE_REF_POINTS = 600             # points at reference odds
SCORE_REF_ODDS = 1 / 10            # reference odds (1:10)
SCORE_PDO = 20                     # points to double the odds

# Risk band thresholds (points)
RISK_BAND_THRESHOLDS = [600, 680, 760]  # <600 High, 600-680 Medium-Low, ...

# ---------------------------------------------------------------------------
# API budgets (per client identity, per process)
# ---------------------------------------------------------------------------
RATE_LIMIT_REQUESTS_PER_MINUTE = 120
RATE_LIMIT_ROWS_PER_MINUTE = 6000   # a 500-row batch costs 500; 12 such per minute
RATE_LIMIT_WINDOW_SECONDS = 60.0

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = str(MLRUNS_DIR)
MLFLOW_EXPERIMENT = "explainable-credit-risk-scoring"
