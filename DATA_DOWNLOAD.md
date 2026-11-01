# Data Download Instructions

## Primary dataset: Home Credit Default Risk (Kaggle)

The repository commits only a small `data/synthetic_sample.csv` (~200 rows) that mirrors the Home Credit schema so the pipeline runs end-to-end without a Kaggle download. Raw Home Credit data is NOT committed (see `.gitignore`: `data/raw/`).

### Option A — Kaggle API (recommended)

1. Install the Kaggle CLI:

```bash
pip install kaggle
```

2. Obtain your Kaggle API token from https://www.kaggle.com/account (API section) and place it at `~/.kaggle/kaggle.json` (set `chmod 600` on Linux/macOS).

3. Accept the competition rules at https://www.kaggle.com/c/home-credit-default-risk/rules (required once before downloading).

4. Download the competition files:

```bash
kaggle competitions download -c home-credit-default-risk -p data/raw
unzip data/raw/home-credit-default-risk.zip -d data/raw
```

The key file is `application_train.csv`. Additional tables (`bureau.csv`, `previous_application.csv`, etc.) can be merged in a Stage-2 feature step (optional).

### Option B — Kaggle website (manual)

1. Go to https://www.kaggle.com/c/home-credit-default-risk/data.
2. Download `application_train.csv` (and optional supplementary tables).
3. Place `application_train.csv` under `data/raw/`.

### Fallback: Lending Club

If Kaggle access is unavailable, see `src/data/load.py` for the Lending Club fallback path. The synthetic sample keeps the full pipeline runnable regardless of data source.
