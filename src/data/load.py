"""Data loading: Kaggle download (Home Credit) or local CSV, with Lending Club fallback."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import ID_COL, RAW_DIR, SYNTHETIC_SAMPLE, TARGET_COL


def _try_kaggle_download() -> Path | None:
    """Attempt to download Home Credit data via the Kaggle API.

    Returns the path to application_train.csv if successful, else None.
    Requires ~/.kaggle/kaggle.json and accepted competition rules.
    """
    try:
        import kaggle
    except ImportError:
        return None

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / "application_train.csv"
    if target.exists():
        return target

    try:
        kaggle.api.competition_download_files(
            "home-credit-default-risk", path=str(RAW_DIR), quiet=True
        )
    except Exception:
        return None

    return target if target.exists() else None


def load_application_train(path: str | Path | None = None) -> pd.DataFrame:
    """Load the Home Credit application_train.csv.

    Priority:
      1. explicit `path` if provided and exists
      2. data/raw/application_train.csv (Kaggle download)
      3. data/synthetic_sample.csv (committed sample, always runnable)
    """
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    candidates.append(RAW_DIR / "application_train.csv")
    candidates.append(SYNTHETIC_SAMPLE)

    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return pd.read_csv(candidate)

    # Last resort: attempt Kaggle download
    downloaded = _try_kaggle_download()
    if downloaded is not None:
        return pd.read_csv(downloaded)

    raise FileNotFoundError(  # noqa: TRY003
        "No application data found. Provide a CSV path, place "
        "application_train.csv under data/raw/, or use data/synthetic_sample.csv."
    )


def load_application_test(path: str | Path | None = None) -> pd.DataFrame | None:
    """Load application_test.csv if available (Home Credit holds-out set)."""
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    candidates.append(RAW_DIR / "application_test.csv")

    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return pd.read_csv(candidate)
    return None


def split_xy(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None]:
    """Separate features from target and ID column.

    Returns (X, y, ids). y and ids are None when the column is absent
    (e.g. application_test.csv without labels).
    """
    ids = df[ID_COL] if ID_COL in df.columns else None
    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    drop_cols = [c for c in [ID_COL, TARGET_COL] if c in df.columns]
    X = df.drop(columns=drop_cols)
    return X, y, ids
