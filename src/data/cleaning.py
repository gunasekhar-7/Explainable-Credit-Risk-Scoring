"""Cleaning: missing-value strategy, outlier caps, type fixes for credit data."""
from __future__ import annotations

import pandas as pd

from src.config import (
    MISSING_CATEGORY,
    MISSING_MEDIAN_MAX,
    MISSING_THRESHOLD_DROP,
    WINSOR_LOWER,
    WINSOR_UPPER,
)


def flag_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Add an _IS_MISSING indicator column for every column with missing values."""
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            df[f"{col}_IS_MISSING"] = df[col].isna().astype(int)
    return df


def drop_high_missing(
    df: pd.DataFrame, threshold: float = MISSING_THRESHOLD_DROP
) -> pd.DataFrame:
    """Drop columns with missing fraction above `threshold`."""
    missing_frac = df.isna().mean()
    to_drop = missing_frac[missing_frac > threshold].index.tolist()
    return df.drop(columns=to_drop), to_drop  # type: ignore[return-value]


def median_impute_low_missing(
    df: pd.DataFrame, max_missing: float = MISSING_MEDIAN_MAX
) -> pd.DataFrame:
    """Median-impute numeric columns whose missing fraction is <= max_missing."""
    df = df.copy()
    for col in df.select_dtypes(include="number").columns:
        frac = df[col].isna().mean()
        if 0 < frac <= max_missing:
            df[col] = df[col].fillna(df[col].median())
    return df


def categorical_missing_to_category(
    df: pd.DataFrame, category: str = MISSING_CATEGORY
) -> pd.DataFrame:
    """Fill missing categorical/object values with a dedicated MISSING category."""
    df = df.copy()
    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].fillna(category).astype(str)
    return df


def winsorize(
    df: pd.DataFrame, lower: float = WINSOR_LOWER, upper: float = WINSOR_UPPER
) -> pd.DataFrame:
    """Winsorize numeric columns at the given quantiles (credit-risk convention)."""
    df = df.copy()
    for col in df.select_dtypes(include="number").columns:
        lo = df[col].quantile(lower)
        hi = df[col].quantile(upper)
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def fix_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert flag columns to int/binary; coerce DAYS-based columns to numeric."""
    df = df.copy()
    flag_cols = [c for c in df.columns if c.upper().startswith("FLAG_")]
    for col in flag_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    for col in df.columns:
        if col.upper().startswith("DAYS_"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean(
    df: pd.DataFrame,
    *,
    drop_high_missing_cols: bool = True,
    winsorize_cols: bool = True,
) -> pd.DataFrame:
    """Full cleaning pipeline for application data."""
    df = fix_types(df)
    df = categorical_missing_to_category(df)
    if drop_high_missing_cols:
        df, _ = drop_high_missing(df)  # type: ignore[assignment]
    df = median_impute_low_missing(df)
    if winsorize_cols:
        df = winsorize(df)
    return df
