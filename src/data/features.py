"""WoE / IV transformations: binning, monotonicity, maps persisted to JSON."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CORR_DROP_THRESHOLD,
    IV_MEDIUM,
    IV_STRONG,
    IV_USELESS,
    IV_WEAK,
    MISSING_CATEGORY,
    RARE_CATEGORY_THRESHOLD,
    WOE_BIN_COUNT,
    WOE_SMOOTHING,
)


def bin_numeric(
    series: pd.Series, n_bins: int = WOE_BIN_COUNT
) -> tuple[pd.Series, np.ndarray]:
    """Quantile-based binning of a numeric series. Returns binned series + bin edges.

    Edges are quantiles of the *observed values*, so cutting the same series with
    them yields `n_bins` populated bins. (Deriving edges from `series.rank()`
    instead would put the edges on a 1..n scale and collapse every real-valued
    feature into a single bin.)
    """
    series = pd.to_numeric(series, errors="coerce")
    _, edges = pd.qcut(series, q=n_bins, retbins=True, duplicates="drop")
    edges = np.asarray(edges, dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    binned = pd.cut(series, bins=edges, include_lowest=True, duplicates="drop")  # type: ignore[call-overload]
    return binned, edges


def bin_categorical(
    series: pd.Series, rare_threshold: float = RARE_CATEGORY_THRESHOLD
) -> pd.Series:
    """Group categories below `rare_threshold` frequency into 'Other'."""
    freq = series.value_counts(normalize=True)
    rare = set(freq[freq < rare_threshold].index)
    filled = series.fillna(MISSING_CATEGORY).astype(str)
    return filled.apply(lambda v: "Other" if v in rare else v)


def compute_woe_iv(
    y: pd.Series, x_binned: pd.Series
) -> tuple[pd.DataFrame, float]:
    """Compute WoE and IV per bin for a binned feature against binary target y.

    Returns (woe_df, iv). woe_df has columns: bin, n, n_events, n_nonevents,
    pct_events, pct_nonevents, woe, iv_contribution.
    """
    df = pd.DataFrame({"bin": x_binned.astype(str), "y": y.values})
    grouped = df.groupby("bin", observed=True).agg(
        n=("y", "size"), n_events=("y", "sum")
    )
    grouped["n_nonevents"] = grouped["n"] - grouped["n_events"]

    n_events_total = grouped["n_events"].sum()
    n_nonevents_total = grouped["n_nonevents"].sum()

    grouped["pct_events"] = grouped["n_events"] / max(n_events_total, 1)
    grouped["pct_nonevents"] = grouped["n_nonevents"] / max(n_nonevents_total, 1)

    pe = grouped["pct_events"].clip(lower=WOE_SMOOTHING)
    pn = grouped["pct_nonevents"].clip(lower=WOE_SMOOTHING)
    grouped["woe"] = np.log(pn / pe)
    grouped["iv_contribution"] = (pn - pe) * grouped["woe"]

    iv = float(grouped["iv_contribution"].sum())
    return grouped.reset_index(), iv


def iv_strength(iv: float) -> str:
    if iv < IV_USELESS:
        return "useless"
    if iv < IV_WEAK:
        return "weak"
    if iv < IV_MEDIUM:
        return "medium"
    if iv < IV_STRONG:
        return "strong"
    return "very_strong_review_leakage"


def woe_monotonic(woe_df: pd.DataFrame) -> bool:
    """Check whether WoE values are monotonic across ordered bins."""
    woes = woe_df["woe"].values
    return bool(
        np.all(np.diff(woes) >= 0) or np.all(np.diff(woes) <= 0)  # type: ignore[type-var, operator]
    )


def apply_woe_maps(
    X: pd.DataFrame, maps: dict
) -> pd.DataFrame:
    """Replace raw feature values with WoE values using persisted binning maps.

    Missing or unusable values fall back to a neutral 0.0 WoE contribution
    (WoE 0 == the bin's odds match the population), so partial applicants are
    scorable. Columns are assembled in one pass: assigning column-by-column
    into an empty frame dominates inference latency otherwise.
    """
    X_woe, _labels = apply_woe_maps_with_labels(X, maps)
    return X_woe


def apply_woe_maps_with_labels(
    X: pd.DataFrame, maps: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Like :func:`apply_woe_maps`, but also return the bin each value landed in.

    The second frame carries the human-readable bin membership per feature
    (the interval for numeric maps, the category for categorical maps), so a
    reason code can say *which* bin drove the contribution instead of only
    naming the column. Unprovided or unusable values get ``None``: their WoE is
    the neutral 0.0, which means they carry no contribution to explain.
    """
    columns = list(maps)
    n_rows = len(X)
    if n_rows == 0:
        return (
            pd.DataFrame({c: pd.Series(dtype=float) for c in columns}, index=X.index),
            pd.DataFrame({c: pd.Series(dtype=object) for c in columns}, index=X.index),
        )

    # Assembled as two dense matrices and wrapped once at the end. Building a
    # Series per feature per frame dominated inference latency, because a
    # single-row request pays that construction cost for every column.
    woe_matrix = np.zeros((n_rows, len(columns)), dtype=float)
    label_matrix = np.empty((n_rows, len(columns)), dtype=object)
    label_matrix[:] = None

    for j, col in enumerate(columns):
        spec = maps[col]
        woe_map = spec["woe_map"]
        if col not in X.columns:
            # Feature not provided at inference: neutral zero WoE contribution
            continue
        values = X[col]
        keys: np.ndarray
        if spec["type"] == "numeric":
            # Coerce first: a column of `None` arrives as object dtype, and
            # pd.cut cannot compare None against numeric bin edges.
            numeric = pd.to_numeric(values, errors="coerce")
            known = numeric.notna().to_numpy()
            binned = pd.cut(
                numeric, bins=spec["bin_edges"], include_lowest=True, duplicates="drop"
            )
            # Bin keys are each interval's own string form, matching how training
            # wrote them; map once per bin instead of once per row.
            categories = np.asarray([str(c) for c in binned.cat.categories], dtype=object)
            woe_by_bin = np.asarray(
                [woe_map.get(str(c), 0.0) for c in binned.cat.categories], dtype=float
            )
            codes = np.asarray(binned.cat.codes)
            inside = codes >= 0
            # -1 == the value had no bin (missing): neutral WoE, and no label.
            woe_matrix[:, j] = np.where(inside, woe_by_bin[np.clip(codes, 0, None)], 0.0)
            keys = np.asarray(np.where(inside, categories[np.clip(codes, 0, None)], None))
        else:
            known = values.notna().to_numpy()
            keys = np.asarray(
                np.where(known, values.to_numpy(dtype=object, na_value=None), MISSING_CATEGORY)
            )
            # Collapse categories the maps never saw onto the rare bucket.
            known_categories = np.asarray([*woe_map], dtype=object)
            unknown = ~np.isin(keys, known_categories) & (keys != MISSING_CATEGORY)
            keys = np.asarray(np.where(unknown, "Other", keys))
            woe_matrix[:, j] = np.fromiter(
                (woe_map.get(k, 0.0) for k in keys.tolist()), dtype=np.float64, count=n_rows
            )
        labels = keys.copy()
        labels[~known] = None
        label_matrix[:, j] = labels

    return (
        pd.DataFrame(woe_matrix, index=X.index, columns=columns),
        pd.DataFrame(label_matrix, index=X.index, columns=columns),
    )


def select_features_by_iv(
    maps: dict, min_iv: float = IV_USELESS
) -> list[str]:
    """Return features with IV >= min_iv, sorted by IV desc."""
    ranked = [
        (col, spec["iv"]) for col, spec in maps.items()
    ]
    ranked = [(c, iv) for c, iv in ranked if iv >= min_iv]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked]


def drop_correlated(
    X: pd.DataFrame, threshold: float = CORR_DROP_THRESHOLD
) -> tuple[pd.DataFrame, list[str]]:
    """Drop one feature of any pair with |correlation| > threshold."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return X.drop(columns=to_drop), to_drop


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Credit-specific domain features (Phase 1.4).

    Applied identically at training and at inference: a model that selects one
    of these ratios must receive it at serving time too, otherwise the feature
    is silently neutral and the deployed model is weaker than the trained one.
    Operands are coerced, because a request may carry ``None`` for any field.
    """
    df = df.copy()

    def numeric(col: str) -> pd.Series:
        return pd.to_numeric(df[col], errors="coerce")

    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["ANNUITY_INCOME_RATIO"] = numeric("AMT_ANNUITY") / numeric(
            "AMT_INCOME_TOTAL"
        ).replace(0, np.nan)
    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["CREDIT_INCOME_RATIO"] = numeric("AMT_CREDIT") / numeric(
            "AMT_INCOME_TOTAL"
        ).replace(0, np.nan)
    if {"DAYS_EMPLOYED", "DAYS_BIRTH"}.issubset(df.columns):
        df["DAYS_EMPLOYED_PERCENT"] = numeric("DAYS_EMPLOYED") / numeric(
            "DAYS_BIRTH"
        ).replace(0, np.nan)
    if {"CNT_CHILDREN", "CNT_FAM_MEMBERS"}.issubset(df.columns):
        df["CHILDREN_RATIO"] = numeric("CNT_CHILDREN") / numeric(
            "CNT_FAM_MEMBERS"
        ).replace(0, np.nan)
    return df


def enforce_monotonic_woe(
    woe_df: pd.DataFrame,
    bin_edges: np.ndarray,
    min_bins: int = 3,
    iv_threshold: float = 0.0,
) -> tuple[pd.DataFrame, np.ndarray, bool]:
    """Merge adjacent bins until WoE is monotonic or the min_bins floor is reached.

    Returns (woe_df, edges, degenerate). degenerate is True when bin count fell
    to min_bins or IV dropped below iv_threshold after a merge, signaling the
    caller may drop the feature.
    """
    woe_df = woe_df.reset_index(drop=True).copy()
    edges = bin_edges.copy()
    degenerate = False
    changed = True
    while changed and len(woe_df) > min_bins:
        woes = woe_df["woe"].values
        diffs = np.diff(woes)  # type: ignore[type-var]
        increasing = np.all(diffs >= 0)  # type: ignore[operator]
        decreasing = np.all(diffs <= 0)  # type: ignore[operator]
        if increasing or decreasing:
            changed = False
            break
        # find first non-monotonic step and merge it with the next bin
        idx = int(np.argmax(np.abs(diffs[1:] - diffs[:-1]))) if len(diffs) > 1 else 0
        if idx >= len(woe_df) - 1:
            idx = len(woe_df) - 2
        # Merge bin idx+1 into bin idx: drop row idx+1 and the edge between them.
        # `edges` has len(woe_df)+1 entries, so the shared boundary is edges[idx+1].
        # (Deleting edges[idx+2] would instead weld bin idx+1 to bin idx+2, and at
        # the last index it would delete the +inf edge entirely.)
        woe_df = woe_df.drop(index=idx + 1).reset_index(drop=True)
        edges = np.delete(edges, idx + 1)
        if len(woe_df) <= min_bins:
            degenerate = True
            break
    if iv_threshold > 0 and float(woe_df["iv_contribution"].sum()) < iv_threshold:
        degenerate = True
    return woe_df, edges, degenerate


def build_woe_maps(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    maps_path: str | Path | None = None,
    enforce_monotonic: bool = True,
) -> dict:
    """Build WoE/IV binning maps for every feature.

    Returns dict: {col: {"type": numeric|categorical, "iv": float, "strength": str,
    "monotonic": bool, "bins": [...], "woe_map": {bin_value: woe}}}.
    """
    maps: dict = {}
    for col in numeric_cols:
        binned, edges = bin_numeric(X[col])
        woe_df, iv = compute_woe_iv(y, binned)
        degenerate = False
        if enforce_monotonic:
            # Merging changes bin membership, so WoE must be re-derived from the
            # merged edges and the check repeated: a single call reasons about its
            # own merge, not about the WoE actually observed afterwards.
            for _ in range(len(edges)):
                if woe_monotonic(woe_df):
                    break
                bins_before = len(woe_df)
                woe_df, edges, degenerate = enforce_monotonic_woe(
                    woe_df, edges, min_bins=3, iv_threshold=IV_USELESS
                )
                binned = pd.cut(X[col], bins=edges, include_lowest=True, duplicates="drop")  # type: ignore[call-overload]
                woe_df, iv = compute_woe_iv(y, binned)
                if degenerate or len(woe_df) >= bins_before:
                    break
            # A feature that cannot be binned monotonically is dropped rather than
            # shipped with a misleading `monotonic: False` map.
            degenerate = degenerate or not woe_monotonic(woe_df)
        bin_labels = [str(interval) for interval in woe_df["bin"]]
        if degenerate:
            # Feature failed monotonicity/IV floor: exclude from maps entirely
            # so select_features_by_iv and the scorecard never receive it.
            continue
        maps[col] = {
            "type": "numeric",
            "iv": iv,
            "strength": iv_strength(iv),
            "monotonic": woe_monotonic(woe_df),
            "bin_edges": edges.tolist(),
            "bin_labels": bin_labels,
            "woe_map": dict(zip(bin_labels, woe_df["woe"].tolist(), strict=True)),
        }

    for col in categorical_cols:
        binned = bin_categorical(X[col])
        woe_df, iv = compute_woe_iv(y, binned)
        maps[col] = {
            "type": "categorical",
            "iv": iv,
            "strength": iv_strength(iv),
            "monotonic": None,
            "woe_map": dict(zip(woe_df["bin"].tolist(), woe_df["woe"].tolist(), strict=True)),
        }

    if maps_path is not None:
        Path(maps_path).parent.mkdir(parents=True, exist_ok=True)
        with open(maps_path, "w", encoding="utf-8") as fh:
            json.dump(maps, fh, indent=2)

    return maps
