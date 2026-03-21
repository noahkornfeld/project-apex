"""
Macro / Broadcast Features — Bible §3.3
=========================================
Extracts and normalizes macro features from macro_features.parquet.
These are broadcast into the global context vector g_t (not per-asset x_t).

Instruments (§3.3.1):
    QQQ          — Benchmark regime (handled separately in benchmark_features.py)
    VIX          — Equity volatility regime
    10Y Yield    — Discount rate
    3M Yield     — Short rate
    Yield Spread — Cycle phase (10Y - 3M)
    Oil          — Inflation shock
    Gold         — Monetary stress
    Dollar Index — Liquidity regime
    HYG          — Credit stress

Output features (9 values → part of g_t):
    vix_level, vix_4w_trend,
    yield_10y, yield_3m, yield_spread,
    oil_log_ret, gold_log_ret, dxy_log_ret, hyg_log_ret

vix_4w_trend replaces the old 5-day vix_change.  A 20-trading-day (4-week)
delta gives the agent a meaningful VIX trajectory signal — is fear expanding
or contracting — rather than noisy 1-day noise.  Diagnostic evidence showed
corr(rolling_beta, vix_5d_change) ≈ 0 (p=0.80) across all folds, confirming
the 5-day window carried no usable signal.
"""

import numpy as np
import pandas as pd
from typing import Optional

EPS = 1e-8

MACRO_FEATURE_NAMES = [
    "vix_level",
    "vix_4w_trend",
    "yield_10y",
    "yield_3m",
    "yield_spread",
    "oil_log_ret",
    "gold_log_ret",
    "dxy_log_ret",
    "hyg_log_ret",
]


def _find_column(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """Return the first candidate column name that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def compute_macro_broadcast_features(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract 9 macro broadcast features from macro_features.parquet.

    Column names are detected flexibly to handle naming variations.

    Args:
        macro_df : Raw macro_features.parquet DataFrame (DatetimeIndex or date column).

    Returns:
        DataFrame with DatetimeIndex and 9 columns (MACRO_FEATURE_NAMES).
        HYG rows before April 2007 will have hyg_log_ret = 0.0 (imputed).
    """
    df = macro_df.copy()
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    out = pd.DataFrame(index=df.index, dtype=float)

    # --- VIX ---------------------------------------------------------------
    # vix_level   : spot level (1-day window)
    # vix_4w_trend: 20-trading-day (4-week) delta — direction of fear expansion.
    #   Replaces the old 5-day vix_change whose empirical correlation with
    #   rolling portfolio beta was ~0 (p=0.80) across all 8 OOS folds.
    #   A 20-day window captures the meaningful VIX trajectory: is vol in a
    #   sustained rising trend (risk-off) or a sustained falling trend (bull)?
    vix_close = _find_column(df, ["VIX_Close", "VIX_close", "^VIX_Close"])
    if vix_close:
        vix_ser = df[vix_close].astype(float)
        out["vix_level"]    = vix_ser
        out["vix_4w_trend"] = vix_ser.diff(20)       # 4-week (20 trading days)
    else:
        out["vix_level"]    = np.nan
        out["vix_4w_trend"] = np.nan

    # --- Yields ------------------------------------------------------------
    y10 = _find_column(df, ["10Y_Yield_Close", "Yield_10Y", "10Y_Close", "TNX_Close", "^TNX_Close", "yield_10y"])
    y3m = _find_column(df, ["3M_Yield_Close",  "Yield_3M",  "3M_Close",  "IRX_Close", "^IRX_Close", "yield_3m"])
    ysp = _find_column(df, ["Yield_Spread", "yield_spread", "Yield_spread"])

    out["yield_10y"]   = df[y10].values  if y10 else np.nan
    out["yield_3m"]    = df[y3m].values  if y3m else np.nan
    if ysp:
        out["yield_spread"] = df[ysp].values
    elif y10 and y3m:
        out["yield_spread"] = df[y10].values - df[y3m].values
    else:
        out["yield_spread"] = np.nan

    # --- Commodity / FX log returns ----------------------------------------
    def _get_log_ret(candidates_close, candidates_ret):
        col = _find_column(df, candidates_ret)
        if col:
            return df[col].values.astype(float)
        col = _find_column(df, candidates_close)
        if col:
            prices = df[col].astype(float)
            return np.log(prices / prices.shift(1)).values
        return np.full(len(df), np.nan)

    out["oil_log_ret"]  = _get_log_ret(
        ["Oil_Close",          "CL=F_Close"],
        ["Oil_log_return",     "Oil_log_ret",   "CL=F_log_return"],
    )
    out["gold_log_ret"] = _get_log_ret(
        ["Gold_Close",         "GC=F_Close"],
        ["Gold_log_return",    "Gold_log_ret",  "GC=F_log_return"],
    )
    out["dxy_log_ret"]  = _get_log_ret(
        ["Dollar_Index_Close", "DX-Y.NYB_Close"],
        ["Dollar_Index_log_return", "DXY_log_return", "Dollar_Index_log_ret"],
    )
    out["hyg_log_ret"]  = _get_log_ret(
        ["HYG_Close"],
        ["HYG_log_return", "HYG_log_ret"],
    )

    # Forward-fill then fill with 0 for pre-launch HYG and any gaps
    out = out.ffill().fillna(0.0)

    return out[MACRO_FEATURE_NAMES]


def normalize_macro_features(
    macro_feat_df: pd.DataFrame,
    fit_end_date: Optional[str] = None,
    clip: float = 4.0,
) -> pd.DataFrame:
    """
    Apply fixed-scale normalization to macro broadcast features (§3.6.3).

    Fits on data up to `fit_end_date` (or all data if None).
    Same statistics used for OOS — no refitting.

    Args:
        macro_feat_df : Raw macro features (9 columns).
        fit_end_date  : ISO date string; normalization fitted on data up to here.
        clip          : Clipping threshold (norm_clip_threshold from config).

    Returns:
        Normalized DataFrame (same shape), clipped to ±clip.
    """
    from features.normalizers import FixedScaleNormalizer

    norm = FixedScaleNormalizer(clip=clip)
    if fit_end_date:
        fit_data = macro_feat_df[macro_feat_df.index <= pd.Timestamp(fit_end_date)]
    else:
        fit_data = macro_feat_df

    norm.fit(fit_data)
    return norm.transform(macro_feat_df)
