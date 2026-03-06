"""
Benchmark Features — Bible §3.5
=================================
Computes QQQ benchmark features that are broadcast into g_t.

§3.5.1 Benchmark Features List:
    QQQ 1-Week Return    — Log return over 5 trading days
    QQQ 4-Week Volatility— Std of daily log returns × √252 over 20 days
    QQQ 12-Week Return   — Log return over 60 trading days
"""

import numpy as np
import pandas as pd
from typing import Optional

BENCHMARK_FEATURE_NAMES = [
    "qqq_ret_1w",
    "qqq_vol_4w",
    "qqq_ret_12w",
]

DAYS_1W  = 5
DAYS_4W  = 20
DAYS_12W = 60
ANNUALIZE = np.sqrt(252)
EPS = 1e-8


def compute_benchmark_features(
    macro_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute QQQ benchmark features (§3.5.1) from macro_features.parquet.

    Args:
        macro_df : macro_features.parquet DataFrame (DatetimeIndex or date col).

    Returns:
        DataFrame with DatetimeIndex and columns BENCHMARK_FEATURE_NAMES.
    """
    df = macro_df.copy()
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Locate QQQ log return column
    qqq_lr_col = None
    for candidate in ["QQQ_log_return", "QQQ_log_ret", "QQQ_Log_Return"]:
        if candidate in df.columns:
            qqq_lr_col = candidate
            break

    qqq_close_col = None
    for candidate in ["QQQ_Close", "QQQ_close"]:
        if candidate in df.columns:
            qqq_close_col = candidate
            break

    if qqq_lr_col is not None:
        qqq_lr = df[qqq_lr_col].astype(float)
    elif qqq_close_col is not None:
        qqq_close = df[qqq_close_col].astype(float)
        qqq_lr = np.log(qqq_close / qqq_close.shift(1))
    else:
        qqq_lr = pd.Series(0.0, index=df.index)

    # QQQ 1-week return (sum of last 5 log returns)
    qqq_ret_1w = qqq_lr.rolling(DAYS_1W, min_periods=DAYS_1W).sum()

    # QQQ 4-week volatility (annualized std of daily log returns over 20 days)
    qqq_vol_4w = qqq_lr.rolling(DAYS_4W, min_periods=DAYS_4W).std() * ANNUALIZE

    # QQQ 12-week return
    qqq_ret_12w = qqq_lr.rolling(DAYS_12W, min_periods=DAYS_12W).sum()

    out = pd.DataFrame(
        {
            "qqq_ret_1w":  qqq_ret_1w,
            "qqq_vol_4w":  qqq_vol_4w,
            "qqq_ret_12w": qqq_ret_12w,
        },
        index=df.index,
    )

    return out.ffill().fillna(0.0)


def normalize_benchmark_features(
    bench_df: pd.DataFrame,
    fit_end_date: Optional[str] = None,
    clip: float = 4.0,
) -> pd.DataFrame:
    """
    Apply fixed-scale normalization to benchmark features (§3.6.3).

    Args:
        bench_df      : Raw benchmark features DataFrame.
        fit_end_date  : ISO date; normalize fitted on data up to this date.
        clip          : Clip threshold from config.

    Returns:
        Normalized DataFrame, clipped to ±clip.
    """
    from features.normalizers import FixedScaleNormalizer

    norm = FixedScaleNormalizer(clip=clip)
    if fit_end_date:
        fit_data = bench_df[bench_df.index <= pd.Timestamp(fit_end_date)]
    else:
        fit_data = bench_df

    norm.fit(fit_data)
    return norm.transform(bench_df)
