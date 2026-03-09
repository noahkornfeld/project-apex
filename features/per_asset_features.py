"""
Per-Asset Time-Series Features — Bible §3.1
============================================
Computes all 17 per-asset features for a single security's price history.

All computations are strictly causal: at time t, only data up to and
including t is used. Pandas rolling() windows enforce this automatically.

Feature list (order matches config per_asset_ts_features):
    open, close, volume, log_ret,
    ret_1w, ret_4w, ret_12w,
    vol_1w, vol_4w, vol_52w,
    volume_z_4w, beta_26w_mkt, rel_strength_4w,
    vol_ratio_1w_4w, RSI_14,
    bollinger_percent_b, bollinger_bandwidth
"""

import numpy as np
import pandas as pd
from typing import Optional

# ---------------------------------------------------------------------------
# Trading-day window constants
# ---------------------------------------------------------------------------
DAYS_1W  = 5
DAYS_4W  = 20
DAYS_12W = 60
DAYS_26W = 130
DAYS_52W = 260
DAYS_RSI = 14
DAYS_BOLL = 20
ANNUALIZE = np.sqrt(252)

EPS = 1e-8


# ---------------------------------------------------------------------------
# RSI helper
# ---------------------------------------------------------------------------

def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (Wilder smoothing via EWM).
    Returns values in [0, 100]; NaN for the first `period` rows.
    Causal by construction — uses only past data.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + EPS)
    return 100.0 - (100.0 / (1.0 + rs))


# ---------------------------------------------------------------------------
# Bollinger Bands helper
# ---------------------------------------------------------------------------

def _compute_bollinger(close: pd.Series, window: int = 20):
    """
    Bollinger Bands (%B and bandwidth).

    Returns:
        pct_b      — (close - lower) / (upper - lower); position within bands
        bandwidth  — (upper - lower) / mid; relative band width
    """
    mid   = close.rolling(window, min_periods=window).mean()
    std   = close.rolling(window, min_periods=window).std()
    upper = mid + 2.0 * std
    lower = mid - 2.0 * std

    pct_b     = (close - lower) / (upper - lower + EPS)
    bandwidth = (upper - lower) / (mid + EPS)

    return pct_b, bandwidth


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_per_asset_features(
    bars: pd.DataFrame,
    qqq_close: pd.Series,
    qqq_log_ret: pd.Series,
) -> pd.DataFrame:
    """
    Compute all 17 §3.1 per-asset time-series features for one security.

    Args:
        bars         : DataFrame with DatetimeIndex and columns
                       [open, close, volume].  `close` must already be
                       total-return adjusted.
        qqq_close    : DatetimeIndex series of QQQ adjusted close prices.
        qqq_log_ret  : DatetimeIndex series of QQQ daily log returns.

    Returns:
        DataFrame with DatetimeIndex matching `bars.index` and 17 feature
        columns in the canonical order defined by §3.1.1.
        NaN appears in early rows where rolling windows are not yet full —
        the normalizer and panel builder handle these via forward-fill or 0.
    """
    close  = bars["close"].astype(float)
    open_  = bars["open"].astype(float)
    volume = bars["volume"].astype(float)

    # --- Log return (daily) -----------------------------------------------
    log_ret = np.log(close / close.shift(1))

    # --- Multi-window log returns (causal rolling sum) --------------------
    ret_1w  = log_ret.rolling(DAYS_1W,  min_periods=DAYS_1W).sum()
    ret_4w  = log_ret.rolling(DAYS_4W,  min_periods=DAYS_4W).sum()
    ret_12w = log_ret.rolling(DAYS_12W, min_periods=DAYS_12W).sum()

    # --- Realized volatility (annualized) ---------------------------------
    vol_1w  = log_ret.rolling(DAYS_1W,  min_periods=DAYS_1W).std()  * ANNUALIZE
    vol_4w  = log_ret.rolling(DAYS_4W,  min_periods=DAYS_4W).std()  * ANNUALIZE
    vol_52w = log_ret.rolling(DAYS_52W, min_periods=DAYS_4W).std() * ANNUALIZE

    # --- Volume z-score (4-week causal) -----------------------------------
    vol_mean = volume.rolling(DAYS_4W, min_periods=DAYS_4W).mean()
    vol_std  = volume.rolling(DAYS_4W, min_periods=DAYS_4W).std()
    volume_z_4w = (volume - vol_mean) / (vol_std + EPS)

    # --- 26-week beta vs QQQ (rolling cov / var) -------------------------
    qqq_lr_aligned = qqq_log_ret.reindex(close.index)
    cov_26w  = log_ret.rolling(DAYS_26W, min_periods=DAYS_26W // 2).cov(qqq_lr_aligned)
    var_mkt  = qqq_lr_aligned.rolling(DAYS_26W, min_periods=DAYS_26W // 2).var()
    beta_26w_mkt = cov_26w / (var_mkt + EPS)

    # --- Relative strength vs QQQ (4-week) --------------------------------
    qqq_ret_4w_aligned = (
        qqq_log_ret
        .rolling(DAYS_4W, min_periods=DAYS_4W)
        .sum()
        .reindex(close.index)
    )
    rel_strength_4w = ret_4w - qqq_ret_4w_aligned

    # --- Vol ratio 1w / 4w -----------------------------------------------
    vol_ratio_1w_4w = vol_1w / (vol_4w + EPS)

    # --- RSI 14 -----------------------------------------------------------
    rsi_14 = _compute_rsi(close, DAYS_RSI)

    # --- Bollinger Bands -------------------------------------------------
    boll_pct_b, boll_bw = _compute_bollinger(close, DAYS_BOLL)

    # --- Assemble ---------------------------------------------------------
    df = pd.DataFrame(
        {
            "open":                open_,
            "close":               close,
            "volume":              volume,
            "log_ret":             log_ret,
            "ret_1w":              ret_1w,
            "ret_4w":              ret_4w,
            "ret_12w":             ret_12w,
            "vol_1w":              vol_1w,
            "vol_4w":              vol_4w,
            "vol_52w":             vol_52w,
            "volume_z_4w":         volume_z_4w,
            "beta_26w_mkt":        beta_26w_mkt,
            "rel_strength_4w":     rel_strength_4w,
            "vol_ratio_1w_4w":     vol_ratio_1w_4w,
            "RSI_14":              rsi_14,
            "bollinger_percent_b": boll_pct_b,
            "bollinger_bandwidth": boll_bw,
        },
        index=bars.index,
    )

    return df


def compute_all_securities_features(
    daily_bars: pd.DataFrame,
    qqq_close: pd.Series,
    qqq_log_ret: pd.Series,
) -> dict:
    """
    Compute §3.1 features for every security in daily_bars.

    Args:
        daily_bars   : Full daily_bars.parquet DataFrame.
        qqq_close    : QQQ adjusted close (DatetimeIndex).
        qqq_log_ret  : QQQ daily log return (DatetimeIndex).

    Returns:
        dict mapping security_id (int) → feature DataFrame (DatetimeIndex, 17 cols).
    """
    all_features: dict = {}
    grouped = daily_bars.groupby("security_id", sort=True)

    for sid, grp in grouped:
        grp = grp.set_index("date").sort_index()
        if len(grp) < DAYS_4W:
            continue
        try:
            feat_df = compute_per_asset_features(grp, qqq_close, qqq_log_ret)
            all_features[int(sid)] = feat_df
        except Exception:
            pass

    return all_features
