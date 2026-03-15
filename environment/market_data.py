"""
environment/market_data.py
Market data cache builder for the Trading Environment (Phase 5).

Precomputes per-slot arrays needed by the environment step():
  adj_close   [T, K_max] – adjusted close (close * adj_factor)
  adj_open    [T, K_max] – adjusted open  (open  * adj_factor)
  adv63       [T, K_max] – 63-trading-day trailing ADV (dollar volume)
  vol_252     [T, K_max] – 252-day trailing annualised return vol  (σ §5.4)
  gap_vol_252 [T, K_max] – 252-day trailing gap vol (σ_gap §5.4)
  sector_ids  [T, K_max] – GICS sector embedding index per slot (-1 inactive)
  qqq_close   [T]        – QQQ adjusted close
  vix         [T]        – VIX level
  weekly_idx  [W]        – integer panel row indices for weekly rebalance dates

All rolling windows are causal (trailing only, no future look-ahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

EPS = 1e-8
TRADING_DAYS_YEAR = 252
ADV_WINDOW = 63       # §5.4
VOL_WINDOW = 252      # 52 weeks ≈ 252 trading days  §5.4
GAP_VOL_WINDOW = 252  # 52 weeks  §5.4

# Mapping from raw GICS sector codes to contiguous zero-based embedding indices.
# Must match num_sectors in the model config (len = 12: 11 sectors + 1 unknown).
GICS_TO_IDX: Dict[int, int] = {
    10: 0,   # Energy
    15: 1,   # Materials
    20: 2,   # Industrials
    25: 3,   # Consumer Discretionary
    30: 4,   # Consumer Staples
    35: 5,   # Health Care
    40: 6,   # Financials
    45: 7,   # Information Technology
    50: 8,   # Communication Services
    55: 9,   # Utilities
    60: 10,  # Real Estate
    -1: 11,  # Unknown / inactive fallback
}
GICS_UNKNOWN_IDX = 11  # fallback index for unrecognised sector codes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_market_data(
    bars_path: str,
    macro_path: str,
    cal_path: str,
    ndx_path: str,
    shared_panel_path: str,
    fold_train_start: Optional[str] = None,
    fold_train_end:   Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Load all raw data and assemble the market-data cache.

    Parameters
    ----------
    bars_path   : path to daily_bars.parquet
    macro_path  : path to macro_features.parquet
    cal_path    : path to trading_calendar.parquet
    ndx_path    : path to ndx_membership.parquet
    shared_panel_path : path to the shared feature_panel_shared.npz
    fold_train_start / fold_train_end : optional ISO date strings to
        restrict the weekly_idx to a particular fold's training window.

    Returns
    -------
    dict with keys described in module docstring.
    """
    # ------------------------------------------------------------------
    # 1. Load panel metadata
    # ------------------------------------------------------------------
    panel = np.load(shared_panel_path, allow_pickle=True)
    active_ids = panel["active_ids"]          # [T, K_max]  int64
    dates_str  = panel["dates"].astype(str)   # [T] str  "YYYY-MM-DD"
    T, K_max   = active_ids.shape

    dates_pd = pd.to_datetime(dates_str)      # DatetimeIndex (length T)
    date_to_t = {d: i for i, d in enumerate(dates_str)}

    # ------------------------------------------------------------------
    # 2. Build per-security price history
    # ------------------------------------------------------------------
    bars = pd.read_parquet(bars_path, columns=["date", "security_id",
                                                "open", "close",
                                                "volume", "adj_factor"])
    bars["date"] = pd.to_datetime(bars["date"])
    bars = bars.sort_values(["security_id", "date"]).reset_index(drop=True)

    # close and open in daily_bars.parquet are ALREADY total-return adjusted
    # (split + dividend adjusted to current share basis).  adj_factor converts
    # back to the raw historical price (close * adj_factor = raw).  We do NOT
    # multiply here — close/open are the adjusted prices we want.
    bars["adj_close"] = bars["close"]
    bars["adj_open"]  = bars["open"]

    # ------------------------------------------------------------------
    # Price sanity: forward-fill corrupted near-zero prices.
    # A bad split-adjustment or data-provider error produces a single day
    # where adj_close collapses to near-zero then snaps back to the normal
    # price range.  Detect this as: price < 1% of the trailing 21-day median.
    # A real crash (price stays down) is NOT affected — the rolling median
    # falls with the price over many days.
    # ------------------------------------------------------------------
    def _fix_prices(s: pd.Series) -> pd.Series:
        med = s.rolling(21, min_periods=3).median()
        corrupted = (med > 0) & (s < med * 0.01)
        if corrupted.any():
            import logging as _log
            _log.getLogger(__name__).warning(
                f"[market_data] security_id={s.name}: replacing {corrupted.sum()} "
                f"corrupted price(s) with forward-fill"
            )
            s = s.copy()
            s[corrupted] = np.nan
            s = s.ffill().bfill()
        return s

    bars["adj_close"] = bars.groupby("security_id")["adj_close"].transform(_fix_prices)
    bars["adj_open"]  = bars.groupby("security_id")["adj_open"].transform(_fix_prices)

    bars["dv"]        = bars["adj_close"] * bars["volume"]   # dollar volume (adjusted)

    # Daily log-return per security (for vol)
    bars["log_ret"] = bars.groupby("security_id")["adj_close"].transform(
        lambda s: np.log(s / s.shift(1))
    )
    # Overnight gap: open/prev_close - 1
    bars["gap"] = bars.groupby("security_id").apply(
        lambda g: g["adj_open"] / g["adj_close"].shift(1) - 1,
        include_groups=False,
    ).reset_index(level=0, drop=True)

    # ------------------------------------------------------------------
    # 3. Rolling statistics per security (causal trailing windows)
    # ------------------------------------------------------------------
    def _rolling(series: pd.Series, window: int, fn: str) -> pd.Series:
        r = series.rolling(window, min_periods=max(1, window // 4))
        return getattr(r, fn)()

    bars["adv63"]       = bars.groupby("security_id")["dv"].transform(
        lambda s: _rolling(s, ADV_WINDOW, "mean"))
    bars["vol_252"]     = bars.groupby("security_id")["log_ret"].transform(
        lambda s: _rolling(s, VOL_WINDOW, "std") * np.sqrt(TRADING_DAYS_YEAR))
    bars["gap_vol_252"] = bars.groupby("security_id")["gap"].transform(
        lambda s: _rolling(s, GAP_VOL_WINDOW, "std"))

    bars["date_str"] = bars["date"].dt.strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # 4. Map to slot arrays  [T, K_max]  — vectorised numpy approach
    # ------------------------------------------------------------------
    _cols    = ["adj_close", "adj_open", "adv63", "vol_252", "gap_vol_252"]
    all_sids = np.unique(active_ids[active_ids >= 0])   # [N_sids]
    n_sids   = len(all_sids)

    # Map security_id → dense column index (0 .. n_sids-1)
    sid_to_col: Dict[int, int] = {int(s): i for i, s in enumerate(all_sids)}

    # Filter bars to only the sids/dates we need
    sid_set   = set(sid_to_col.keys())
    date_set  = set(dates_str.tolist())
    bars_filt = bars.loc[
        bars["security_id"].isin(sid_set) & bars["date_str"].isin(date_set),
        ["date_str", "security_id"] + _cols,
    ].copy()

    # Integer row/column indices for scatter
    t_vec = bars_filt["date_str"].map(date_to_t).values.astype(np.intp)   # [N]
    c_vec = bars_filt["security_id"].map(sid_to_col).values.astype(np.intp)  # [N]

    # Keep only rows where both indices resolved (no NaN/missing)
    ok = np.isfinite(t_vec.astype(float)) & np.isfinite(c_vec.astype(float))
    t_vec = t_vec[ok];  c_vec = c_vec[ok]
    bars_filt = bars_filt.iloc[ok]

    # Build dense [T, n_sids] arrays via scatter; vol default = 0.20 (safe fallback)
    _defaults = {"adj_close": 0.0, "adj_open": 0.0, "adv63": 0.0,
                 "vol_252": 0.20, "gap_vol_252": 0.0}
    dense: Dict[str, np.ndarray] = {}
    for col, default in _defaults.items():
        arr = np.full((T, n_sids), default, dtype=np.float32)
        vals = bars_filt[col].values.astype(np.float32)
        vals = np.where(np.isfinite(vals), vals, default)   # replace NaN
        arr[t_vec, c_vec] = vals
        dense[col] = arr

    # Map active_ids [T, K_max] → dense column index [T, K_max]
    flat     = active_ids.ravel()                                      # [T*K_max]
    flat_col = (pd.Series(flat.astype(np.int64))
                .map(sid_to_col)
                .fillna(-1)
                .values
                .astype(np.intp)
                .reshape(T, K_max))                                     # [T, K_max]

    # Fancy-index dense → slot arrays in one numpy op per column
    valid_k  = flat_col >= 0                                           # [T, K_max]
    col_safe = np.where(valid_k, flat_col, 0)                          # clip -1→0
    T_idx    = np.arange(T, dtype=np.intp)[:, None]                    # [T, 1]

    adj_close_arr   = np.where(valid_k, dense["adj_close"]  [T_idx, col_safe], 0.0 ).astype(np.float32)
    adj_open_arr    = np.where(valid_k, dense["adj_open"]   [T_idx, col_safe], 0.0 ).astype(np.float32)
    adv63_arr       = np.where(valid_k, dense["adv63"]      [T_idx, col_safe], 0.0 ).astype(np.float32)
    vol_252_arr     = np.where(valid_k, dense["vol_252"]    [T_idx, col_safe], 0.20).astype(np.float32)
    gap_vol_252_arr = np.where(valid_k, dense["gap_vol_252"][T_idx, col_safe], 0.0 ).astype(np.float32)

    # ------------------------------------------------------------------
    # 5. QQQ close and VIX from macro_features.parquet
    # ------------------------------------------------------------------
    macro = pd.read_parquet(macro_path, columns=["date", "QQQ_Close", "VIX_Close"])
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro.set_index("date").reindex(dates_pd, method="ffill")

    qqq_close = macro["QQQ_Close"].values.astype(np.float32)
    vix        = macro["VIX_Close"].fillna(20.0).values.astype(np.float32)

    # ------------------------------------------------------------------
    # 6. Sector IDs  [T, K_max]
    # ------------------------------------------------------------------
    sector_ids_arr = _build_sector_ids(ndx_path, dates_str, active_ids, T, K_max)

    # ------------------------------------------------------------------
    # 7. Weekly rebalance indices
    # ------------------------------------------------------------------
    cal = pd.read_parquet(cal_path, columns=["date", "is_week_start"])
    cal["date"] = pd.to_datetime(cal["date"])
    cal["date_str"] = cal["date"].dt.strftime("%Y-%m-%d")
    weekly_dates = set(cal.loc[cal["is_week_start"], "date_str"].tolist())

    # Filter to fold date range if provided
    start_pd = pd.Timestamp(fold_train_start) if fold_train_start else dates_pd[0]
    end_pd   = pd.Timestamp(fold_train_end)   if fold_train_end   else dates_pd[-1]

    weekly_idx = np.array([
        i for i, d in enumerate(dates_str)
        if d in weekly_dates and start_pd <= dates_pd[i] <= end_pd
    ], dtype=np.int64)

    # Drop any weekly rebalance dates where every slot has adj_close = 0.
    # These arise from market closures (e.g. Hurricane Sandy 2012-10-29) or
    # pre-membership warm-up rows where no NDX stock has data yet.  If such a
    # date is used as p_next in step(), close_next = 0 → ret_asset ≈ -1.0 for
    # all active slots → instant portfolio ruin (observed: Fold 2 MaxDD = -100%).
    if len(weekly_idx) > 0:
        has_price = (adj_close_arr[weekly_idx] > 1e-8).any(axis=1)
        n_dropped = int((~has_price).sum())
        if n_dropped > 0:
            import logging as _log
            bad_dates = ", ".join(dates_str[weekly_idx[~has_price]])
            _log.getLogger(__name__).warning(
                f"[market_data] Skipping {n_dropped} weekly rebalance date(s) "
                f"with no valid adj_close (market holiday / data gap): {bad_dates}"
            )
            weekly_idx = weekly_idx[has_price]

    return dict(
        adj_close   = adj_close_arr,
        adj_open    = adj_open_arr,
        adv63       = adv63_arr,
        vol_252     = vol_252_arr,
        gap_vol_252 = gap_vol_252_arr,
        sector_ids  = sector_ids_arr,
        qqq_close   = qqq_close,
        vix         = vix,
        weekly_idx  = weekly_idx,
        dates_str   = dates_str,
        active_ids  = active_ids,
    )


# ---------------------------------------------------------------------------
# Sector ID helper
# ---------------------------------------------------------------------------

def _build_sector_ids(
    ndx_path: str,
    dates_str: np.ndarray,
    active_ids: np.ndarray,
    T: int,
    K_max: int,
) -> np.ndarray:
    """Build [T, K_max] int32 array of GICS sector embedding indices per slot per day.

    Uses an as-of rule: for each (panel_date, security_id) pair, the sector
    code is taken from the most recent NDX snapshot whose date ≤ panel_date.
    This preserves correct historical sector assignments for companies that
    changed sectors (e.g. moved from Consumer Discretionary to Technology).

    Raw GICS codes (e.g. 10, 45, 50) are mapped to contiguous indices via
    GICS_TO_IDX so they are valid nn.Embedding lookup indices.  Inactive slots
    (security_id < 0) retain -1; unknown/missing codes map to GICS_UNKNOWN_IDX.
    """
    ndx = pd.read_parquet(ndx_path, columns=["date", "security_id", "sector_code"])
    ndx["date"] = pd.to_datetime(ndx["date"])
    ndx["security_id"] = ndx["security_id"].astype(np.int64)
    ndx_sorted = (ndx[["date", "security_id", "sector_code"]]
                  .sort_values("date")
                  .reset_index(drop=True))

    # Build flat query: one row per active (t, k) slot
    dates_dt    = pd.to_datetime(dates_str)                     # [T] DatetimeSeries
    flat_dates  = np.repeat(dates_dt, K_max)                    # [T*K_max]
    flat_sids   = active_ids.ravel().astype(np.int64)           # [T*K_max]
    flat_pos    = np.arange(T * K_max, dtype=np.intp)           # original positions

    active_mask = flat_sids >= 0
    queries = pd.DataFrame({
        "date":        flat_dates[active_mask],
        "security_id": flat_sids[active_mask],
        "_pos":        flat_pos[active_mask],
    }).sort_values("date").reset_index(drop=True)

    # merge_asof: for each (query_date, security_id), find the most recent
    # ndx snapshot with snapshot_date ≤ query_date  (as-of / backward rule)
    merged = pd.merge_asof(
        queries,
        ndx_sorted,
        on="date",
        by="security_id",
        direction="backward",
    )

    # Vectorised GICS code → embedding index mapping
    sc_num  = pd.to_numeric(merged["sector_code"], errors="coerce").fillna(-1).astype(int)
    emb_idx = (sc_num.map(GICS_TO_IDX)
                     .fillna(GICS_UNKNOWN_IDX)
                     .astype(np.int32))

    # Scatter results back into flat [T*K_max] array; inactive slots stay -1
    flat_result = np.full(T * K_max, -1, dtype=np.int32)
    flat_result[merged["_pos"].values.astype(np.intp)] = emb_idx.values

    return flat_result.reshape(T, K_max)


# ---------------------------------------------------------------------------
# Synthetic market data builder (for tests)
# ---------------------------------------------------------------------------

def make_synthetic_market_data(
    n_days: int = 300,
    K_max: int = 10,
    n_active: int = 8,
    seed: int = 42,
    weekly_every: int = 5,
    F: int = 25,
) -> Dict[str, np.ndarray]:
    """Build a small synthetic market-data dict for unit tests.

    Assets follow geometric Brownian motion with deterministic QQQ benchmark.
    All rolling stats are simplified (constant) for predictability.
    """
    rng = np.random.default_rng(seed)

    # Prices
    log_rets = rng.normal(0.0008, 0.012, size=(n_days, K_max)).astype(np.float32)
    log_rets[:, n_active:] = 0.0   # inactive slots
    prices = np.exp(np.cumsum(log_rets, axis=0))  # [T, K_max]
    prices = np.clip(prices, 0.01, None)

    adj_close   = prices.copy()
    adj_open    = prices * (1 + rng.normal(0, 0.003, size=prices.shape)).astype(np.float32)

    # Cost model inputs (constant for simplicity)
    adv63       = np.full((n_days, K_max), 5e8, dtype=np.float32)
    vol_252     = np.full((n_days, K_max), 0.25, dtype=np.float32)
    gap_vol_252 = np.full((n_days, K_max), 0.008, dtype=np.float32)

    # Mask inactive
    adv63      [:, n_active:] = 0.0
    vol_252    [:, n_active:] = 0.0
    gap_vol_252[:, n_active:] = 0.0

    # QQQ close (simple trend)
    qqq_log = rng.normal(0.0005, 0.010, size=n_days).astype(np.float32)
    qqq_close = np.exp(np.cumsum(qqq_log)).astype(np.float32)

    # VIX (constant at 15)
    vix = np.full(n_days, 15.0, dtype=np.float32)

    # Sector IDs (cycle 0..3 for active, -1 for inactive)
    sector_ids = np.full((n_days, K_max), -1, dtype=np.int32)
    for k in range(n_active):
        sector_ids[:, k] = k % 4

    # Active IDs
    active_ids = np.full((n_days, K_max), -1, dtype=np.int64)
    for k in range(n_active):
        active_ids[:, k] = k + 100   # synthetic security_ids

    # Mask panel
    mask_panel = np.zeros((n_days, K_max), dtype=np.float32)
    mask_panel[:, :n_active] = 1.0

    # Weekly index (every `weekly_every` trading days)
    weekly_idx = np.arange(0, n_days, weekly_every, dtype=np.int64)

    # Dates (synthetic)
    import pandas as pd
    base = pd.Timestamp("2010-01-04")
    all_dates = [
        (base + pd.offsets.BDay(i)).strftime("%Y-%m-%d")
        for i in range(n_days)
    ]
    dates_str = np.array(all_dates, dtype=object)

    # Minimal x_panel and g_panel (zeros)
    x_panel = np.zeros((n_days, K_max, F), dtype=np.float32)
    g_panel  = np.zeros((n_days, 20), dtype=np.float32)

    return dict(
        adj_close   = adj_close,
        adj_open    = adj_open,
        adv63       = adv63,
        vol_252     = vol_252,
        gap_vol_252 = gap_vol_252,
        sector_ids  = sector_ids,
        qqq_close   = qqq_close,
        vix         = vix,
        weekly_idx  = weekly_idx,
        dates_str   = dates_str,
        active_ids  = active_ids,
        mask_panel  = mask_panel,
        x_panel     = x_panel,
        g_panel     = g_panel,
    )
