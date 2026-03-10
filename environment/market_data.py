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
    bars["dv"]        = bars["close"] * bars["volume"]   # dollar volume (adjusted)

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

    # Build lookup: (date_str, security_id) -> values
    bars["date_str"] = bars["date"].dt.strftime("%Y-%m-%d")
    bars_idx = bars.set_index(["date_str", "security_id"])

    # ------------------------------------------------------------------
    # 4. Map to slot arrays  [T, K_max]
    # ------------------------------------------------------------------
    adj_close_arr   = np.zeros((T, K_max), dtype=np.float32)
    adj_open_arr    = np.zeros((T, K_max), dtype=np.float32)
    adv63_arr       = np.zeros((T, K_max), dtype=np.float32)
    vol_252_arr     = np.zeros((T, K_max), dtype=np.float32)
    gap_vol_252_arr = np.zeros((T, K_max), dtype=np.float32)

    # Vectorised fill using a pivot approach (faster than row-by-row)
    _cols = ["adj_close", "adj_open", "adv63", "vol_252", "gap_vol_252"]
    pivot = bars[["date_str", "security_id"] + _cols].copy()
    pivot = pivot.dropna(subset=["date_str"])

    # Build date_str → row_idx map
    # Build security_id → column_idx map
    all_sids = np.unique(active_ids[active_ids >= 0])
    sid_set  = set(all_sids.tolist())

    # For each unique (date_str, sid) lookup efficiently
    bars_lookup: Dict[str, Dict[int, tuple]] = {}
    for row in pivot.itertuples(index=False):
        d = row.date_str
        s = int(row.security_id)
        if s in sid_set:
            if d not in bars_lookup:
                bars_lookup[d] = {}
            bars_lookup[d][s] = (row.adj_close, row.adj_open,
                                  row.adv63, row.vol_252, row.gap_vol_252)

    for t_idx in range(T):
        d = dates_str[t_idx]
        day_data = bars_lookup.get(d, {})
        for k in range(K_max):
            sid = int(active_ids[t_idx, k])
            if sid >= 0 and sid in day_data:
                vals = day_data[sid]
                adj_close_arr  [t_idx, k] = vals[0] if vals[0] == vals[0] else 0.0
                adj_open_arr   [t_idx, k] = vals[1] if vals[1] == vals[1] else 0.0
                adv63_arr      [t_idx, k] = vals[2] if vals[2] == vals[2] else 0.0
                vol_252_arr    [t_idx, k] = vals[3] if vals[3] == vals[3] else 0.20
                gap_vol_252_arr[t_idx, k] = vals[4] if vals[4] == vals[4] else 0.0

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

    Raw GICS codes (e.g. 10, 45, 50) are mapped to contiguous indices via
    GICS_TO_IDX so they are valid nn.Embedding lookup indices.  Inactive slots
    (security_id < 0) retain -1; the model clamps these to 0 and zeros via mask.
    """
    ndx = pd.read_parquet(ndx_path, columns=["date", "security_id", "sector_code"])
    ndx["date"] = pd.to_datetime(ndx["date"])

    # Build security_id -> sector_code from most recent snapshot
    sid_to_sector: Dict[int, int] = {}
    for snap_date in sorted(ndx["date"].unique()):
        snap = ndx[ndx["date"] == snap_date]
        for row in snap.itertuples(index=False):
            try:
                sc = int(row.sector_code)
            except (ValueError, TypeError):
                sc = -1
            sid_to_sector[int(row.security_id)] = sc

    sector_ids_arr = np.full((T, K_max), -1, dtype=np.int32)
    for t_idx in range(T):
        for k in range(K_max):
            sid = int(active_ids[t_idx, k])
            if sid >= 0:
                raw_code = sid_to_sector.get(sid, -1)
                sector_ids_arr[t_idx, k] = GICS_TO_IDX.get(raw_code, GICS_UNKNOWN_IDX)

    return sector_ids_arr


# ---------------------------------------------------------------------------
# Synthetic market data builder (for tests)
# ---------------------------------------------------------------------------

def make_synthetic_market_data(
    n_days: int = 300,
    K_max: int = 10,
    n_active: int = 8,
    seed: int = 42,
    weekly_every: int = 5,
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
    x_panel = np.zeros((n_days, K_max, 25), dtype=np.float32)
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
