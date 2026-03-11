"""
Cross-Sectional Features — Bible §3.2
=======================================
Computes 8 cross-sectional features at each time t across the active NDX universe.

These features are inherently causal: they use only the cross-section of
currently-active assets at time t and their pre-computed time-series values.

Feature list (§3.2.1):
    ret_rank_4w           — Percentile rank of 4-week return
    ret_z_4w              — Z-score of 4-week return across active universe
    ret_z_13w             — Z-score of 13-week return across active universe
    vol_z_4w              — Z-score of 4-week realized vol across active universe
    volume_z_cs_4w        — Cross-sectional volume z-score
    ret_z_4w_sector       — Sector-relative z-score of 4w return (within sector)
    vol_z_4w_sector       — Sector-relative vol z-score (within sector)
    momentum_sector_residual — ret_4w - sector_mean(ret_4w)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

EPS = 1e-8

CS_FEATURE_NAMES = [
    "ret_rank_4w",
    "ret_z_4w",
    "ret_z_13w",
    "vol_z_4w",
    "volume_z_cs_4w",
    "ret_z_4w_sector",
    "vol_z_4w_sector",
    "momentum_sector_residual",
]


def _cs_zscore(values: np.ndarray) -> np.ndarray:
    """Z-score a 1-D array across its finite elements. NaN-safe."""
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return np.zeros_like(values, dtype=float)
    mu  = float(finite.mean())
    std = float(finite.std())
    if std < EPS:
        return np.zeros_like(values, dtype=float)
    return (values - mu) / (std + EPS)


def _cs_rank(values: np.ndarray) -> np.ndarray:
    """Percentile rank in [0, 1] for a 1-D array. NaN-safe."""
    out = np.full_like(values, np.nan, dtype=float)
    finite_mask = np.isfinite(values)
    if finite_mask.sum() < 1:
        return out
    ranks = pd.Series(values[finite_mask]).rank(pct=True).values
    out[finite_mask] = ranks
    return out


def _sector_zscore(values: np.ndarray, sector_codes: np.ndarray) -> np.ndarray:
    """
    Sector-relative z-score: within each sector, z-score the values.
    Falls back to 0 for sectors with fewer than 2 finite members.
    """
    out = np.zeros_like(values, dtype=float)
    for sector in np.unique(sector_codes):
        mask   = sector_codes == sector
        v      = values[mask].astype(float)
        finite = v[np.isfinite(v)]
        if len(finite) < 2:
            out[mask] = 0.0
            continue
        mu  = float(finite.mean())
        std = float(finite.std())
        if std < EPS:
            out[mask] = 0.0
        else:
            out[mask] = (v - mu) / (std + EPS)
    return out


def compute_cross_sectional_features(
    ts_features: Dict[int, pd.DataFrame],
    active_membership: pd.DataFrame,
) -> Dict[int, pd.DataFrame]:
    """
    Compute cross-sectional features for all dates and all active securities.

    Args:
        ts_features     : {security_id: DataFrame(DatetimeIndex, 17 TS features)}.
                          Only 'ret_4w', 'ret_13w', 'vol_4w', 'volume_z_4w' are used.
        active_membership: DataFrame with columns [date, security_id, sector_code].
                           Contains one row per (date, active_security) using
                           the as-of membership rule from Phase 2.

    Returns:
        {security_id: DataFrame(DatetimeIndex, 8 CS feature columns)}.
        Dates not in active_membership for a security will have NaN.
    """
    # Build output containers: sid → list of (date, cs_dict)
    results: Dict[int, list] = {sid: [] for sid in ts_features}

    # Process date by date
    for date, grp in active_membership.groupby("date", sort=True):
        active_sids    = grp["security_id"].values.astype(int)
        sector_codes   = grp["sector_code"].values.astype(str)

        # Gather raw values for this date from pre-computed TS features
        ret_4w_vals    = np.full(len(active_sids), np.nan)
        ret_13w_vals   = np.full(len(active_sids), np.nan)
        vol_4w_vals    = np.full(len(active_sids), np.nan)
        volume_z_vals  = np.full(len(active_sids), np.nan)

        for i, sid in enumerate(active_sids):
            if sid not in ts_features:
                continue
            feat_df = ts_features[sid]
            if date not in feat_df.index:
                continue
            row = feat_df.loc[date]
            ret_4w_vals[i]   = row.get("ret_4w",    np.nan)
            ret_13w_vals[i]  = row.get("ret_13w",   np.nan)
            vol_4w_vals[i]   = row.get("vol_4w",    np.nan)
            volume_z_vals[i] = row.get("volume_z_4w", np.nan)

        # --- Cross-sectional computations --------------------------------
        ret_rank_4w            = _cs_rank(ret_4w_vals)
        ret_z_4w               = _cs_zscore(ret_4w_vals)
        ret_z_13w              = _cs_zscore(ret_13w_vals)
        vol_z_4w               = _cs_zscore(vol_4w_vals)
        volume_z_cs_4w         = _cs_zscore(volume_z_vals)
        ret_z_4w_sector        = _sector_zscore(ret_4w_vals, sector_codes)
        vol_z_4w_sector        = _sector_zscore(vol_4w_vals, sector_codes)

        # Sector-mean residual standardised by sector std
        sector_mean_ret = np.zeros_like(ret_4w_vals, dtype=float)
        sector_std_ret  = np.ones_like(ret_4w_vals,  dtype=float)
        for sector in np.unique(sector_codes):
            mask   = sector_codes == sector
            v      = ret_4w_vals[mask].astype(float)
            finite = v[np.isfinite(v)]
            if len(finite) >= 2:
                sector_mean_ret[mask] = float(finite.mean())
                sector_std_ret[mask]  = max(float(finite.std()), EPS)
            elif len(finite) == 1:
                sector_mean_ret[mask] = float(finite[0])
        momentum_sector_residual = (ret_4w_vals - sector_mean_ret) / sector_std_ret

        # --- Store results per security -----------------------------------
        for i, sid in enumerate(active_sids):
            if sid not in results:
                results[sid] = []
            results[sid].append((
                date,
                {
                    "ret_rank_4w":            ret_rank_4w[i],
                    "ret_z_4w":               ret_z_4w[i],
                    "ret_z_13w":              ret_z_13w[i],
                    "vol_z_4w":               vol_z_4w[i],
                    "volume_z_cs_4w":         volume_z_cs_4w[i],
                    "ret_z_4w_sector":        ret_z_4w_sector[i],
                    "vol_z_4w_sector":        vol_z_4w_sector[i],
                    "momentum_sector_residual": momentum_sector_residual[i],
                },
            ))

    # Convert to DataFrames
    cs_dataframes: Dict[int, pd.DataFrame] = {}
    for sid, records in results.items():
        if not records:
            continue
        dates = [r[0] for r in records]
        rows  = [r[1] for r in records]
        cs_dataframes[sid] = pd.DataFrame(rows, index=pd.DatetimeIndex(dates))
        cs_dataframes[sid].index.name = "date"

    return cs_dataframes
