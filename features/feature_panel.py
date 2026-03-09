"""
Feature Panel Builder — Bible §3 / §4.1
=========================================
Orchestrates full [T, K_max, F] feature panel precomputation.

Output tensors:
    x_panel       : [T, K_max, F]  float32  — per-asset features (normalized, clipped)
    g_panel       : [T, D_global]  float32  — global context (macro + benchmark + port-state)
    mask_panel    : [T, K_max]     float32  — 1.0=tradeable, 0.0=inactive/padded
    active_ids    : [T, K_max]     int64    — security_id per slot; -1=inactive

F = 25 (17 TS + 8 CS)
D_global = 9 macro + 3 benchmark + 8 portfolio-state = 20

Causality guarantees:
    - All TS features computed from rolling windows (past data only)
    - CS features computed at each t from the active cross-section at t
    - Causal per-asset normalizer uses running stats over past 260 days
    - As-of rule: membership at t = most recent snapshot with date ≤ t
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.per_asset_features import compute_all_securities_features
from features.cross_sectional_features import compute_cross_sectional_features
from features.macro_broadcast_features import (
    compute_macro_broadcast_features,
    normalize_macro_features,
    MACRO_FEATURE_NAMES,
)
from features.benchmark_features import (
    compute_benchmark_features,
    normalize_benchmark_features,
    BENCHMARK_FEATURE_NAMES,
)
from features.portfolio_state_features import (
    compute_portfolio_state_stub,
    PORTFOLIO_STATE_FEATURE_NAMES,
)
from features.normalizers import CausalPerAssetNormalizer, clip_features

TS_FEATURE_NAMES = [
    "open", "close", "volume", "log_ret",
    "ret_1w", "ret_4w", "ret_12w",
    "vol_1w", "vol_4w", "vol_52w",
    "volume_z_4w", "beta_26w_mkt", "rel_strength_4w",
    "vol_ratio_1w_4w", "RSI_14",
    "bollinger_percent_b", "bollinger_bandwidth",
]
CS_FEATURE_NAMES = [
    "ret_rank_4w", "ret_z_4w", "ret_z_12w", "vol_z_4w",
    "volume_z_cs_4w", "ret_z_4w_sector", "vol_z_4w_sector",
    "momentum_sector_residual",
]
ALL_ASSET_FEATURE_NAMES = TS_FEATURE_NAMES + CS_FEATURE_NAMES

F_TS    = len(TS_FEATURE_NAMES)   # 17
F_CS    = len(CS_FEATURE_NAMES)   # 8
F_TOTAL = F_TS + F_CS              # 25

D_MACRO     = len(MACRO_FEATURE_NAMES)          # 9
D_BENCHMARK = len(BENCHMARK_FEATURE_NAMES)      # 3
D_PORT      = len(PORTFOLIO_STATE_FEATURE_NAMES) # 8
D_GLOBAL    = D_MACRO + D_BENCHMARK + D_PORT    # 20


# ---------------------------------------------------------------------------
# Tradeability lookup helper
# ---------------------------------------------------------------------------

def build_tradeability_lookup(daily_bars: pd.DataFrame) -> dict:
    """
    Build a (security_id, date) -> bool lookup for §2.4.1 tradeability.

    An asset is tradeable on date t only if ALL of the following hold:
        - close(t)  is finite and > 0
        - volume(t) is finite and > 0
        - open(t+1) is finite and > 0  (next row in this security's bars)
    """
    bars = daily_bars[["security_id", "date", "open", "close", "volume"]].copy()
    bars["date"] = pd.to_datetime(bars["date"])
    bars = bars.sort_values(["security_id", "date"]).reset_index(drop=True)

    bars["next_open"] = bars.groupby("security_id")["open"].shift(-1)

    bars["is_tradeable"] = (
        bars["close"].notna()    & (bars["close"]    > 0) &
        bars["volume"].notna()   & (bars["volume"]   > 0) &
        bars["next_open"].notna() & (bars["next_open"] > 0)
    )

    return dict(zip(
        zip(bars["security_id"].astype(int).tolist(),
            pd.DatetimeIndex(bars["date"]).tolist()),
        bars["is_tradeable"].tolist(),
    ))


# ---------------------------------------------------------------------------
# As-of membership helper
# ---------------------------------------------------------------------------

def _build_asof_membership(
    ndx_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a daily active membership table using the as-of rule (§2.5).

    For each trading day t, active members = those in the most recent
    ndx_membership snapshot with date ≤ t.  No forward fill ever.

    Returns:
        DataFrame with columns [date, security_id, sector_code].
    """
    ndx_df       = ndx_df.copy()
    ndx_df["date"] = pd.to_datetime(ndx_df["date"])
    snapshot_dates = ndx_df["date"].drop_duplicates().sort_values().values

    calendar_df   = calendar_df.copy()
    calendar_df["date"] = pd.to_datetime(calendar_df["date"])
    trading_days  = calendar_df["date"].sort_values().values

    rows = []
    snap_idx = 0  # pointer into snapshot_dates

    for tday in trading_days:
        # Advance snapshot pointer: find most recent snapshot ≤ tday
        while (snap_idx + 1 < len(snapshot_dates)
               and snapshot_dates[snap_idx + 1] <= tday):
            snap_idx += 1

        if snap_idx < 0 or snapshot_dates[snap_idx] > tday:
            continue  # no snapshot yet → skip this day

        snap_date = snapshot_dates[snap_idx]
        members   = ndx_df[ndx_df["date"] == snap_date][
            ["security_id", "sector_code"]
        ].copy()
        members["date"] = tday
        rows.append(members)

    if not rows:
        return pd.DataFrame(columns=["date", "security_id", "sector_code"])

    result = pd.concat(rows, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    return result


# ---------------------------------------------------------------------------
# Panel assembler per time step
# ---------------------------------------------------------------------------

def _pack_day_into_slot(
    date: pd.Timestamp,
    active_sids: np.ndarray,
    ts_norm: Dict[int, pd.DataFrame],
    cs_feat: Dict[int, pd.DataFrame],
    K_max: int,
    clip: float,
    tradeable_lookup: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pack one trading day's features into [K_max, F], mask [K_max], ids [K_max].

    Slot assignment: sorted by security_id for determinism.
    If tradeable_lookup is provided, only assets passing §2.4.1 tradeability
    (valid close, volume, and next-day open) are assigned slots.
    Inactive / non-tradeable slots → 0.0 (features), 0.0 (mask), -1 (ids).
    """
    x_day    = np.zeros((K_max, F_TOTAL), dtype=np.float32)
    mask_day = np.zeros(K_max, dtype=np.float32)
    ids_day  = np.full(K_max, -1, dtype=np.int64)

    active_sids_sorted = sorted(active_sids)
    slot = 0

    for sid in active_sids_sorted:
        if slot >= K_max:
            break

        # --- §2.4.1 tradeability gate ------------------------------------
        if tradeable_lookup is not None:
            if not tradeable_lookup.get((int(sid), date), False):
                continue

        # --- TS features (17) --------------------------------------------
        ts_vals = np.zeros(F_TS, dtype=np.float32)
        if sid in ts_norm and date in ts_norm[sid].index:
            row = ts_norm[sid].loc[date]
            for fi, feat in enumerate(TS_FEATURE_NAMES):
                v = row.get(feat, 0.0)
                ts_vals[fi] = 0.0 if (v is None or np.isnan(v)) else float(v)

        # --- CS features (8) ---------------------------------------------
        cs_vals = np.zeros(F_CS, dtype=np.float32)
        if sid in cs_feat and date in cs_feat[sid].index:
            row = cs_feat[sid].loc[date]
            for fi, feat in enumerate(CS_FEATURE_NAMES):
                v = row.get(feat, 0.0)
                cs_vals[fi] = 0.0 if (v is None or np.isnan(v)) else float(v)

        all_vals = np.concatenate([ts_vals, cs_vals])
        all_vals = np.clip(all_vals, -clip, clip)

        x_day[slot]    = all_vals
        mask_day[slot] = 1.0
        ids_day[slot]  = sid
        slot += 1

    return x_day, mask_day, ids_day


# ---------------------------------------------------------------------------
# Main builder class
# ---------------------------------------------------------------------------

class FeaturePanelBuilder:
    """
    Builds the full [T, K_max, F] feature panel for Project Apex (§3, §4.1).

    Usage:
        builder = FeaturePanelBuilder(data_dir="Ticker_Data")
        result  = builder.build(norm_fit_end_date="2023-12-31")
        # result keys: x_panel, g_panel, mask_panel, active_ids, dates
    """

    def __init__(
        self,
        data_dir: str = "Ticker_Data",
        K_max: int = 110,
        norm_window_weeks: int = 52,
        clip: float = 4.0,
    ):
        self.data_dir          = Path(data_dir)
        self.K_max             = K_max
        self.norm_window_days  = norm_window_weeks * 5
        self.clip              = clip

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self):
        daily_bars   = pd.read_parquet(self.data_dir / "daily_bars.parquet")
        ndx_df       = pd.read_parquet(self.data_dir / "ndx_membership.parquet")
        macro_df     = pd.read_parquet(self.data_dir / "macro_features.parquet")
        calendar_df  = pd.read_parquet(self.data_dir / "trading_calendar.parquet")

        daily_bars["date"] = pd.to_datetime(daily_bars["date"])
        ndx_df["date"]     = pd.to_datetime(ndx_df["date"])
        if "date" in macro_df.columns:
            macro_df["date"] = pd.to_datetime(macro_df["date"])
        calendar_df["date"] = pd.to_datetime(calendar_df["date"])

        return daily_bars, ndx_df, macro_df, calendar_df

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        norm_fit_end_date: Optional[str] = None,
        verbose: bool = True,
    ) -> dict:
        """
        Precompute the full [T, K_max, F] panel.

        Args:
            norm_fit_end_date : ISO date string.  Fixed-scale macro/benchmark
                                normalizers are fitted on data up to this date.
                                Per-asset causal normalizer is always causal.
            verbose           : Print progress to stdout.

        Returns dict with keys:
            x_panel     : np.ndarray [T, K_max, F]  float32
            g_panel     : np.ndarray [T, D_global]  float32
            mask_panel  : np.ndarray [T, K_max]     float32
            active_ids  : np.ndarray [T, K_max]     int64
            dates       : pd.DatetimeIndex           length T
            feature_names_x : list[str]  length F (asset features)
            feature_names_g : list[str]  length D_global (global features)
        """
        if verbose:
            print("=" * 60)
            print("FEATURE PANEL BUILDER — Phase 3")
            print("=" * 60)

        # 1. Load data
        if verbose:
            print("\n[1/6] Loading parquet files...")
        daily_bars, ndx_df, macro_df, calendar_df = self._load_data()
        tradeable_lookup = build_tradeability_lookup(daily_bars)
        if verbose:
            print(f"    daily_bars  : {len(daily_bars):,} rows")
            excluded = sum(1 for v in tradeable_lookup.values() if not v)
            print(f"    Non-tradeable asset-days (gated): {excluded:,}")
            print(f"    membership  : {len(ndx_df):,} rows")
            print(f"    macro       : {len(macro_df):,} rows")
            print(f"    calendar    : {len(calendar_df):,} trading days")

        # 2. Build as-of membership
        if verbose:
            print("\n[2/6] Building as-of membership (§2.5)...")
        asof_membership = _build_asof_membership(ndx_df, calendar_df)
        if verbose:
            print(f"    Active-membership rows: {len(asof_membership):,}")

        # 3. Compute QQQ series for per-asset features
        macro_idx = macro_df.set_index("date") if "date" in macro_df.columns else macro_df.copy()
        macro_idx.index = pd.to_datetime(macro_idx.index)

        qqq_close_col = next(
            (c for c in ["QQQ_Close", "QQQ_close"] if c in macro_idx.columns), None
        )
        qqq_lr_col = next(
            (c for c in ["QQQ_log_return", "QQQ_log_ret"] if c in macro_idx.columns), None
        )
        qqq_close   = macro_idx[qqq_close_col].astype(float) if qqq_close_col else pd.Series(dtype=float)
        qqq_log_ret = (
            macro_idx[qqq_lr_col].astype(float) if qqq_lr_col
            else np.log(qqq_close / qqq_close.shift(1))
        )

        # 4. Compute per-asset TS features for all securities
        if verbose:
            print("\n[3/6] Computing §3.1 per-asset TS features...")
        ts_raw = compute_all_securities_features(daily_bars, qqq_close, qqq_log_ret)
        if verbose:
            print(f"    Features computed for {len(ts_raw)} securities")

        # 5. Apply per-asset causal normalization (§3.6.1)
        if verbose:
            print("\n[4/6] Applying causal per-asset normalization (§3.6.1)...")
        normalizer = CausalPerAssetNormalizer(
            window=self.norm_window_days,
            clip=self.clip,
        )
        ts_norm: Dict[int, pd.DataFrame] = {}
        for sid, feat_df in ts_raw.items():
            ts_norm[sid] = normalizer.fit_transform(feat_df)

        # 6. Compute cross-sectional features (§3.2)
        if verbose:
            print("\n[5/6] Computing §3.2 cross-sectional features...")
        cs_feat = compute_cross_sectional_features(ts_raw, asof_membership)
        if verbose:
            print(f"    CS features computed for {len(cs_feat)} securities")

        # 7. Macro + benchmark global features
        if verbose:
            print("\n[6/6] Computing global context g_t (§3.3 + §3.5 + §3.4)...")
        macro_raw   = compute_macro_broadcast_features(macro_df)
        bench_raw   = compute_benchmark_features(macro_df)

        macro_norm  = normalize_macro_features(macro_raw,  norm_fit_end_date, self.clip)
        bench_norm  = normalize_benchmark_features(bench_raw, norm_fit_end_date, self.clip)

        # 8. Pack into panel arrays
        trading_days = calendar_df["date"].sort_values().values
        T            = len(trading_days)

        x_panel    = np.zeros((T, self.K_max, F_TOTAL), dtype=np.float32)
        g_panel    = np.zeros((T, D_GLOBAL),             dtype=np.float32)
        mask_panel = np.zeros((T, self.K_max),            dtype=np.float32)
        active_ids = np.full((T, self.K_max), -1,         dtype=np.int64)

        port_stub = compute_portfolio_state_stub(n_dates=T)  # [T, 8] zeros

        # Build set of active sids per date for fast lookup
        membership_by_date: Dict[pd.Timestamp, np.ndarray] = {}
        for date, grp in asof_membership.groupby("date"):
            membership_by_date[pd.Timestamp(date)] = grp["security_id"].values.astype(int)

        macro_idx2    = macro_norm.index
        bench_idx2    = bench_norm.index

        for t_idx, tday in enumerate(trading_days):
            tday_ts = pd.Timestamp(tday)

            # --- Per-asset features → x_panel[t] -------------------------
            active_sids = membership_by_date.get(tday_ts, np.array([], dtype=int))
            if len(active_sids) > 0:
                x_day, mask_day, ids_day = _pack_day_into_slot(
                    tday_ts, active_sids, ts_norm, cs_feat,
                    self.K_max, self.clip, tradeable_lookup,
                )
                x_panel[t_idx]    = x_day
                mask_panel[t_idx] = mask_day
                active_ids[t_idx] = ids_day

            # --- Global context → g_panel[t] ------------------------------
            macro_row = (
                macro_norm.loc[tday_ts].values.astype(np.float32)
                if tday_ts in macro_idx2
                else np.zeros(D_MACRO, dtype=np.float32)
            )
            bench_row = (
                bench_norm.loc[tday_ts].values.astype(np.float32)
                if tday_ts in bench_idx2
                else np.zeros(D_BENCHMARK, dtype=np.float32)
            )
            port_row = port_stub[t_idx]  # [8] zeros

            g_panel[t_idx] = np.concatenate([macro_row, bench_row, port_row])

        dates = pd.DatetimeIndex(trading_days)

        if verbose:
            print(f"\n{'='*60}")
            print("PANEL PRECOMPUTATION COMPLETE")
            print(f"{'='*60}")
            print(f"  x_panel shape    : {x_panel.shape}")
            print(f"  g_panel shape    : {g_panel.shape}")
            print(f"  mask_panel shape : {mask_panel.shape}")
            print(f"  active_ids shape : {active_ids.shape}")
            active_per_day = mask_panel.sum(axis=1)
            print(f"  Mean active assets/day : {active_per_day.mean():.1f}")
            print(f"  Max  active assets/day : {active_per_day.max():.0f}")
            print(f"  x_panel max abs value  : {np.abs(x_panel).max():.4f}  (clip={self.clip})")

        feature_names_g = MACRO_FEATURE_NAMES + BENCHMARK_FEATURE_NAMES + PORTFOLIO_STATE_FEATURE_NAMES

        return {
            "x_panel":        x_panel,
            "g_panel":        g_panel,
            "mask_panel":     mask_panel,
            "active_ids":     active_ids,
            "dates":          dates,
            "feature_names_x": ALL_ASSET_FEATURE_NAMES,
            "feature_names_g": feature_names_g,
        }

    def save(self, panel_dict: dict, out_dir: str):
        """Save panel arrays to compressed numpy format."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path / "feature_panel.npz",
            x_panel=panel_dict["x_panel"],
            g_panel=panel_dict["g_panel"],
            mask_panel=panel_dict["mask_panel"],
            active_ids=panel_dict["active_ids"],
            dates=panel_dict["dates"].astype(str),
        )
        print(f"Panel saved to {out_path / 'feature_panel.npz'}")

    @staticmethod
    def load(npz_path: str) -> dict:
        """Load a saved panel from disk."""
        data = np.load(npz_path, allow_pickle=True)
        return {
            "x_panel":    data["x_panel"],
            "g_panel":    data["g_panel"],
            "mask_panel": data["mask_panel"],
            "active_ids": data["active_ids"],
            "dates":      pd.DatetimeIndex(data["dates"]),
        }


# ---------------------------------------------------------------------------
# Observation reconstruction helper (used at training time)
# ---------------------------------------------------------------------------

def get_observation(
    panel_dict: dict,
    t_idx: int,
    L: int = 60,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct the observation at time step t from the pre-computed panel.
    Returns the L-week lookback window as required by the model (§8.3.1).

    Args:
        panel_dict : Dict returned by FeaturePanelBuilder.build().
        t_idx      : Current time index (0-based).
        L          : Lookback window in weeks (= trading days here).

    Returns:
        x_t       : [L, K_max, F]  float32  — per-asset feature window
        g_t       : [D_global]     float32  — global context at t
        mask_t    : [K_max]        float32  — tradability mask at t
        ids_t     : [K_max]        int64    — security IDs at t
    """
    start = max(0, t_idx - L + 1)
    x_window = panel_dict["x_panel"][start : t_idx + 1]  # [≤L, K_max, F]

    # Pad front with zeros if we're in the warm-up period
    if x_window.shape[0] < L:
        pad = np.zeros(
            (L - x_window.shape[0], x_window.shape[1], x_window.shape[2]),
            dtype=np.float32,
        )
        x_window = np.concatenate([pad, x_window], axis=0)

    return (
        x_window,
        panel_dict["g_panel"][t_idx],
        panel_dict["mask_panel"][t_idx],
        panel_dict["active_ids"][t_idx],
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Phase 3 Feature Panel")
    parser.add_argument("--data_dir",  default="Ticker_Data", help="Path to Ticker_Data folder")
    parser.add_argument("--out_dir",   default="Ticker_Data", help="Output directory for panel")
    parser.add_argument("--norm_end",  default=None,          help="Normalization fit end date (ISO)")
    parser.add_argument("--K_max",     type=int, default=110)
    parser.add_argument("--clip",      type=float, default=4.0)
    args = parser.parse_args()

    builder = FeaturePanelBuilder(
        data_dir=args.data_dir,
        K_max=args.K_max,
        clip=args.clip,
    )
    panel = builder.build(norm_fit_end_date=args.norm_end)
    builder.save(panel, args.out_dir)
