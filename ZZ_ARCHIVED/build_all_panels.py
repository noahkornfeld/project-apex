"""
Build All Feature Panels — one per fold from master_config.yaml
================================================================
Computes fold-independent parts (x_panel, mask_panel, active_ids) ONCE,
then generates fold-specific g_panel for each fold's normalization boundary.

Output structure:
    Ticker_Data/panels/
        shared/
            feature_panel_shared.npz   # x_panel, mask_panel, active_ids, dates
        fold_1/
            g_panel.npz                # g_panel normalized with train_end
        fold_2/
            g_panel.npz
        ...
        fold_8/
            g_panel.npz

Usage:
    python scripts/build_all_panels.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
from features.normalizers import CausalPerAssetNormalizer
from features.feature_panel import (
    _build_asof_membership,
    _pack_day_into_slot,
    build_tradeability_lookup,
    TS_FEATURE_NAMES,
    CS_FEATURE_NAMES,
    ALL_ASSET_FEATURE_NAMES,
    F_TS, F_CS, F_TOTAL,
    D_MACRO, D_BENCHMARK, D_PORT, D_GLOBAL,
)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_folds_from_config(config_path: Path) -> list:
    """Read fold definitions from master_config.yaml."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    folds_raw = cfg["evaluation"]["folds"]
    folds = []
    for fd in folds_raw:
        folds.append({
            "fold":        fd["fold"],
            "train_start": fd["train_start"],
            "train_end":   fd["train_end"],
            "test_start":  fd["test_start"],
            "test_end":    fd["test_end"],
        })
    return folds


def load_architecture_from_config(config_path: Path) -> dict:
    """Read K_max and clip from master_config.yaml."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    arch = cfg["architecture"]
    feat = cfg["features"]
    return {
        "K_max": arch["K_max"],
        "clip":  feat["norm_clip_threshold"],
        "norm_window_weeks": feat["norm_window_weeks"],
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir: Path):
    """Load all required parquets."""
    daily_bars  = pd.read_parquet(data_dir / "daily_bars.parquet")
    ndx_df      = pd.read_parquet(data_dir / "ndx_membership.parquet")
    macro_df    = pd.read_parquet(data_dir / "macro_features.parquet")
    calendar_df = pd.read_parquet(data_dir / "trading_calendar.parquet")

    daily_bars["date"]  = pd.to_datetime(daily_bars["date"])
    ndx_df["date"]      = pd.to_datetime(ndx_df["date"])
    if "date" in macro_df.columns:
        macro_df["date"] = pd.to_datetime(macro_df["date"])
    calendar_df["date"] = pd.to_datetime(calendar_df["date"])

    return daily_bars, ndx_df, macro_df, calendar_df


# ---------------------------------------------------------------------------
# Build shared (fold-independent) parts
# ---------------------------------------------------------------------------

def build_shared(
    daily_bars: pd.DataFrame,
    ndx_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    K_max: int,
    norm_window_weeks: int,
    clip: float,
) -> dict:
    """
    Compute everything that is identical across folds:
        - x_panel   [T, K_max, F]
        - mask_panel [T, K_max]
        - active_ids [T, K_max]
        - dates      [T]

    Also returns raw macro/benchmark DataFrames for fold-specific normalization.
    """
    # 0. Tradeability lookup (§2.4.1 gate)
    print("[0/5] Building §2.4.1 tradeability lookup...")
    tradeable_lookup = build_tradeability_lookup(daily_bars)
    excluded = sum(1 for v in tradeable_lookup.values() if not v)
    print(f"      Non-tradeable asset-days (gated): {excluded:,}")
    t0 = time.time()

    # 1. As-of membership
    print("[1/5] Building as-of membership (§2.5)...")
    asof_membership = _build_asof_membership(ndx_df, calendar_df)
    print(f"      Active-membership rows: {len(asof_membership):,}")

    # 2. QQQ series for per-asset features
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

    # 3. Per-asset TS features (§3.1)
    print("[2/5] Computing §3.1 per-asset TS features (all securities)...")
    ts_raw = compute_all_securities_features(daily_bars, qqq_close, qqq_log_ret)
    print(f"      Features computed for {len(ts_raw)} securities")

    # 4. Per-asset causal normalization (§3.6.1) — fold-independent
    print("[3/5] Applying causal per-asset normalization (§3.6.1)...")
    norm_window_days = norm_window_weeks * 5
    normalizer = CausalPerAssetNormalizer(window=norm_window_days, clip=clip)
    ts_norm = {}
    for sid, feat_df in ts_raw.items():
        ts_norm[sid] = normalizer.fit_transform(feat_df)

    # 5. Cross-sectional features (§3.2) — fold-independent
    print("[4/5] Computing §3.2 cross-sectional features...")
    cs_feat = compute_cross_sectional_features(ts_raw, asof_membership)
    print(f"      CS features computed for {len(cs_feat)} securities")

    # 6. Pack x_panel, mask_panel, active_ids
    print("[5/5] Assembling x_panel [T, K_max, F]...")
    trading_days = calendar_df["date"].sort_values().values
    T = len(trading_days)

    x_panel    = np.zeros((T, K_max, F_TOTAL), dtype=np.float32)
    mask_panel = np.zeros((T, K_max),           dtype=np.float32)
    active_ids = np.full((T, K_max), -1,        dtype=np.int64)

    # Pre-group membership by date for fast lookup
    membership_by_date = {}
    for date, grp in asof_membership.groupby("date"):
        membership_by_date[pd.Timestamp(date)] = grp["security_id"].values.astype(int)

    for t_idx, tday in enumerate(trading_days):
        tday_ts = pd.Timestamp(tday)
        active_sids = membership_by_date.get(tday_ts, np.array([], dtype=int))
        if len(active_sids) > 0:
            x_day, mask_day, ids_day = _pack_day_into_slot(
                tday_ts, active_sids, ts_norm, cs_feat, K_max, clip, tradeable_lookup,
            )
            x_panel[t_idx]    = x_day
            mask_panel[t_idx] = mask_day
            active_ids[t_idx] = ids_day

        if (t_idx + 1) % 1000 == 0:
            print(f"      Packed {t_idx + 1:,}/{T:,} days...")

    dates = pd.DatetimeIndex(trading_days)

    elapsed = time.time() - t0
    active_per_day = mask_panel.sum(axis=1)
    print(f"\n      Shared panel complete in {elapsed:.1f}s")
    print(f"      x_panel shape    : {x_panel.shape}")
    print(f"      Mean active/day  : {active_per_day.mean():.1f}")
    print(f"      Max  active/day  : {active_per_day.max():.0f}")
    print(f"      x_panel max |v|  : {np.abs(x_panel).max():.4f}  (clip={clip})")

    # 7. Raw macro/benchmark features (not yet normalized — fold-specific)
    print("\n      Computing raw macro & benchmark features...")
    macro_raw = compute_macro_broadcast_features(macro_df)
    bench_raw = compute_benchmark_features(macro_df)

    return {
        "x_panel":    x_panel,
        "mask_panel": mask_panel,
        "active_ids": active_ids,
        "dates":      dates,
        "trading_days": trading_days,
        "macro_raw":  macro_raw,
        "bench_raw":  bench_raw,
    }


# ---------------------------------------------------------------------------
# Build fold-specific g_panel
# ---------------------------------------------------------------------------

def build_g_panel_for_fold(
    shared: dict,
    norm_fit_end_date: str,
    clip: float,
) -> np.ndarray:
    """
    Build fold-specific g_panel [T, D_global] by normalizing macro/benchmark
    features using statistics fitted only on data up to norm_fit_end_date.
    """
    macro_raw = shared["macro_raw"]
    bench_raw = shared["bench_raw"]
    T         = len(shared["trading_days"])

    macro_norm = normalize_macro_features(macro_raw, norm_fit_end_date, clip)
    bench_norm = normalize_benchmark_features(bench_raw, norm_fit_end_date, clip)
    port_stub  = compute_portfolio_state_stub(n_dates=T)

    g_panel = np.zeros((T, D_GLOBAL), dtype=np.float32)

    macro_idx = macro_norm.index
    bench_idx = bench_norm.index

    for t_idx, tday in enumerate(shared["trading_days"]):
        tday_ts = pd.Timestamp(tday)

        macro_row = (
            macro_norm.loc[tday_ts].values.astype(np.float32)
            if tday_ts in macro_idx
            else np.zeros(D_MACRO, dtype=np.float32)
        )
        bench_row = (
            bench_norm.loc[tday_ts].values.astype(np.float32)
            if tday_ts in bench_idx
            else np.zeros(D_BENCHMARK, dtype=np.float32)
        )
        port_row = port_stub[t_idx]

        g_panel[t_idx] = np.concatenate([macro_row, bench_row, port_row])

    return g_panel


# ---------------------------------------------------------------------------
# Save utilities
# ---------------------------------------------------------------------------

def save_shared(shared: dict, out_dir: Path):
    """Save fold-independent arrays."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "feature_panel_shared.npz",
        x_panel=shared["x_panel"],
        mask_panel=shared["mask_panel"],
        active_ids=shared["active_ids"],
        dates=shared["dates"].astype(str),
    )
    print(f"  Shared panel saved to {out_dir / 'feature_panel_shared.npz'}")


def save_fold_g_panel(g_panel: np.ndarray, fold_dir: Path):
    """Save fold-specific g_panel."""
    fold_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fold_dir / "g_panel.npz", g_panel=g_panel)


def save_complete_fold(shared: dict, g_panel: np.ndarray, fold_dir: Path):
    """Save a complete fold panel (all arrays in one file)."""
    fold_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        fold_dir / "feature_panel.npz",
        x_panel=shared["x_panel"],
        g_panel=g_panel,
        mask_panel=shared["mask_panel"],
        active_ids=shared["active_ids"],
        dates=shared["dates"].astype(str),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config_path = PROJECT_ROOT / "config" / "master_config.yaml"
    data_dir    = PROJECT_ROOT / "Ticker_Data"
    panels_dir  = data_dir / "panels_v2"

    print("=" * 70)
    print("  BUILD ALL FEATURE PANELS")
    print("=" * 70)

    # Load config
    folds = load_folds_from_config(config_path)
    arch  = load_architecture_from_config(config_path)
    K_max = arch["K_max"]
    clip  = arch["clip"]
    norm_window_weeks = arch["norm_window_weeks"]

    print(f"\n  Config loaded:")
    print(f"    K_max             : {K_max}")
    print(f"    clip              : {clip}")
    print(f"    norm_window_weeks : {norm_window_weeks}")
    print(f"    Folds             : {len(folds)}")
    for fd in folds:
        print(f"      Fold {fd['fold']}: train {fd['train_start']}..{fd['train_end']}  "
              f"test {fd['test_start']}..{fd['test_end']}")

    # Load data
    print(f"\n{'=' * 70}")
    print("  STEP 1: Load Data")
    print(f"{'=' * 70}")
    daily_bars, ndx_df, macro_df, calendar_df = load_data(data_dir)
    print(f"  daily_bars  : {len(daily_bars):,} rows")
    print(f"  membership  : {len(ndx_df):,} rows")
    print(f"  macro       : {len(macro_df):,} rows")
    print(f"  calendar    : {len(calendar_df):,} trading days")

    # Build shared (fold-independent)
    print(f"\n{'=' * 70}")
    print("  STEP 2: Build Shared Panel (fold-independent)")
    print(f"{'=' * 70}")
    t_total = time.time()
    shared = build_shared(
        daily_bars, ndx_df, macro_df, calendar_df,
        K_max, norm_window_weeks, clip,
    )

    # Save shared panel
    save_shared(shared, panels_dir / "shared")

    # Build fold-specific g_panels
    print(f"\n{'=' * 70}")
    print("  STEP 3: Build Fold-Specific g_panels")
    print(f"{'=' * 70}")

    for fd in folds:
        fold_num  = fd["fold"]
        train_end = fd["train_end"]
        fold_name = f"fold_{fold_num}"

        print(f"\n  --- Fold {fold_num} (norm fitted up to {train_end}) ---")
        t_fold = time.time()

        g_panel = build_g_panel_for_fold(shared, train_end, clip)

        fold_dir = panels_dir / fold_name

        # Save fold-specific g_panel
        save_fold_g_panel(g_panel, fold_dir)

        # Also save complete panel for convenience
        save_complete_fold(shared, g_panel, fold_dir)

        elapsed = time.time() - t_fold
        print(f"    g_panel shape  : {g_panel.shape}")
        print(f"    g_panel max |v|: {np.abs(g_panel).max():.4f}")
        print(f"    Saved to {fold_dir / 'feature_panel.npz'}")
        print(f"    Time: {elapsed:.1f}s")

    # Summary
    total_elapsed = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"  ALL PANELS BUILT SUCCESSFULLY")
    print(f"{'=' * 70}")
    print(f"  Total time     : {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Folds built    : {len(folds)}")
    print(f"  Output dir     : {panels_dir}")
    print(f"\n  Structure:")
    print(f"    {panels_dir}/shared/feature_panel_shared.npz  (x, mask, ids, dates)")
    for fd in folds:
        fn = f"fold_{fd['fold']}"
        print(f"    {panels_dir}/{fn}/feature_panel.npz  (complete)")
        print(f"    {panels_dir}/{fn}/g_panel.npz         (g_panel only)")

    print(f"\n  Loading example:")
    print(f"    panel = np.load('Ticker_Data/panels/fold_1/feature_panel.npz')")
    print(f"    x = panel['x_panel']       # [{shared['x_panel'].shape[0]}, {K_max}, {F_TOTAL}]")
    print(f"    g = panel['g_panel']       # [{shared['x_panel'].shape[0]}, {D_GLOBAL}]")
    print(f"    mask = panel['mask_panel'] # [{shared['x_panel'].shape[0]}, {K_max}]")


if __name__ == "__main__":
    main()
