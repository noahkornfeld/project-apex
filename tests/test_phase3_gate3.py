"""
Phase 3 Gate 3 Tests — Feature Engineering
============================================

Testing Milestones (Gate 3):
1. Causality    — Shuffling future bars into feature window changes output;
                  causal implementation does NOT (temporal leakage trap §11.3)
2. Normalizer   — IS-only normalizer differs from full-range normalizer on OOS data
3. Shape        — Panel shape is [T, K_max, F] with correct F count
4. Clipping     — No feature value outside ±clip threshold after normalization
5. Cross-sect   — At each t, z-scores have mean≈0, std≈1 across active assets

Bible Reference: Phase 3 specification (§3, §11.3)
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.per_asset_features import compute_per_asset_features, DAYS_52W
from features.cross_sectional_features import compute_cross_sectional_features, _cs_zscore
from features.normalizers import CausalPerAssetNormalizer, FixedScaleNormalizer, clip_features
from features.macro_broadcast_features import compute_macro_broadcast_features, MACRO_FEATURE_NAMES
from features.benchmark_features import compute_benchmark_features, BENCHMARK_FEATURE_NAMES
from features.portfolio_state_features import (
    compute_portfolio_state_stub,
    N_PORTFOLIO_STATE_FEATURES,
    PORTFOLIO_STATE_FEATURE_NAMES,
)
from features import TS_FEATURE_NAMES, CS_FEATURE_NAMES, F_TS, F_CS, F_TOTAL

DATA_DIR  = Path(__file__).parent.parent / "Ticker_Data"
CLIP      = 4.0
K_MAX     = 110
F_EXPECTED = 25   # 17 TS + 8 CS


# ===========================================================================
# Synthetic data factories
# ===========================================================================

def _make_bars(n_days: int = 300, seed: int = 0) -> pd.DataFrame:
    """Create synthetic daily bars for one security."""
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-02", periods=n_days, freq="B")

    log_ret = rng.normal(0.0005, 0.015, size=n_days)
    prices  = 100.0 * np.exp(np.cumsum(log_ret))

    return pd.DataFrame({
        "open":   prices * rng.uniform(0.995, 1.005, size=n_days),
        "close":  prices,
        "volume": rng.integers(1_000_000, 10_000_000, size=n_days).astype(float),
    }, index=dates)


def _make_qqq(bars: pd.DataFrame, seed: int = 99) -> tuple:
    """Create synthetic QQQ close and log return aligned to bars.index."""
    rng     = np.random.default_rng(seed)
    lr      = rng.normal(0.0004, 0.012, size=len(bars))
    prices  = 350.0 * np.exp(np.cumsum(lr))
    close   = pd.Series(prices, index=bars.index)
    log_ret = pd.Series(lr,    index=bars.index)
    return close, log_ret


def _make_macro_df(n_days: int = 300) -> pd.DataFrame:
    """Create a minimal synthetic macro DataFrame."""
    rng   = np.random.default_rng(42)
    dates = pd.date_range("2018-01-02", periods=n_days, freq="B")

    qqq_close = 350.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n_days)))
    qqq_lr    = np.log(qqq_close / np.roll(qqq_close, 1))
    qqq_lr[0] = 0.0

    return pd.DataFrame({
        "date":                  dates,
        "QQQ_Close":             qqq_close,
        "QQQ_log_return":        qqq_lr,
        "VIX_Close":             rng.uniform(12, 35, n_days),
        "VIX_change":            rng.normal(0, 1.5, n_days),
        "Yield_10Y":             rng.uniform(1.5, 4.5, n_days),
        "Yield_3M":              rng.uniform(0.5, 3.5, n_days),
        "Yield_Spread":          rng.uniform(-0.5, 2.5, n_days),
        "Oil_Close":             rng.uniform(40, 100, n_days),
        "Oil_log_return":        rng.normal(0, 0.02, n_days),
        "Gold_Close":            rng.uniform(1200, 2000, n_days),
        "Gold_log_return":       rng.normal(0, 0.008, n_days),
        "Dollar_Index_Close":    rng.uniform(90, 110, n_days),
        "Dollar_Index_log_return": rng.normal(0, 0.005, n_days),
        "HYG_Close":             rng.uniform(75, 90, n_days),
        "HYG_log_return":        rng.normal(0, 0.005, n_days),
    })


def _make_active_membership(
    sids: list,
    sector_map: dict,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Create a simple active-membership DataFrame for all dates and given sids."""
    rows = []
    for d in dates:
        for sid in sids:
            rows.append({
                "date":        d,
                "security_id": sid,
                "sector_code": sector_map.get(sid, "45"),
            })
    return pd.DataFrame(rows)


# ===========================================================================
# Gate 3.1 — Causality
# ===========================================================================

class TestGate3Causality:
    """
    Gate 3 - Causality: Temporal leakage trap (§11.3)

    MUST BE TRUE: Shuffling future bars into the feature-computation window
    changes the output of a non-causal implementation, but the causal
    implementation produces the same result regardless.
    """

    def test_causal_normalizer_ignores_future(self):
        """
        Causal normalizer at time t uses only data up to t.
        Appending future rows with extreme values must NOT change any t's z-score
        that has already been computed up to that point.
        """
        bars     = _make_bars(n_days=200)
        qqq_c, qqq_lr = _make_qqq(bars)
        feat_df  = compute_per_asset_features(bars, qqq_c, qqq_lr)

        norm = CausalPerAssetNormalizer(window=52, clip=CLIP)
        z_original = norm.fit_transform(feat_df)

        # Append 50 rows with extreme values (future data)
        extra_dates = pd.date_range(feat_df.index[-1] + pd.Timedelta(days=1),
                                    periods=50, freq="B")
        extreme = pd.DataFrame(
            np.full((50, feat_df.shape[1]), 1e6),
            columns=feat_df.columns,
            index=extra_dates,
        )
        feat_extended = pd.concat([feat_df, extreme])

        z_extended = norm.fit_transform(feat_extended)

        # All rows corresponding to original period must be unchanged
        z_orig_vals     = z_original.values
        z_extended_vals = z_extended.loc[feat_df.index].values

        np.testing.assert_allclose(
            z_orig_vals, z_extended_vals, rtol=1e-5, atol=1e-5,
            err_msg="Causal normalizer must not change historical rows when future data is appended",
        )

    def test_rolling_features_are_causal(self):
        """
        Rolling-window TS features (vol_1w, ret_4w, etc.) at time t must use
        only data up to t.  Injecting a future extreme value must NOT change
        any row at or before the original last date.
        """
        bars         = _make_bars(n_days=150)
        qqq_c, qqq_lr = _make_qqq(bars)
        feat_orig    = compute_per_asset_features(bars, qqq_c, qqq_lr)

        # Modify one future (out-of-sample) day — should not touch past rows
        bars_mod = bars.copy()
        future_idx = bars_mod.index[-1] + pd.Timedelta(days=3)
        new_row = pd.DataFrame(
            {"open": [1e6], "close": [1e6], "volume": [1e10]},
            index=[future_idx],
        )
        bars_future = pd.concat([bars_mod, new_row])
        qqq_c_ext   = pd.concat([qqq_c, pd.Series([qqq_c.iloc[-1]], index=[future_idx])])
        qqq_lr_ext  = pd.concat([qqq_lr, pd.Series([0.0], index=[future_idx])])

        feat_future = compute_per_asset_features(bars_future, qqq_c_ext, qqq_lr_ext)

        for col in ["ret_4w", "vol_4w", "RSI_14", "beta_26w_mkt"]:
            orig_vals  = feat_orig[col].dropna().values
            ext_vals   = feat_future.loc[feat_orig.index, col].dropna().values
            np.testing.assert_allclose(
                orig_vals, ext_vals, rtol=1e-5, atol=1e-6,
                err_msg=f"Feature '{col}' changed after appending future data — causality violation!",
            )

    def test_future_shuffle_does_change_noncausal(self):
        """
        Control test: verify that a non-causal (global) z-score DOES change
        when future data is appended.  Confirms leakage trap is meaningful.
        """
        series = pd.Series(np.arange(100, dtype=float))

        global_z_orig  = (series - series.mean()) / series.std()

        extreme_future = pd.Series(np.concatenate([series.values, [1e6] * 20]))
        global_z_ext   = (extreme_future - extreme_future.mean()) / extreme_future.std()

        # Original values must be DIFFERENT when future extreme is included
        orig_in_ext = global_z_ext.iloc[:100].values
        assert not np.allclose(global_z_orig.values, orig_in_ext, atol=1e-3), \
            "Non-causal z-score should change when future data is appended"


# ===========================================================================
# Gate 3.2 — Normalizer Leakage Trap
# ===========================================================================

class TestGate3Normalizer:
    """
    Gate 3 — Normalizer: IS-only normalizer differs from full-range normalizer on OOS data.
    Verifies that FixedScaleNormalizer fitted on IS data produces different
    results than one fitted on full data (IS + OOS) — proving IS-only fitting matters.
    """

    def test_is_normalizer_differs_from_full_range(self):
        """
        Fitting macro normalizer on IS data vs. all data produces different
        OOS normalized values.  This is the Normalizer Leakage Trap (§11.3).
        """
        macro_df = _make_macro_df(n_days=500)
        macro_df = macro_df.set_index("date")
        macro_df.index = pd.to_datetime(macro_df.index)

        # Only the 9 columns we care about
        feat_cols = [
            "VIX_Close", "VIX_change", "Yield_10Y", "Yield_3M", "Yield_Spread",
            "Oil_log_return", "Gold_log_return", "Dollar_Index_log_return", "HYG_log_return",
        ]
        feat_df = macro_df[feat_cols].astype(float)

        split_idx = 350
        is_data   = feat_df.iloc[:split_idx]
        oos_data  = feat_df.iloc[split_idx:]

        # IS-only normalizer (correct)
        norm_is = FixedScaleNormalizer(clip=CLIP)
        norm_is.fit(is_data)
        z_is  = norm_is.transform(oos_data)

        # Full-range normalizer (cheating — uses OOS data in fit)
        norm_full = FixedScaleNormalizer(clip=CLIP)
        norm_full.fit(feat_df)
        z_full = norm_full.transform(oos_data)

        # They must DIFFER — IS stats ≠ full-range stats
        assert not np.allclose(z_is.values, z_full.values, atol=1e-3), \
            "IS-only and full-range normalizers must produce different OOS results"

    def test_causal_normalizer_is_not_global(self):
        """
        The causal per-asset normalizer must use the running mean/std,
        not the global mean/std.  They must differ on any non-stationary series.
        """
        n = 260
        # Trending series: mean shifts over time
        trend  = np.arange(n, dtype=float) * 0.5
        noise  = np.random.default_rng(7).normal(0, 1, n)
        series = trend + noise

        feat_df = pd.DataFrame({"f": series})
        norm    = CausalPerAssetNormalizer(window=52, clip=99.0)
        z       = norm.fit_transform(feat_df)

        global_z = (feat_df - feat_df.mean()) / feat_df.std()

        # Should differ because the series is non-stationary
        max_diff = np.abs(z["f"].values - global_z["f"].values).max()
        assert max_diff > 0.01, \
            "Causal normalizer should differ from global z-score on non-stationary data"


# ===========================================================================
# Gate 3.3 — Shape
# ===========================================================================

class TestGate3Shape:
    """Gate 3 — Shape: Panel shape is [T, K_max, F] with correct F count."""

    def test_ts_feature_count(self):
        """Each security produces exactly 17 TS features."""
        bars         = _make_bars(n_days=100)
        qqq_c, qqq_lr = _make_qqq(bars)
        feat_df      = compute_per_asset_features(bars, qqq_c, qqq_lr)

        assert feat_df.shape[1] == F_TS, \
            f"Expected {F_TS} TS features, got {feat_df.shape[1]}"

    def test_ts_feature_names_match_config(self):
        """Feature column names must match TS_FEATURE_NAMES list exactly."""
        bars         = _make_bars(n_days=100)
        qqq_c, qqq_lr = _make_qqq(bars)
        feat_df      = compute_per_asset_features(bars, qqq_c, qqq_lr)

        assert list(feat_df.columns) == TS_FEATURE_NAMES, \
            f"Column names mismatch.\nExpected: {TS_FEATURE_NAMES}\nGot: {list(feat_df.columns)}"

    def test_cs_feature_count(self):
        """Cross-sectional features produce exactly 8 columns."""
        n_assets = 10
        n_days   = 100
        sids     = list(range(1, n_assets + 1))
        sector_map = {sid: "45" if sid <= 5 else "20" for sid in sids}

        bars_dict = {sid: _make_bars(n_days, seed=sid) for sid in sids}
        common_dates = bars_dict[1].index

        ts_all = {}
        for sid in sids:
            qqq_c, qqq_lr = _make_qqq(bars_dict[sid], seed=99)
            ts_all[sid] = compute_per_asset_features(bars_dict[sid], qqq_c, qqq_lr)

        membership = _make_active_membership(sids, sector_map, common_dates)
        cs_all = compute_cross_sectional_features(ts_all, membership)

        for sid, cs_df in cs_all.items():
            assert cs_df.shape[1] == F_CS, \
                f"Security {sid}: expected {F_CS} CS features, got {cs_df.shape[1]}"

    def test_total_feature_count(self):
        """F_TOTAL must equal 25 (17 TS + 8 CS)."""
        assert F_TS + F_CS == F_EXPECTED, \
            f"Expected F={F_EXPECTED}, got F_TS={F_TS} + F_CS={F_CS} = {F_TS + F_CS}"
        assert F_TOTAL == F_EXPECTED

    def test_macro_feature_count(self):
        """Macro broadcast features must have exactly 9 columns."""
        macro_df = _make_macro_df(200)
        out = compute_macro_broadcast_features(macro_df)
        assert out.shape[1] == len(MACRO_FEATURE_NAMES), \
            f"Expected {len(MACRO_FEATURE_NAMES)} macro features, got {out.shape[1]}"

    def test_benchmark_feature_count(self):
        """Benchmark features must have exactly 3 columns."""
        macro_df = _make_macro_df(200)
        out = compute_benchmark_features(macro_df)
        assert out.shape[1] == len(BENCHMARK_FEATURE_NAMES), \
            f"Expected {len(BENCHMARK_FEATURE_NAMES)} benchmark features, got {out.shape[1]}"

    def test_portfolio_state_stub_shape(self):
        """Portfolio-state stub must return [T, 8] for multi-step, [8] for single."""
        stub_multi  = compute_portfolio_state_stub(n_dates=100)
        stub_single = compute_portfolio_state_stub()

        assert stub_multi.shape  == (100, N_PORTFOLIO_STATE_FEATURES)
        assert stub_single.shape == (N_PORTFOLIO_STATE_FEATURES,)

    def test_portfolio_state_stub_is_zeros(self):
        """All portfolio-state stub values must be 0.0 (§3.4 stub)."""
        stub = compute_portfolio_state_stub(n_dates=50)
        assert np.all(stub == 0.0), "Portfolio-state stub must be all zeros"

    def test_portfolio_state_feature_names_count(self):
        """Portfolio-state feature list must have exactly 8 features."""
        assert len(PORTFOLIO_STATE_FEATURE_NAMES) == 8


# ===========================================================================
# Gate 3.4 — Clipping
# ===========================================================================

class TestGate3Clipping:
    """Gate 3 — Clipping: No feature value outside ±clip_threshold after normalization."""

    def test_causal_normalizer_clips(self):
        """After causal normalization, all values must be in [-CLIP, +CLIP]."""
        bars         = _make_bars(n_days=300)
        qqq_c, qqq_lr = _make_qqq(bars)
        feat_df      = compute_per_asset_features(bars, qqq_c, qqq_lr)

        norm = CausalPerAssetNormalizer(window=52, clip=CLIP)
        z    = norm.fit_transform(feat_df)

        max_val = z.abs().max().max()
        assert max_val <= CLIP + 1e-6, \
            f"Causal normalizer: max abs value {max_val:.4f} exceeds clip {CLIP}"

    def test_fixed_scale_normalizer_clips(self):
        """Fixed-scale macro normalizer must clip output to ±CLIP."""
        macro_df = _make_macro_df(300)
        macro_df_idx = macro_df.set_index("date")
        # Add some extreme outlier rows
        macro_df_idx.iloc[0] *= 1000

        feat_cols = [
            "VIX_Close", "VIX_change", "Yield_10Y", "Yield_3M", "Yield_Spread",
            "Oil_log_return", "Gold_log_return", "Dollar_Index_log_return", "HYG_log_return",
        ]
        feat_df = macro_df_idx[feat_cols].astype(float)

        norm = FixedScaleNormalizer(clip=CLIP)
        z    = norm.fit_transform(feat_df)

        max_val = z.abs().max().max()
        assert max_val <= CLIP + 1e-6, \
            f"FixedScaleNormalizer: max abs value {max_val:.4f} exceeds clip {CLIP}"

    def test_extreme_inputs_are_clipped(self):
        """Features with extreme raw values must be clipped to ±CLIP."""
        bars         = _make_bars(n_days=200)
        # Inject a spike
        bars.iloc[100, bars.columns.get_loc("close")] = 1e8
        qqq_c, qqq_lr = _make_qqq(bars)
        feat_df       = compute_per_asset_features(bars, qqq_c, qqq_lr)

        norm = CausalPerAssetNormalizer(window=52, clip=CLIP)
        z    = norm.fit_transform(feat_df)

        assert z.abs().max().max() <= CLIP + 1e-6

    def test_clip_utility_function(self):
        """clip_features() must clip any DataFrame to ±clip and fill NaN with 0."""
        df  = pd.DataFrame({"a": [1.0, 10.0, -10.0, np.nan], "b": [-5.0, 3.0, 0.0, 2.0]})
        out = clip_features(df, clip=4.0)

        assert out["a"].max()  <=  4.0
        assert out["a"].min()  >= -4.0
        assert out.isna().sum().sum() == 0


# ===========================================================================
# Gate 3.5 — Cross-Sectional Properties
# ===========================================================================

class TestGate3CrossSectional:
    """
    Gate 3 — Cross-sect: At each t, z-scores have mean≈0, std≈1 across active assets.
    """

    def test_cs_zscore_mean_zero_std_one(self):
        """_cs_zscore helper must produce mean≈0 and std≈1 on finite values."""
        rng    = np.random.default_rng(10)
        values = rng.normal(5.0, 3.0, size=50)   # non-zero mean, non-unit std

        z = _cs_zscore(values)

        assert abs(z.mean()) < 0.05,  f"CS z-score mean should be ≈0, got {z.mean():.4f}"
        assert abs(z.std()  - 1.0) < 0.05, f"CS z-score std should be ≈1, got {z.std():.4f}"

    def test_cs_features_mean_zero_at_each_t(self):
        """
        Cross-sectional z-score features (ret_z_4w, ret_z_13w, vol_z_4w)
        must have mean ≈ 0 and std ≈ 1 across active assets at each t.
        """
        n_assets = 30
        n_days   = 200
        sids     = list(range(1, n_assets + 1))
        sector_map = {sid: str(10 * ((sid % 5) + 1)) for sid in sids}

        ts_all = {}
        for sid in sids:
            bars         = _make_bars(n_days, seed=sid)
            qqq_c, qqq_lr = _make_qqq(bars, seed=99)
            ts_all[sid]  = compute_per_asset_features(bars, qqq_c, qqq_lr)

        dates      = ts_all[1].index
        membership = _make_active_membership(sids, sector_map, dates)
        cs_all     = compute_cross_sectional_features(ts_all, membership)

        z_score_cols = ["ret_z_4w", "ret_z_13w", "vol_z_4w"]

        # Check a sample of dates (skip early warm-up; ret_13w needs 65 days)
        check_dates = dates[70:]
        for tday in check_dates[::10]:  # every 10th date for speed
            vals_by_col = {col: [] for col in z_score_cols}
            for sid, cs_df in cs_all.items():
                if tday in cs_df.index:
                    for col in z_score_cols:
                        v = cs_df.loc[tday, col]
                        if np.isfinite(v):
                            vals_by_col[col].append(v)

            for col in z_score_cols:
                vals = np.array(vals_by_col[col])
                if len(vals) < 5:
                    continue
                mean_val = vals.mean()
                std_val  = vals.std()
                assert abs(mean_val) < 0.15, \
                    f"At {tday.date()}, {col} cross-section mean = {mean_val:.4f}, expected ≈ 0"
                assert 0.7 < std_val < 1.3, \
                    f"At {tday.date()}, {col} cross-section std = {std_val:.4f}, expected ≈ 1"

    def test_ret_rank_4w_is_in_unit_interval(self):
        """ret_rank_4w (percentile rank) must be in [0, 1] for all active assets."""
        n_assets = 20
        n_days   = 150
        sids     = list(range(1, n_assets + 1))
        sector_map = {sid: "45" for sid in sids}

        ts_all = {}
        for sid in sids:
            bars        = _make_bars(n_days, seed=sid * 3)
            qqq_c, lr   = _make_qqq(bars, seed=99)
            ts_all[sid] = compute_per_asset_features(bars, qqq_c, lr)

        dates      = ts_all[1].index
        membership = _make_active_membership(sids, sector_map, dates)
        cs_all     = compute_cross_sectional_features(ts_all, membership)

        for sid, cs_df in cs_all.items():
            col = "ret_rank_4w"
            finite_vals = cs_df[col].dropna()
            if len(finite_vals) == 0:
                continue
            assert finite_vals.min() >= -0.01, f"ret_rank_4w below 0 for sid {sid}"
            assert finite_vals.max() <= 1.01,  f"ret_rank_4w above 1 for sid {sid}"

    def test_momentum_sector_residual_sector_mean_zero(self):
        """
        momentum_sector_residual = ret_4w - sector_mean(ret_4w).
        The mean of residuals within each sector must be ≈ 0.
        """
        n_assets = 20
        n_days   = 150
        sids     = list(range(1, n_assets + 1))
        # Two sectors: sids 1-10 → "45", sids 11-20 → "20"
        sector_map = {sid: "45" if sid <= 10 else "20" for sid in sids}

        ts_all = {}
        for sid in sids:
            bars        = _make_bars(n_days, seed=sid * 5)
            qqq_c, lr   = _make_qqq(bars, seed=99)
            ts_all[sid] = compute_per_asset_features(bars, qqq_c, lr)

        dates      = ts_all[1].index
        membership = _make_active_membership(sids, sector_map, dates)
        cs_all     = compute_cross_sectional_features(ts_all, membership)

        tday = dates[100]
        for sector in ["45", "20"]:
            sector_sids = [sid for sid, s in sector_map.items() if s == sector]
            residuals = []
            for sid in sector_sids:
                if sid in cs_all and tday in cs_all[sid].index:
                    v = cs_all[sid].loc[tday, "momentum_sector_residual"]
                    if np.isfinite(v):
                        residuals.append(v)
            if len(residuals) >= 3:
                mean_resid = np.mean(residuals)
                assert abs(mean_resid) < 0.15, \
                    f"Sector {sector}: momentum_sector_residual mean = {mean_resid:.4f}, expected ≈ 0"


# ===========================================================================
# Integration test — real data (skipped if data not available)
# ===========================================================================

class TestGate3Integration:
    """Integration tests on real parquet data (skipped if data unavailable)."""

    @pytest.fixture(autouse=True)
    def check_data(self):
        if not (DATA_DIR / "daily_bars.parquet").exists():
            pytest.skip("Parquet data not found — skipping integration tests")

    def test_real_macro_features_shape(self):
        """Macro features from real data must have exactly 9 columns."""
        macro_df = pd.read_parquet(DATA_DIR / "macro_features.parquet")
        out      = compute_macro_broadcast_features(macro_df)
        assert out.shape[1] == 9
        assert list(out.columns) == MACRO_FEATURE_NAMES

    def test_real_benchmark_features_shape(self):
        """Benchmark features from real data must have exactly 3 columns."""
        macro_df = pd.read_parquet(DATA_DIR / "macro_features.parquet")
        out      = compute_benchmark_features(macro_df)
        assert out.shape[1] == 3
        assert list(out.columns) == BENCHMARK_FEATURE_NAMES

    def test_real_macro_no_nan_after_fill(self):
        """After compute_macro_broadcast_features, no NaN should remain."""
        macro_df = pd.read_parquet(DATA_DIR / "macro_features.parquet")
        out      = compute_macro_broadcast_features(macro_df)
        assert out.isna().sum().sum() == 0, "Macro features must have 0 NaN after fill"

    def test_real_single_security_features(self):
        """A real security must produce a 18-column TS feature DataFrame."""
        daily_bars = pd.read_parquet(DATA_DIR / "daily_bars.parquet")
        macro_df   = pd.read_parquet(DATA_DIR / "macro_features.parquet")

        # Take the security with most rows
        sid = daily_bars.groupby("security_id").size().idxmax()
        bars = daily_bars[daily_bars["security_id"] == sid].set_index("date").sort_index()

        macro_idx    = macro_df.set_index("date") if "date" in macro_df.columns else macro_df
        macro_idx.index = pd.to_datetime(macro_idx.index)

        qqq_lr_col  = next((c for c in ["QQQ_log_return", "QQQ_log_ret"] if c in macro_idx.columns), None)
        qqq_c_col   = next((c for c in ["QQQ_Close", "QQQ_close"] if c in macro_idx.columns), None)

        qqq_c  = macro_idx[qqq_c_col]  if qqq_c_col  else pd.Series(dtype=float)
        qqq_lr = macro_idx[qqq_lr_col] if qqq_lr_col else pd.Series(dtype=float)

        feat_df = compute_per_asset_features(bars, qqq_c, qqq_lr)

        assert feat_df.shape[1] == F_TS, f"Expected {F_TS} features, got {feat_df.shape[1]}"
        assert list(feat_df.columns) == TS_FEATURE_NAMES


# ===========================================================================
# Summary helper
# ===========================================================================

if __name__ == "__main__":
    import subprocess, sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=str(Path(__file__).parent.parent),
    )
    sys.exit(result.returncode)
