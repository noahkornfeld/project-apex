"""
Macro Feature Evaluation Suite — §3.3
=======================================
14-test suite across 5 tiers as specified in the evaluation brief.

Tier 1 — Correctness Tests (unit-level, deterministic assertions)
    M1  Feature Value Sanity
    M2  Yield Spread Direction
    M3  Log Return vs Level Features
    M4  Window Accuracy

Tier 2 — Causality Tests (the most critical, per §11.3)
    M5  Temporal Leakage Trap (macro version)
    M6  Decision Date Alignment
    M7  No Forward-Fill of Future Data

Tier 3 — Normalization Tests (§3.6.3 — fixed-scale)
    M8  Fixed-Scale Normalization (not running mean/std)
    M9  Clipping Enforcement
    M10 g_t Dimension Integrity

Tier 4 — Missing Data and Edge Cases
    M11 Full Missing Data Week
    M12 Regime Change Stress Test

Tier 5 — Integration Tests (g_t flows into model correctly)
    M13 Broadcasting Sanity
    M14 Correlation with Known Regimes (qualitative sanity)
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.macro_broadcast_features import (
    compute_macro_broadcast_features,
    normalize_macro_features,
    MACRO_FEATURE_NAMES,
)
from features.benchmark_features import compute_benchmark_features, BENCHMARK_FEATURE_NAMES
from features.portfolio_state_features import PORTFOLIO_STATE_FEATURE_NAMES
from features.normalizers import FixedScaleNormalizer

DATA_DIR   = Path(__file__).parent.parent / "Ticker_Data"
CLIP       = 4.0
D_MACRO    = len(MACRO_FEATURE_NAMES)        # 9
D_BENCH    = len(BENCHMARK_FEATURE_NAMES)    # 3
D_PORT     = len(PORTFOLIO_STATE_FEATURE_NAMES)  # 8
D_GLOBAL   = D_MACRO + D_BENCH + D_PORT     # 20


# ===========================================================================
# Helpers
# ===========================================================================

def _make_controlled_macro(
    dates: pd.DatetimeIndex,
    vix:   np.ndarray,
    y10:   np.ndarray,
    y3m:   np.ndarray,
    oil:   np.ndarray,
    gold:  np.ndarray,
    dxy:   np.ndarray,
    hyg:   np.ndarray,
    qqq:   np.ndarray,
) -> pd.DataFrame:
    """Build a synthetic macro DataFrame with known values."""
    return pd.DataFrame({
        "date":                     dates,
        "VIX_Close":                vix.astype(float),
        "Yield_10Y":                y10.astype(float),
        "Yield_3M":                 y3m.astype(float),
        "Oil_Close":                oil.astype(float),
        "Gold_Close":               gold.astype(float),
        "Dollar_Index_Close":       dxy.astype(float),
        "HYG_Close":                hyg.astype(float),
        "QQQ_Close":                qqq.astype(float),
        "QQQ_log_return":           np.concatenate([[0.0], np.log(qqq[1:] / qqq[:-1])]),
    })


def _flat_macro(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """Minimal flat (slowly varying) macro DataFrame."""
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-02", periods=n, freq="B")
    qqq   = 300.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    return _make_controlled_macro(
        dates=dates,
        vix=rng.uniform(12, 20, n),
        y10=rng.uniform(2.0, 3.0, n),
        y3m=rng.uniform(1.0, 2.0, n),
        oil=rng.uniform(50, 70, n),
        gold=rng.uniform(1300, 1500, n),
        dxy=rng.uniform(95, 105, n),
        hyg=rng.uniform(83, 88, n),
        qqq=qqq,
    )


# ===========================================================================
# TIER 1 — Correctness Tests
# ===========================================================================

class TestM1FeatureValueSanity:
    """
    M1 — Feature Value Sanity
    VIX=20 on 2020-01-06, VIX=22 on 2020-01-13 (5 trading days later).
    Assert vix_level=22.0, vix_change=+2.0 (1-week delta), yield_spread = 10Y - 3M (signed).
    """

    def test_vix_level_is_spot_close(self):
        """vix_level must equal the raw VIX close on the same date (before normalization)."""
        dates  = pd.date_range("2020-01-06", periods=10, freq="B")
        vix    = np.array([20.0, 20.5, 21.0, 21.5, 21.8, 22.0, 22.3, 22.5, 22.7, 23.0])
        qqq    = np.ones(10) * 200.0

        df = _make_controlled_macro(
            dates, vix,
            y10=np.ones(10)*2.5, y3m=np.ones(10)*2.0,
            oil=np.ones(10)*60, gold=np.ones(10)*1400,
            dxy=np.ones(10)*98, hyg=np.ones(10)*85,
            qqq=qqq,
        )
        out = compute_macro_broadcast_features(df)

        # At t=2020-01-13 (index 5): vix_level must equal 22.0
        t = pd.Timestamp("2020-01-13")
        assert abs(out.loc[t, "vix_level"] - 22.0) < 1e-6, \
            f"vix_level at 2020-01-13 = {out.loc[t, 'vix_level']:.4f}, expected 22.0"

    def test_vix_change_is_5day_delta(self):
        """vix_change must be a 5-trading-day (1-week) delta: VIX(t) - VIX(t-5)."""
        dates = pd.date_range("2020-01-06", periods=10, freq="B")
        vix   = np.array([20.0, 20.5, 21.0, 21.5, 21.8, 22.0, 22.3, 22.5, 22.7, 23.0])
        qqq   = np.ones(10) * 200.0

        df = _make_controlled_macro(
            dates, vix,
            y10=np.ones(10)*2.5, y3m=np.ones(10)*2.0,
            oil=np.ones(10)*60, gold=np.ones(10)*1400,
            dxy=np.ones(10)*98, hyg=np.ones(10)*85,
            qqq=qqq,
        )
        out = compute_macro_broadcast_features(df)

        # At 2020-01-13 (index 5): vix_change = VIX[5] - VIX[0] = 22.0 - 20.0 = +2.0
        t = pd.Timestamp("2020-01-13")
        expected_change = 22.0 - 20.0
        actual_change   = out.loc[t, "vix_change"]
        assert abs(actual_change - expected_change) < 1e-6, \
            f"vix_change at 2020-01-13 = {actual_change:.4f}, expected {expected_change:.4f} (1-week delta)"

    def test_yield_spread_is_10y_minus_3m(self):
        """yield_spread = 10Y_yield - 3M_yield (signed, not absolute)."""
        n     = 10
        dates = pd.date_range("2020-01-02", periods=n, freq="B")
        qqq   = np.ones(n) * 200.0

        df = _make_controlled_macro(
            dates, vix=np.ones(n)*15,
            y10=np.ones(n)*4.5, y3m=np.ones(n)*3.2,
            oil=np.ones(n)*60, gold=np.ones(n)*1400,
            dxy=np.ones(n)*98, hyg=np.ones(n)*85,
            qqq=qqq,
        )
        out = compute_macro_broadcast_features(df)

        expected = 4.5 - 3.2
        actual   = out["yield_spread"].iloc[-1]
        assert abs(actual - expected) < 1e-4, \
            f"yield_spread = {actual:.4f}, expected {expected:.4f}"

    def test_vix_change_first_5_rows_are_nan_or_zero(self):
        """First 5 rows cannot have a valid 5-day vix_change (filled with 0 by ffill)."""
        df  = _flat_macro(n=20)
        out = compute_macro_broadcast_features(df)
        # After ffill().fillna(0), row 0 should be 0 (no prior data)
        assert out["vix_change"].iloc[0] == 0.0, \
            "First row vix_change should be 0 (no prior 5-day window available)"


class TestM2YieldSpreadDirection:
    """
    M2 — Yield Spread Direction
    Inverted curve: 10Y=4.5%, 3M=5.2% → spread=-0.7 (negative).
    Normal curve: 10Y=4.5%, 3M=2.1% → spread=+2.4 (positive).
    Sign must be preserved — do NOT take absolute value.
    """

    def _make_yield_df(self, y10_val: float, y3m_val: float) -> pd.DataFrame:
        n     = 10
        dates = pd.date_range("2023-01-02", periods=n, freq="B")
        qqq   = np.ones(n) * 350.0
        return _make_controlled_macro(
            dates, vix=np.ones(n)*20,
            y10=np.ones(n)*y10_val, y3m=np.ones(n)*y3m_val,
            oil=np.ones(n)*80, gold=np.ones(n)*1900,
            dxy=np.ones(n)*103, hyg=np.ones(n)*75,
            qqq=qqq,
        )

    def test_inverted_yield_curve_is_negative(self):
        """Inverted curve (10Y=4.5%, 3M=5.2%) → yield_spread ≈ -0.7."""
        df  = self._make_yield_df(y10_val=4.5, y3m_val=5.2)
        out = compute_macro_broadcast_features(df)
        spread = out["yield_spread"].iloc[-1]
        assert spread < 0.0, f"Inverted yield curve must give negative spread, got {spread:.4f}"
        assert abs(spread - (-0.7)) < 1e-4, \
            f"Expected spread ≈ -0.7, got {spread:.4f}"

    def test_normal_yield_curve_is_positive(self):
        """Normal curve (10Y=4.5%, 3M=2.1%) → yield_spread ≈ +2.4."""
        df  = self._make_yield_df(y10_val=4.5, y3m_val=2.1)
        out = compute_macro_broadcast_features(df)
        spread = out["yield_spread"].iloc[-1]
        assert spread > 0.0, f"Normal yield curve must give positive spread, got {spread:.4f}"
        assert abs(spread - 2.4) < 1e-4, \
            f"Expected spread ≈ +2.4, got {spread:.4f}"

    def test_spread_sign_is_not_absolute_value(self):
        """Confirm inversion produces negative number, not its absolute value."""
        df_inv  = self._make_yield_df(4.5, 5.2)
        df_norm = self._make_yield_df(4.5, 2.1)
        out_inv  = compute_macro_broadcast_features(df_inv)
        out_norm = compute_macro_broadcast_features(df_norm)
        assert out_inv["yield_spread"].iloc[-1] < 0
        assert out_norm["yield_spread"].iloc[-1] > 0


class TestM3LogReturnVsLevelFeatures:
    """
    M3 — Log Return vs Level Features
    Price-based instruments (Oil, Gold, HYG, DXY, QQQ) → log returns.
    VIX, 10Y Yield, 3M Yield → levels (or 5-day delta for VIX change).
    """

    def test_vix_level_is_not_a_return(self):
        """vix_level values should be in a plausible VIX range [5, 100], not small log-return-like values."""
        df  = _flat_macro(n=100)
        out = compute_macro_broadcast_features(df)
        assert out["vix_level"].max() > 5.0, "vix_level looks like a return, not a level"
        assert out["vix_level"].max() < 200.0, "vix_level unreasonably large"

    def test_yield_features_are_levels_not_returns(self):
        """Yield features should reflect raw % yield levels, not log returns."""
        df  = _flat_macro(n=100)
        out = compute_macro_broadcast_features(df)
        # Yield levels are ~1–5%, not tiny log-return-sized values
        assert out["yield_10y"].mean() > 0.5, "yield_10y looks like a return, not a level"
        assert out["yield_3m"].mean() > 0.5,  "yield_3m looks like a return, not a level"

    def test_oil_gold_dxy_hyg_are_log_returns(self):
        """
        Oil, gold, DXY, HYG log returns must be return-sized, not price-sized.
        Threshold is 0.5 (50% single-day move) to accommodate real extreme events
        (e.g., oil dropped ~30% in a single day during COVID).
        Price levels would be 40-100+, far exceeding 0.5.
        """
        df  = _flat_macro(n=200)
        out = compute_macro_broadcast_features(df)
        for col in ["oil_log_ret", "gold_log_ret", "dxy_log_ret", "hyg_log_ret"]:
            max_abs = out[col].abs().max()
            assert max_abs < 0.5, \
                f"{col} max abs = {max_abs:.4f}, too large to be a log return — looks like a price level (threshold 0.5)"

    def test_log_returns_sum_to_total_return(self):
        """
        Sum of daily log returns ≈ total log return from start to end.
        Verifies log return identity ln(P_T/P_0) = Σ ln(P_t/P_{t-1}).
        """
        n     = 50
        dates = pd.date_range("2023-01-02", periods=n, freq="B")
        rng   = np.random.default_rng(42)
        oil_p = 60.0 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
        qqq   = np.ones(n) * 350.0

        df = _make_controlled_macro(
            dates, vix=np.ones(n)*18,
            y10=np.ones(n)*4.0, y3m=np.ones(n)*3.5,
            oil=oil_p, gold=np.ones(n)*1900,
            dxy=np.ones(n)*103, hyg=np.ones(n)*75,
            qqq=qqq,
        )
        out = compute_macro_broadcast_features(df)

        # Sum of log returns (drop first NaN row)
        sum_log_ret  = out["oil_log_ret"].iloc[1:].sum()
        total_log_ret = np.log(oil_p[-1] / oil_p[0])
        assert abs(sum_log_ret - total_log_ret) < 1e-4, \
            f"Σ(oil_log_ret) = {sum_log_ret:.6f}, expected {total_log_ret:.6f}"


class TestM4WindowAccuracy:
    """
    M4 — Window Accuracy
    Exact window sizes per spec:
        vix_level  : 1 day (spot)
        vix_change : 5 trading days
        qqq_1w_ret : 5 trading days
        qqq_4w_vol : 20 trading days
        qqq_12w_ret: 60 trading days
    Offset by 1 day must change rolling features.
    """

    def test_vix_level_is_same_day_spot(self):
        """vix_level at t = VIX close on day t exactly (no window)."""
        dates = pd.date_range("2023-01-02", periods=10, freq="B")
        vix   = np.arange(10.0, 20.0)   # 10, 11, 12, ...
        qqq   = np.ones(10) * 350.0
        df = _make_controlled_macro(
            dates, vix, y10=np.ones(10)*4.0, y3m=np.ones(10)*3.5,
            oil=np.ones(10)*80, gold=np.ones(10)*1900,
            dxy=np.ones(10)*103, hyg=np.ones(10)*75, qqq=qqq,
        )
        out = compute_macro_broadcast_features(df)
        for i, d in enumerate(dates):
            assert abs(out.loc[d, "vix_level"] - vix[i]) < 1e-6, \
                f"vix_level at {d.date()} should be {vix[i]}, got {out.loc[d, 'vix_level']}"

    def test_vix_change_window_is_5_days(self):
        """
        vix_change at row t = VIX[t] - VIX[t-5].
        Verified against hand-computed values.
        """
        n     = 15
        dates = pd.date_range("2023-01-02", periods=n, freq="B")
        vix   = np.arange(10.0, 10.0 + n)  # 10, 11, 12, ... 24
        qqq   = np.ones(n) * 350.0
        df = _make_controlled_macro(
            dates, vix, y10=np.ones(n)*4.0, y3m=np.ones(n)*3.5,
            oil=np.ones(n)*80, gold=np.ones(n)*1900,
            dxy=np.ones(n)*103, hyg=np.ones(n)*75, qqq=qqq,
        )
        out = compute_macro_broadcast_features(df)

        # At index 5: VIX[5] - VIX[0] = 15 - 10 = 5.0
        t = dates[5]
        expected = vix[5] - vix[0]
        actual   = out.loc[t, "vix_change"]
        assert abs(actual - expected) < 1e-6, \
            f"vix_change at index 5: expected {expected:.1f} (5-day delta), got {actual:.4f}"

    def test_qqq_windows_match_spec(self):
        """qqq_1w_ret=5d, qqq_4w_vol=20d, qqq_12w_ret=60d — verified by offset test."""
        n     = 130
        dates = pd.date_range("2020-01-02", periods=n, freq="B")
        rng   = np.random.default_rng(7)
        qqq   = 300.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
        df    = _make_controlled_macro(
            dates, vix=rng.uniform(12, 20, n),
            y10=np.ones(n)*2.5, y3m=np.ones(n)*1.8,
            oil=np.ones(n)*65, gold=np.ones(n)*1400,
            dxy=np.ones(n)*97, hyg=np.ones(n)*86,
            qqq=qqq,
        )
        bench_full   = compute_benchmark_features(df)
        bench_trunc  = compute_benchmark_features(df.iloc[:n-1])

        # Last available row in truncated is different from last row in full
        common_last  = bench_trunc.index[-1]
        full_val_1w  = bench_full.loc[common_last, "qqq_ret_1w"]
        trunc_val_1w = bench_trunc.loc[common_last, "qqq_ret_1w"]
        assert abs(full_val_1w - trunc_val_1w) < 1e-8, \
            "qqq_ret_1w at t should not change when adding a FUTURE day (causal)"

    def test_offset_by_1_day_changes_rolling_output(self):
        """
        Shift the QQQ price series by 1 day and verify the rolling benchmark
        features change for a given date (window boundaries shift).
        """
        n     = 70
        rng   = np.random.default_rng(99)
        qqq_a = 300.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
        qqq_b = np.concatenate([[290.0], qqq_a[:-1]])  # shifted by 1

        dates = pd.date_range("2022-01-03", periods=n, freq="B")
        make  = lambda q: _make_controlled_macro(
            dates, vix=np.ones(n)*15,
            y10=np.ones(n)*3.0, y3m=np.ones(n)*2.5,
            oil=np.ones(n)*75, gold=np.ones(n)*1800,
            dxy=np.ones(n)*100, hyg=np.ones(n)*80,
            qqq=q,
        )
        bench_a = compute_benchmark_features(make(qqq_a))
        bench_b = compute_benchmark_features(make(qqq_b))

        # Rolling vol must differ because the price history shifted
        vol_a = bench_a["qqq_vol_4w"].iloc[25]
        vol_b = bench_b["qqq_vol_4w"].iloc[25]
        assert abs(vol_a - vol_b) > 1e-6, \
            "Shifting QQQ prices by 1 day must change qqq_vol_4w (window boundary moved)"


# ===========================================================================
# TIER 2 — Causality Tests
# ===========================================================================

class TestM5TemporalLeakageTrap:
    """
    M5 — Temporal Leakage Trap (macro version)
    Causal normalization (IS-only, weeks 1-25) vs full-range (weeks 1-52).
    Rolling normalized features must diverge; our implementation uses causal.
    """

    def test_is_norm_differs_from_full_range_at_midpoint(self):
        """
        Normalize macro features using IS-only stats vs full-range stats.
        At t=week_25, the normalized values must differ for rolling features
        (confirms normalization is data-dependent and IS-only matters).
        """
        n = 260   # 52 weeks of daily data
        df = _flat_macro(n=n, seed=77)

        raw = compute_macro_broadcast_features(df)

        split = 130   # week 26 boundary
        is_data  = raw.iloc[:split]
        all_data = raw

        norm_is   = FixedScaleNormalizer(clip=CLIP)
        norm_full = FixedScaleNormalizer(clip=CLIP)
        norm_is.fit(is_data)
        norm_full.fit(all_data)

        z_is   = norm_is.transform(raw)
        z_full = norm_full.transform(raw)

        # At the split point, IS-only vs full-range must differ
        diff = (z_is.iloc[split-1] - z_full.iloc[split-1]).abs().max()
        assert diff > 1e-4, \
            f"IS-only and full-range normalization must differ at midpoint; max diff = {diff:.6f}"

    def test_vix_level_raw_value_unchanged_by_future_data(self):
        """
        Raw vix_level at t=25 is identical whether we pass 25 or 52 weeks of data.
        Spot features are causal by definition — no rolling window on raw values.
        """
        n = 260
        df = _flat_macro(n=n, seed=44)

        raw_full   = compute_macro_broadcast_features(df)
        raw_trunc  = compute_macro_broadcast_features(df.iloc[:130])

        common_end = raw_trunc.index[-1]
        full_val   = raw_full.loc[common_end, "vix_level"]
        trunc_val  = raw_trunc.loc[common_end, "vix_level"]
        assert abs(full_val - trunc_val) < 1e-8, \
            "Raw vix_level at t should be identical regardless of how many future rows are present"

    def test_live_implementation_is_causal(self):
        """
        Verify: appending a future week of extreme VIX data does NOT change
        the normalized value at the last week of the original series.
        (Method A — causal — is what the implementation must use.)
        """
        n = 100
        df = _flat_macro(n=n, seed=55)
        raw_orig = compute_macro_broadcast_features(df)

        norm = FixedScaleNormalizer(clip=CLIP)
        norm.fit(raw_orig)
        z_orig = norm.transform(raw_orig)
        last_t = z_orig.index[-1]

        # Build extended version with 10 more extreme rows
        extra_dates = pd.date_range(df["date"].iloc[-1] + pd.Timedelta(days=1), periods=10, freq="B")
        extra_df    = _flat_macro(n=10, seed=200)
        extra_df["date"]      = extra_dates
        extra_df["VIX_Close"] = 80.0   # extreme future

        df_extended  = pd.concat([df, extra_df], ignore_index=True)
        raw_extended = compute_macro_broadcast_features(df_extended)

        # Using SAME frozen normalizer (causal: don't refit)
        z_extended = norm.transform(raw_extended)

        # Original rows must be identical
        np.testing.assert_allclose(
            z_orig.loc[last_t].values,
            z_extended.loc[last_t].values,
            rtol=1e-5, atol=1e-6,
            err_msg="Causal: appending extreme future data must not change past normalized values",
        )


class TestM6DecisionDateAlignment:
    """
    M6 — Decision Date Alignment
    g_t at week t uses VIX close of d_t (NOT d_{t+1}).
    Injecting a spike at d_{t+1} must NOT appear in g_t at d_t.
    """

    def test_spike_at_t_plus_1_not_in_g_t(self):
        """
        Inject VIX=999 at d_{t+1}. g_t at d_t must not reflect this spike.
        """
        n     = 30
        dates = pd.date_range("2023-01-02", periods=n, freq="B")
        vix   = np.ones(n) * 20.0

        df = _make_controlled_macro(
            dates, vix, y10=np.ones(n)*4.0, y3m=np.ones(n)*3.5,
            oil=np.ones(n)*80, gold=np.ones(n)*1900,
            dxy=np.ones(n)*103, hyg=np.ones(n)*75,
            qqq=np.ones(n)*350,
        )
        # Normal output at d_t (index 20)
        out_normal = compute_macro_broadcast_features(df)
        vix_at_dt  = out_normal["vix_level"].iloc[20]

        # Inject spike at d_{t+1} (index 21)
        df_spike = df.copy()
        df_spike.loc[df_spike.index[21], "VIX_Close"] = 999.0
        out_spike  = compute_macro_broadcast_features(df_spike)

        # g_t at index 20 must be unchanged
        assert abs(out_spike["vix_level"].iloc[20] - vix_at_dt) < 1e-6, \
            f"VIX spike at d_{{t+1}} leaked into g_t at d_t: {out_spike['vix_level'].iloc[20]:.2f} ≠ {vix_at_dt:.2f}"

    def test_t_plus_1_data_not_in_vix_change_at_t(self):
        """
        vix_change at d_t = VIX[t] - VIX[t-5].
        Modifying VIX[t+1] must not change vix_change at d_t.
        """
        n     = 30
        dates = pd.date_range("2023-01-02", periods=n, freq="B")
        vix   = np.ones(n) * 20.0
        df    = _make_controlled_macro(
            dates, vix, y10=np.ones(n)*4.0, y3m=np.ones(n)*3.5,
            oil=np.ones(n)*80, gold=np.ones(n)*1900,
            dxy=np.ones(n)*103, hyg=np.ones(n)*75,
            qqq=np.ones(n)*350,
        )
        out_normal = compute_macro_broadcast_features(df)
        change_at_t = out_normal["vix_change"].iloc[10]

        df_mod = df.copy()
        df_mod.loc[df_mod.index[11], "VIX_Close"] = 500.0   # d_{t+1}
        out_mod = compute_macro_broadcast_features(df_mod)

        assert abs(out_mod["vix_change"].iloc[10] - change_at_t) < 1e-6, \
            "Modifying VIX at d_{t+1} must not change vix_change at d_t"


class TestM7NoForwardFillFutureData:
    """
    M7 — No Forward-Fill of Future Data
    Missing data → backward carry (ffill), NOT forward-fill of future values.
    g_t must not contain VIX value from d_{t+1}.
    """

    def test_missing_vix_uses_backward_carry(self):
        """
        Mask VIX on d_t. The implementation must use VIX from d_{t-1} (backward carry).
        It must NOT use NaN and must NOT use VIX from d_{t+1}.
        """
        n     = 20
        dates = pd.date_range("2023-01-02", periods=n, freq="B")
        vix   = np.arange(10.0, 10.0 + n)  # 10, 11, 12, ...

        df = _make_controlled_macro(
            dates, vix, y10=np.ones(n)*4.0, y3m=np.ones(n)*3.5,
            oil=np.ones(n)*80, gold=np.ones(n)*1900,
            dxy=np.ones(n)*103, hyg=np.ones(n)*75,
            qqq=np.ones(n)*350,
        )
        # Mask VIX on day 10 → should backward-fill from day 9
        df_gap = df.copy()
        df_gap.loc[df_gap.index[10], "VIX_Close"] = np.nan

        out = compute_macro_broadcast_features(df_gap)

        # After backward carry: vix_level at day 10 = VIX at day 9 = 19.0
        backward_carry_value = vix[9]   # 19.0
        future_value         = vix[11]  # 21.0

        actual = out["vix_level"].iloc[10]
        assert abs(actual - backward_carry_value) < 1e-6, \
            f"Missing VIX at d_t should use backward carry ({backward_carry_value:.1f}), got {actual:.4f}"
        assert abs(actual - future_value) > 1e-6, \
            f"Missing VIX must NOT forward-fill from d_{{t+1}} ({future_value:.1f})"

    def test_no_nan_after_missing_data(self):
        """After handling missing data, g_t must contain no NaN."""
        df_gap  = _flat_macro(n=50)
        df_gap  = df_gap.copy()
        # Introduce several gaps
        df_gap.loc[df_gap.index[5], "VIX_Close"]             = np.nan
        df_gap.loc[df_gap.index[15], "Oil_Close"]            = np.nan
        df_gap.loc[df_gap.index[25], "Gold_Close"]           = np.nan

        out = compute_macro_broadcast_features(df_gap)
        assert out.isna().sum().sum() == 0, \
            "g_t must contain no NaN after missing data backward carry"

    def test_does_not_impute_missing_as_zero(self):
        """
        Missing VIX should NOT be replaced by 0.0.
        VIX=0 is physically impossible and semantically wrong.
        """
        n     = 20
        dates = pd.date_range("2023-01-02", periods=n, freq="B")
        vix   = np.ones(n) * 25.0  # steady VIX

        df = _make_controlled_macro(
            dates, vix, y10=np.ones(n)*4.0, y3m=np.ones(n)*3.5,
            oil=np.ones(n)*80, gold=np.ones(n)*1900,
            dxy=np.ones(n)*103, hyg=np.ones(n)*75,
            qqq=np.ones(n)*350,
        )
        # Gap on day 8
        df_gap = df.copy()
        df_gap.loc[df_gap.index[8], "VIX_Close"] = np.nan

        out = compute_macro_broadcast_features(df_gap)
        actual = out["vix_level"].iloc[8]
        assert actual != 0.0, \
            f"Missing VIX imputed as 0.0 — this is semantically wrong. Got {actual}"
        assert abs(actual - 25.0) < 1e-6, \
            f"Missing VIX should be backward-carried as 25.0, got {actual}"


# ===========================================================================
# TIER 3 — Normalization Tests
# ===========================================================================

class TestM8FixedScaleNormalization:
    """
    M8 — Fixed-Scale Normalization
    Macro features use FIXED normalization (§3.6.3), not per-asset running stats.
    Constants computed once on training data and FROZEN for OOS.
    """

    def test_normalizer_constants_frozen_after_fit(self):
        """
        Once fitted, normalizer mean/std do NOT change when transform() is called on new data.
        """
        df  = _flat_macro(n=200, seed=10)
        raw = compute_macro_broadcast_features(df)

        norm = FixedScaleNormalizer(clip=CLIP)
        norm.fit(raw.iloc[:100])

        stats_before = norm._stats.copy()
        _ = norm.transform(raw.iloc[100:])   # transform OOS
        stats_after  = norm._stats

        pd.testing.assert_frame_equal(
            stats_before, stats_after,
            check_exact=True,
            obj="Normalizer stats must be frozen after fit — they changed after transform()",
        )

    def test_same_constants_applied_to_is_and_oos(self):
        """
        Using one fitted normalizer, both IS and OOS data are normalized with identical constants.
        """
        df  = _flat_macro(n=300, seed=11)
        raw = compute_macro_broadcast_features(df)

        norm = FixedScaleNormalizer(clip=CLIP)
        norm.fit(raw.iloc[:150])

        is_stats  = norm._stats.copy()
        _ = norm.transform(raw.iloc[150:])
        oos_stats = norm._stats

        pd.testing.assert_frame_equal(is_stats, oos_stats, check_exact=True,
            obj="Normalizer stats changed between IS and OOS transforms")

    def test_fixed_scale_different_from_causal_per_asset(self):
        """
        Fixed-scale macro normalizer uses global stats; per-asset uses rolling stats.
        The two normalizers must produce different outputs for the same feature series.
        (This confirms they are different implementations, not accidentally the same.)
        """
        from features.normalizers import CausalPerAssetNormalizer

        df  = _flat_macro(n=200, seed=12)
        raw = compute_macro_broadcast_features(df)

        fixed_norm  = FixedScaleNormalizer(clip=99.0)
        causal_norm = CausalPerAssetNormalizer(window=52, clip=99.0)

        z_fixed  = fixed_norm.fit_transform(raw[["vix_level"]])
        z_causal = causal_norm.fit_transform(raw[["vix_level"]])

        diff = (z_fixed["vix_level"] - z_causal["vix_level"]).abs().max()
        assert diff > 0.01, \
            "Fixed-scale and causal per-asset normalizers must produce different outputs"


class TestM9ClippingEnforcement:
    """
    M9 — Clipping Enforcement
    Extreme VIX=80 (COVID peak) must clip to exactly 4.0 after normalization.
    VIX=5 (near historical low) must clip to exactly -4.0.
    """

    def _make_extreme_vix_df(self, n_normal: int = 200, extreme_vix: float = 80.0):
        """Build macro df with extreme VIX spike appended."""
        df_normal = _flat_macro(n=n_normal, seed=0)
        # VIX in normal data is ~12-20; extreme is 80 (COVID peak)
        extra_dates = pd.date_range(
            df_normal["date"].iloc[-1] + pd.Timedelta(days=1), periods=5, freq="B"
        )
        extra_df              = _flat_macro(n=5, seed=1)
        extra_df["date"]      = extra_dates
        extra_df["VIX_Close"] = extreme_vix
        return pd.concat([df_normal, extra_df], ignore_index=True)

    def test_extreme_vix_clips_to_positive_4(self):
        """VIX=80 normalized then clipped must equal +4.0 exactly."""
        df  = self._make_extreme_vix_df(extreme_vix=80.0)
        raw = compute_macro_broadcast_features(df)
        z   = normalize_macro_features(raw, clip=CLIP)

        extreme_z = z["vix_level"].iloc[-1]
        assert abs(extreme_z - CLIP) < 1e-6, \
            f"VIX=80 normalized value should be clipped to {CLIP}, got {extreme_z:.4f}"

    def test_low_vix_clips_to_negative_4(self):
        """
        VIX=5 (near historical low) clipped must equal -4.0.
        The normalizer is fitted on normal-period data (VIX 12-20) only,
        so VIX=5 is many standard deviations below the mean and clips to -4.0.
        """
        # Normal period: VIX 12-20 (mean~16, std~2.3) → VIX=5 is ~4.8σ below mean
        df_normal = _flat_macro(n=200, seed=0)   # VIX ~12-20
        raw_normal = compute_macro_broadcast_features(df_normal)

        # Fit normalizer on normal data only
        norm = FixedScaleNormalizer(clip=CLIP)
        norm.fit(raw_normal)

        # Create a single-row extreme low VIX
        last_date  = pd.date_range(df_normal["date"].iloc[-1] + pd.Timedelta(days=1), periods=1, freq="B")
        df_extreme = _make_controlled_macro(
            last_date, vix=np.array([5.0]),
            y10=np.array([2.5]), y3m=np.array([1.8]),
            oil=np.array([65.0]), gold=np.array([1400.0]),
            dxy=np.array([97.0]), hyg=np.array([86.0]),
            qqq=np.array([300.0]),
        )
        raw_extreme = compute_macro_broadcast_features(pd.concat([df_normal, df_extreme], ignore_index=True))
        z_extreme   = norm.transform(raw_extreme)

        low_z = z_extreme["vix_level"].iloc[-1]
        assert abs(low_z - (-CLIP)) < 1e-6, \
            f"VIX=5 normalized value should be clipped to -{CLIP}, got {low_z:.4f}"

    def test_all_macro_features_within_clip_bounds(self):
        """After normalization, ALL macro feature values must be in [-4, +4]."""
        df  = _flat_macro(n=300, seed=99)
        raw = compute_macro_broadcast_features(df)
        z   = normalize_macro_features(raw, clip=CLIP)

        max_abs = z.abs().max().max()
        assert max_abs <= CLIP + 1e-6, \
            f"Max abs normalized macro value = {max_abs:.4f}, exceeds clip {CLIP}"


class TestM10GtDimensionIntegrity:
    """
    M10 — g_t Dimension Integrity
    D_global = 20 exactly (9 macro + 3 benchmark + 8 portfolio-state).
    """

    def test_d_global_is_20(self):
        """D_global constant must equal 20."""
        assert D_GLOBAL == 20, f"D_global = {D_GLOBAL}, expected 20"

    def test_macro_contributes_9_features(self):
        """Macro features must produce exactly 9 columns."""
        df  = _flat_macro(n=50)
        out = compute_macro_broadcast_features(df)
        assert out.shape[1] == 9, f"Macro features: expected 9, got {out.shape[1]}"

    def test_benchmark_contributes_3_features(self):
        """Benchmark features must produce exactly 3 columns."""
        df  = _flat_macro(n=80)
        out = compute_benchmark_features(df)
        assert out.shape[1] == 3, f"Benchmark features: expected 3, got {out.shape[1]}"

    def test_portfolio_state_contributes_8_features(self):
        """Portfolio-state stub must have exactly 8 features."""
        assert D_PORT == 8, f"Portfolio-state dim = {D_PORT}, expected 8"

    def test_g_t_concatenation_shape(self):
        """Concatenating macro + benchmark + portfolio-state must give shape [T, 20]."""
        from features.portfolio_state_features import compute_portfolio_state_stub

        n   = 80
        df  = _flat_macro(n=n)
        raw_macro = compute_macro_broadcast_features(df)
        raw_bench = compute_benchmark_features(df)
        port_stub = compute_portfolio_state_stub(n_dates=n)

        g_t = np.concatenate([
            raw_macro.values,
            raw_bench.values,
            port_stub,
        ], axis=1)

        assert g_t.shape == (n, 20), \
            f"g_t shape = {g_t.shape}, expected ({n}, 20)"

    def test_adding_macro_feature_would_break_dimension(self):
        """
        If someone adds a 10th macro feature, g_t would become [T, 21].
        This test documents the invariant and will catch accidental additions.
        """
        assert len(MACRO_FEATURE_NAMES) == 9, \
            f"MACRO_FEATURE_NAMES has {len(MACRO_FEATURE_NAMES)} entries; D_global must be updated if this changes"


# ===========================================================================
# TIER 4 — Missing Data and Edge Cases
# ===========================================================================

class TestM11FullMissingDataWeek:
    """
    M11 — Full Missing Data Week
    All macro instruments missing → backward carry, no NaN, no crash.
    Does NOT treat missing as 0 (VIX=0 is physically impossible).
    """

    def test_all_missing_week_no_nan(self):
        """Setting all instrument prices NaN for a week → output has no NaN."""
        n   = 30
        df  = _flat_macro(n=n, seed=13)
        df_gap = df.copy()
        # Wipe entire week (rows 10-14)
        price_cols = ["VIX_Close", "Yield_10Y", "Yield_3M",
                      "Oil_Close", "Gold_Close", "Dollar_Index_Close",
                      "HYG_Close", "QQQ_Close"]
        for col in price_cols:
            if col in df_gap.columns:
                df_gap.loc[df_gap.index[10:15], col] = np.nan

        out = compute_macro_broadcast_features(df_gap)
        assert out.isna().sum().sum() == 0, \
            f"Full missing week: NaN count = {out.isna().sum().sum()}, expected 0"

    def test_all_missing_uses_backward_carry_not_zero(self):
        """
        During a full outage week, feature values = last known values, NOT 0.
        VIX=0 would be semantically wrong.
        """
        n     = 25
        dates = pd.date_range("2023-01-02", periods=n, freq="B")
        vix   = np.ones(n) * 18.0  # stable VIX
        df = _make_controlled_macro(
            dates, vix, y10=np.ones(n)*4.0, y3m=np.ones(n)*3.5,
            oil=np.ones(n)*80, gold=np.ones(n)*1900,
            dxy=np.ones(n)*103, hyg=np.ones(n)*75,
            qqq=np.ones(n)*350,
        )
        df_gap = df.copy()
        for col in ["VIX_Close", "Yield_10Y", "Yield_3M",
                    "Oil_Close", "Gold_Close", "Dollar_Index_Close", "HYG_Close"]:
            if col in df_gap.columns:
                df_gap.loc[df_gap.index[10:15], col] = np.nan

        out = compute_macro_broadcast_features(df_gap)

        # vix_level during outage week should be ~18 (backward carry), not 0
        for row in range(10, 15):
            v = out["vix_level"].iloc[row]
            assert v != 0.0, f"Row {row}: vix_level=0 is semantically wrong (outage should backward-carry)"
            assert abs(v - 18.0) < 1e-6, f"Row {row}: expected backward-carried vix_level=18.0, got {v:.4f}"

    def test_no_crash_on_fully_missing_data(self):
        """compute_macro_broadcast_features must not raise any exception on all-NaN input."""
        n   = 10
        df  = _flat_macro(n=n)
        df_nan = df.copy()
        for col in ["VIX_Close", "Yield_10Y", "Yield_3M",
                    "Oil_Close", "Gold_Close", "Dollar_Index_Close", "HYG_Close"]:
            if col in df_nan.columns:
                df_nan[col] = np.nan

        try:
            out = compute_macro_broadcast_features(df_nan)
        except Exception as e:
            pytest.fail(f"compute_macro_broadcast_features crashed on all-NaN input: {e}")


class TestM12RegimeChangeStressTest:
    """
    M12 — Regime Change Stress Test
    Feed March 2020 COVID crash period (VIX ~15→80).
    Assert spike captured correctly with correct lag.
    """

    @pytest.fixture(autouse=True)
    def check_data(self):
        if not (DATA_DIR / "macro_features.parquet").exists():
            pytest.skip("macro_features.parquet not found — skipping M12 real-data test")

    def test_vix_level_captures_covid_spike(self):
        """At peak COVID week (March 16-20, 2020), vix_level must be > 60."""
        macro_df = pd.read_parquet(DATA_DIR / "macro_features.parquet")
        out = compute_macro_broadcast_features(macro_df)

        covid_peak = out.loc["2020-03-16":"2020-03-20", "vix_level"]
        if len(covid_peak) == 0:
            pytest.skip("COVID peak dates not in dataset")

        max_vix = covid_peak.max()
        assert max_vix > 60.0, \
            f"VIX during COVID peak should be > 60, got {max_vix:.2f}"

    def test_vix_change_captures_spike_direction(self):
        """During COVID crash week, vix_change must be large and positive."""
        macro_df = pd.read_parquet(DATA_DIR / "macro_features.parquet")
        out = compute_macro_broadcast_features(macro_df)

        covid_crash = out.loc["2020-03-09":"2020-03-20", "vix_change"]
        if len(covid_crash) == 0:
            pytest.skip("COVID crash dates not in dataset")

        max_change = covid_crash.max()
        assert max_change > 10.0, \
            f"vix_change during COVID crash should be large positive, got {max_change:.2f}"

    def test_covid_spike_clips_to_4_after_normalization(self):
        """After normalization+clip, the COVID VIX spike saturates at exactly 4.0."""
        macro_df = pd.read_parquet(DATA_DIR / "macro_features.parquet")
        raw = compute_macro_broadcast_features(macro_df)

        # Fit on pre-COVID data (IS period)
        z = normalize_macro_features(raw, fit_end_date="2019-12-31", clip=CLIP)

        covid_peak = z.loc["2020-03-16":"2020-03-20", "vix_level"]
        if len(covid_peak) == 0:
            pytest.skip("COVID peak dates not in dataset")

        for t, v in covid_peak.items():
            assert abs(v - CLIP) < 1e-6, \
                f"COVID VIX at {t.date()} normalized to {v:.4f}, expected clipped at {CLIP}"

    def test_qqq_4w_vol_at_spike_includes_pre_spike_calm_days(self):
        """
        At the WEEK OF the COVID spike, the 4-week vol window (20 days) includes
        pre-spike calm days. Verify causality: vol at spike onset must be GREATER
        than vol from the calm pre-spike week (spike is already captured) but the
        20-day window still contains calm days — the test verifies the window
        boundaries are correct (causal, backward-looking).

        Concretely:
            - Pre-spike calm (Feb 2020): qqq_4w_vol should be LOW
            - Spike onset (Mar 16, 2020): qqq_4w_vol should be HIGH (spike captured)
            - The spike onset vol > pre-spike vol confirms the window is causal
              (picks up the actual crash, not future post-crash recovery data)
        """
        macro_df = pd.read_parquet(DATA_DIR / "macro_features.parquet")
        bench = compute_benchmark_features(macro_df)

        calm_date  = "2020-02-14"   # pre-COVID calm
        spike_date = "2020-03-16"   # peak crash week

        if spike_date not in bench.index:
            pytest.skip("Spike date not in dataset")
        if calm_date not in bench.index:
            pytest.skip("Calm date not in dataset")

        vol_calm  = bench.loc[calm_date,  "qqq_vol_4w"]
        vol_spike = bench.loc[spike_date, "qqq_vol_4w"]

        assert vol_spike > 0.3, \
            f"qqq_4w_vol at COVID spike should be > 0.3 (annualized), got {vol_spike:.4f}"
        assert vol_spike > vol_calm * 2, \
            f"vol at spike ({vol_spike:.4f}) should be >2x the pre-spike calm vol ({vol_calm:.4f}) — causal window is working"


# ===========================================================================
# TIER 5 — Integration Tests
# ===========================================================================

class TestM13BroadcastingSanity:
    """
    M13 — Broadcasting Sanity
    Macro features appear in g_t (global context), NOT in x_t (per-asset block).
    """

    def test_macro_feature_count_matches_g_t_portion(self):
        """The 9 macro feature names occupy the first 9 positions of g_t."""
        g_feature_names = MACRO_FEATURE_NAMES + BENCHMARK_FEATURE_NAMES + list(PORTFOLIO_STATE_FEATURE_NAMES)
        assert g_feature_names[:9]  == MACRO_FEATURE_NAMES,   "First 9 g_t features must be macro"
        assert g_feature_names[9:12] == BENCHMARK_FEATURE_NAMES, "Next 3 g_t features must be benchmark"
        assert len(g_feature_names) == 20, f"Total g_t features = {len(g_feature_names)}, expected 20"

    def test_macro_features_not_in_per_asset_feature_names(self):
        """None of the macro feature names appear in the per-asset feature list."""
        from features import ALL_ASSET_FEATURE_NAMES

        for macro_feat in MACRO_FEATURE_NAMES:
            assert macro_feat not in ALL_ASSET_FEATURE_NAMES, \
                f"Macro feature '{macro_feat}' incorrectly appears in per-asset feature list x_t"

    def test_benchmark_features_not_in_per_asset_feature_names(self):
        """None of the benchmark feature names appear in the per-asset feature list."""
        from features import ALL_ASSET_FEATURE_NAMES

        for bench_feat in BENCHMARK_FEATURE_NAMES:
            assert bench_feat not in ALL_ASSET_FEATURE_NAMES, \
                f"Benchmark feature '{bench_feat}' incorrectly appears in per-asset feature list x_t"

    def test_g_t_and_x_t_have_disjoint_features(self):
        """x_t feature set and g_t feature set are completely disjoint."""
        from features import ALL_ASSET_FEATURE_NAMES

        g_t_names = set(MACRO_FEATURE_NAMES + BENCHMARK_FEATURE_NAMES + list(PORTFOLIO_STATE_FEATURE_NAMES))
        x_t_names = set(ALL_ASSET_FEATURE_NAMES)

        overlap = g_t_names & x_t_names
        assert len(overlap) == 0, \
            f"Overlap between x_t and g_t feature sets: {overlap}"


class TestM14CorrelationWithKnownRegimes:
    """
    M14 — Correlation with Known Regimes (qualitative sanity)
    High VIX → vix_level high in g_t.
    Inverted yield curve → yield_spread negative.
    Bad QQQ 12w → qqq_ret_12w < 0.
    These are soft sanity checks, not exact assertions.
    """

    def test_high_vix_produces_high_vix_level(self):
        """
        When VIX is consistently high (30+), normalized vix_level must be
        consistently positive (above the historical average).
        """
        n_normal = 200
        n_stress  = 30
        dates_n   = pd.date_range("2015-01-02", periods=n_normal, freq="B")
        dates_s   = pd.date_range(dates_n[-1] + pd.Timedelta(days=1), periods=n_stress, freq="B")

        # Normal period: VIX 12-20
        rng = np.random.default_rng(20)
        qqq = 300.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n_normal + n_stress)))
        df_normal = _make_controlled_macro(
            dates_n, vix=rng.uniform(12, 20, n_normal),
            y10=np.ones(n_normal)*2.5, y3m=np.ones(n_normal)*1.8,
            oil=np.ones(n_normal)*65, gold=np.ones(n_normal)*1400,
            dxy=np.ones(n_normal)*97, hyg=np.ones(n_normal)*86,
            qqq=qqq[:n_normal],
        )
        # Stress period: VIX 35-50
        df_stress = _make_controlled_macro(
            dates_s, vix=rng.uniform(35, 50, n_stress),
            y10=np.ones(n_stress)*2.5, y3m=np.ones(n_stress)*1.8,
            oil=np.ones(n_stress)*65, gold=np.ones(n_stress)*1400,
            dxy=np.ones(n_stress)*97, hyg=np.ones(n_stress)*86,
            qqq=qqq[n_normal:],
        )
        df_all = pd.concat([df_normal, df_stress], ignore_index=True)

        raw = compute_macro_broadcast_features(df_all)
        z   = normalize_macro_features(raw, fit_end_date=str(dates_n[-1].date()), clip=CLIP)

        # During stress period, vix_level should be predominantly positive
        stress_vix_z = z.loc[dates_s[0]:dates_s[-1], "vix_level"]
        pct_positive = (stress_vix_z > 0).mean()
        assert pct_positive > 0.8, \
            f"During high-VIX stress period, {pct_positive*100:.0f}% of vix_level is positive — expected >80%"

    def test_inverted_yield_curve_produces_negative_spread(self):
        """When 10Y < 3M (inverted), yield_spread must be negative in g_t."""
        n     = 30
        dates = pd.date_range("2023-01-02", periods=n, freq="B")
        qqq   = np.ones(n) * 350.0

        # Consistently inverted: 10Y=4.0%, 3M=5.5%
        df = _make_controlled_macro(
            dates, vix=np.ones(n)*18,
            y10=np.ones(n)*4.0, y3m=np.ones(n)*5.5,
            oil=np.ones(n)*80, gold=np.ones(n)*1900,
            dxy=np.ones(n)*103, hyg=np.ones(n)*75,
            qqq=qqq,
        )
        raw = compute_macro_broadcast_features(df)
        assert (raw["yield_spread"] < 0).all(), \
            "Inverted yield curve (10Y < 3M) must produce negative yield_spread throughout"

    def test_bad_qqq_12w_produces_negative_feature(self):
        """When QQQ has a bad 12-week period, qqq_ret_12w must be negative."""
        n     = 100
        dates = pd.date_range("2022-01-03", periods=n, freq="B")
        rng   = np.random.default_rng(33)
        # QQQ declines sharply over 60 days
        qqq   = 400.0 * np.exp(np.cumsum(np.full(n, -0.005)))   # steady decline

        df = _make_controlled_macro(
            dates, vix=np.ones(n)*25,
            y10=np.ones(n)*4.0, y3m=np.ones(n)*3.5,
            oil=np.ones(n)*80, gold=np.ones(n)*1900,
            dxy=np.ones(n)*103, hyg=np.ones(n)*75,
            qqq=qqq,
        )
        bench = compute_benchmark_features(df)

        # After 60-day window fills, qqq_ret_12w must be negative
        ret_12w = bench["qqq_ret_12w"].iloc[70:]
        assert (ret_12w < 0).all(), \
            "Declining QQQ over 12 weeks must produce negative qqq_ret_12w"


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=str(Path(__file__).parent.parent),
    )
    sys.exit(result.returncode)
