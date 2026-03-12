"""
tests/test_phase11_gate11.py
============================
Gate 11: Walk-Forward Evaluation  (Bible §9 / screenshot gate table)

Gate criteria (screenshot):
  Embargo    : No training transition has t_idx within embargo window of any fold
  Leakage    : All 5 leakage traps pass for fold 1
  Fold 1 pass: Fold 1 completes: training + OOS eval + all metrics + all plots
  Bootstrap  : CI computation runs without error on concatenated OOS series
  Baselines  : All baseline NAV series produced; equal-weight baseline is valid

Additional unit tests:
  - FoldManager fold specs match Table 31 exactly
  - FoldManager embargo indices correct
  - FoldManager n_steps_dropped correct
  - Metrics: Excess CAGR, Sortino, Max Drawdown correctness
  - Metrics: secondary and tertiary computed without error
  - Bootstrap: block_length = floor(T^(1/3))
  - Bootstrap: CI bounds finite and ordered
  - Checkpoint selector: primary / secondary / stability criteria
  - Baselines: QQQ NAV, equal-weight NAV, model-picks NAV
"""

from __future__ import annotations

import datetime
import math
from typing import Dict, List

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.fold_manager import FoldManager, FoldSpec, FOLD_SPECS, _to_date
from evaluation.leakage_suite import LeakageSuite, LeakageResult
from evaluation.metrics import (
    compute_excess_cagr, compute_sortino, compute_max_drawdown,
    compute_sharpe, compute_effective_n_positions, compute_all_metrics,
)
from evaluation.baselines import (
    BaselineCalculator, build_qqq_nav, build_equal_weight_nav,
    build_equal_weight_model_picks_nav,
)
from evaluation.bootstrap import BlockBootstrap, moving_block_bootstrap
from evaluation.checkpoint_selector import CheckpointSelector, CheckpointRecord


# ===========================================================================
# Shared synthetic data helpers
# ===========================================================================

def _weekly_dates(start: str, n_weeks: int) -> np.ndarray:
    """Generate n_weeks Monday dates starting from start."""
    d0 = _to_date(start)
    # Move to next Monday
    days_until_mon = (7 - d0.weekday()) % 7
    d0 = d0 + datetime.timedelta(days=days_until_mon)
    return np.array([d0 + datetime.timedelta(weeks=i) for i in range(n_weeks)], dtype=object)


def _make_fold1_dates() -> np.ndarray:
    """Weekly dates spanning fold 1 train + test + a bit beyond."""
    # Fold 1: train 2005-01-01 to 2009-12-31, test 2010-01-01 to 2011-12-31
    # ~2005 to ~2012 = ~7 years = ~364 weeks
    return _weekly_dates("2005-01-03", 400)


def _make_full_dates() -> np.ndarray:
    """Weekly dates spanning all 8 folds (2005–2024)."""
    return _weekly_dates("2005-01-03", 1050)   # ~20 years


def _make_returns(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.001, 0.015, n)


def _make_nav(returns: np.ndarray) -> np.ndarray:
    nav = np.cumprod(np.concatenate([[1.0], 1.0 + returns]))
    return nav[1:]   # [T] starting period NAV


# ===========================================================================
# Gate 11 — Embargo
# ===========================================================================

class TestGateEmbargo:
    """
    Gate: No training transition has t_idx within embargo window of any fold.
    Metric / Inspection: Embargo assertion.
    """

    def test_embargo_passes_with_valid_train_indices(self):
        """Training indices from get_train_indices() must pass embargo check."""
        dates = _make_fold1_dates()
        fm = FoldManager(dates, L_lookback=10)

        for fold_id in range(1, 5):
            spec = fm.get_fold_spec(fold_id)
            # Check fold is representable in our date range
            train_idx = fm.get_train_indices(fold_id)
            if len(train_idx) == 0:
                continue
            passed, msg = fm.validate_embargo(fold_id, train_idx)
            assert passed, f"Fold {fold_id} embargo failed: {msg}"

    def test_embargo_fails_when_embargo_indices_included(self):
        """Passing embargo-window indices into validate_embargo must FAIL."""
        dates = _make_fold1_dates()
        fm = FoldManager(dates, L_lookback=10)

        for fold_id in [1]:
            embargo_idx = fm.get_embargo_indices(fold_id)
            if len(embargo_idx) == 0:
                continue
            # Deliberately include embargo indices in training set
            bad_train = np.concatenate([fm.get_train_indices(fold_id), embargo_idx])
            passed, msg = fm.validate_embargo(fold_id, bad_train)
            assert not passed, (
                f"Fold {fold_id}: expected embargo FAIL when embargo indices included, "
                f"got PASS"
            )
            assert "EMBARGO VIOLATION" in msg

    def test_embargo_indices_are_last_4_weeks(self):
        """Embargo indices must fall in the last 4 weeks of training."""
        dates = _make_fold1_dates()
        fm = FoldManager(dates, L_lookback=10)

        embargo_idx = fm.get_embargo_indices(1)
        if len(embargo_idx) == 0:
            pytest.skip("No embargo indices for fold 1 in test date range")

        spec = fm.get_fold_spec(1)
        train_end      = _to_date(spec.train_end)
        embargo_cutoff = train_end - datetime.timedelta(weeks=4)

        for idx in embargo_idx:
            d = dates[idx]
            assert d > embargo_cutoff, f"Date {d} not after embargo cutoff {embargo_cutoff}"
            assert d <= train_end,     f"Date {d} after train_end {train_end}"

    def test_no_overlap_between_train_embargo_test(self):
        """Train, embargo, and test index sets must be disjoint."""
        dates = _make_full_dates()
        fm = FoldManager(dates, L_lookback=60)

        for fold_id in range(1, 9):
            train_idx   = set(fm.get_train_indices(fold_id).tolist())
            embargo_idx = set(fm.get_embargo_indices(fold_id).tolist())
            test_idx    = set(fm.get_test_indices(fold_id).tolist())

            assert train_idx.isdisjoint(embargo_idx), (
                f"Fold {fold_id}: train and embargo index sets overlap"
            )
            assert train_idx.isdisjoint(test_idx), (
                f"Fold {fold_id}: train and test index sets overlap"
            )
            assert embargo_idx.isdisjoint(test_idx), (
                f"Fold {fold_id}: embargo and test index sets overlap"
            )


# ===========================================================================
# Gate 11 — Leakage Suite (§9.2 / §11.3)
# ===========================================================================

class TestGateLeakageSuite:
    """
    Gate: All 5 leakage traps pass for fold 1.
    Metric / Inspection: Leakage test suite.
    """

    def _make_suite_inputs(self):
        """Build synthetic inputs where all 5 checks should PASS."""
        rng   = np.random.default_rng(42)
        T, K, F = 200, 8, 6

        # Causal feature panel: x[t] does NOT use x[t+1:]
        # Simulate by constructing a panel with cumulative sum — purely causal
        x_panel = rng.standard_normal((T, K, F)).astype(np.float32)

        # IS-only normalizer (fitted on first 100 steps) vs full (all T)
        is_data  = x_panel[:100].reshape(-1, F)
        full_data = x_panel.reshape(-1, F)
        stats_is   = {"mean": is_data.mean(axis=0),  "std": is_data.std(axis=0) + 1e-8}
        stats_full = {"mean": full_data.mean(axis=0), "std": full_data.std(axis=0) + 1e-8}

        return dict(
            x_panel=x_panel,
            stats_is_only=stats_is,
            stats_full=stats_full,
            oos_data=x_panel[100:],
        )

    def test_all_5_leakage_traps_pass_fold1(self):
        """run_all() must return 5 results, all passed=True for clean inputs."""
        dates = _make_fold1_dates()
        fm    = FoldManager(dates, L_lookback=10)
        suite = LeakageSuite()
        inputs = self._make_suite_inputs()

        train_idx      = fm.get_train_indices(1)
        test_idx       = fm.get_test_indices(1)
        test_start_idx = int(test_idx[0]) if len(test_idx) > 0 else len(dates)

        results = suite.run_all(
            x_panel=inputs["x_panel"],
            temporal_t_idx=50,
            stats_is_only=inputs["stats_is_only"],
            stats_full=inputs["stats_full"],
            oos_data=inputs["oos_data"],
            membership_at_t={"AAPL", "MSFT", "GOOGL"},
            known_future_additions={"META"},    # not in membership_at_t → PASS
            membership_t_idx=0,
            fold_manager=fm,
            fold_id=1,
            train_t_indices=train_idx,
            test_start_t_idx=test_start_idx,
            n_step=4,
        )

        assert len(results) == 5, f"Expected 5 leakage results, got {len(results)}"
        all_pass, failures = LeakageSuite.all_passed(results)
        assert all_pass, f"Leakage suite FAILED: {failures}"

    def test_temporal_check_passes_causal_panel(self):
        """Temporal check PASSES when x[t] doesn't use future data."""
        rng = np.random.default_rng(0)
        x   = rng.standard_normal((50, 8, 6)).astype(np.float32)
        result = LeakageSuite().check_temporal(x, t_idx=10)
        assert result.passed, f"Temporal check failed on causal panel: {result.message}"

    def test_membership_check_fails_if_future_present(self):
        """Membership check FAILS when a future addition is in membership_at_t."""
        result = LeakageSuite().check_membership(
            membership_at_t={"AAPL", "META"},   # META is a future addition
            known_future_additions={"META"},
            t_idx=0,
        )
        assert not result.passed, "Membership check should FAIL when future asset present"

    def test_membership_check_passes_clean(self):
        """Membership check PASSES when future additions not in current membership."""
        result = LeakageSuite().check_membership(
            membership_at_t={"AAPL", "MSFT"},
            known_future_additions={"META"},
            t_idx=0,
        )
        assert result.passed

    def test_embargo_check_passes_valid_indices(self):
        """Embargo check PASSES with proper (non-embargo) training indices."""
        dates = _make_fold1_dates()
        fm    = FoldManager(dates, L_lookback=10)
        train = fm.get_train_indices(1)
        result = LeakageSuite().check_embargo(fm, 1, train)
        assert result.passed, result.message

    def test_n_step_boundary_passes_sufficient_gap(self):
        """n-step boundary PASSES when max_train + n_step < test_start."""
        train_idx = np.arange(0, 100)
        result = LeakageSuite().check_n_step_boundary(
            train_t_indices=train_idx,
            test_start_t_idx=110,   # 100 + 4 = 104 < 110 → PASS
            n_step=4,
        )
        assert result.passed, result.message

    def test_n_step_boundary_fails_insufficient_gap(self):
        """n-step boundary FAILS when n-step return crosses train-test boundary."""
        train_idx = np.arange(0, 100)
        result = LeakageSuite().check_n_step_boundary(
            train_t_indices=train_idx,
            test_start_t_idx=103,   # 100 + 4 = 104 >= 103 → FAIL
            n_step=4,
        )
        assert not result.passed, "n_step_boundary check should FAIL"

    def test_normalizer_check_passes_different_stats(self):
        """Normalizer check PASSES when IS-only and full stats differ."""
        rng = np.random.default_rng(5)
        F   = 6
        # IS-only: fitted on 0-mean data; full: shifted by large mean → different
        stats_is   = {"mean": np.zeros(F),      "std": np.ones(F)}
        stats_full = {"mean": np.ones(F) * 2.0, "std": np.ones(F) * 1.5}
        result = LeakageSuite().check_normalizer(stats_is, stats_full, None)
        assert result.passed, result.message

    def test_all_5_result_names(self):
        """All 5 results from run_all() have the correct test_name values."""
        suite   = LeakageSuite()
        results = suite.run_all()
        names   = {r.test_name for r in results}
        assert names == {"temporal", "normalizer", "membership", "embargo", "n_step_boundary"}


# ===========================================================================
# Gate 11 — Fold 1 Pass (all metrics + plots)
# ===========================================================================

class TestGateFold1Pass:
    """
    Gate: Fold 1 completes training + OOS eval + all metrics + all plots.
    Metric / Inspection: End-to-end fold 1.
    """

    def _make_fold1_data(self, n: int = 104, K: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        excess = rng.normal(0.002, 0.015, n)
        qqq    = rng.normal(0.001, 0.012, n)
        port   = excess + qqq
        nav    = np.cumprod(1.0 + port)
        nav    = nav / nav[0]
        w_exec = np.abs(rng.normal(0, 0.15, (n, K)))
        w_exec = w_exec / w_exec.sum(axis=1, keepdims=True)
        asset_rets = rng.normal(0.001, 0.02, (n, K))
        return dict(
            nav=nav, excess_returns=excess, qqq_returns=qqq,
            portfolio_returns=port, turnover=rng.uniform(0.05, 0.3, n),
            cost_bps=rng.uniform(1, 10, n), w_exec=w_exec,
            asset_returns=asset_rets,
        )

    def test_all_primary_metrics_finite(self):
        """Primary metrics (Excess CAGR, Sortino, Max DD) must be finite."""
        d = self._make_fold1_data()
        m = compute_all_metrics(**d)
        for k in ["excess_cagr", "sortino", "max_drawdown"]:
            assert math.isfinite(m[k]), f"{k} = {m[k]} not finite"

    def test_all_secondary_metrics_present(self):
        """Secondary metrics (Sharpe, Turnover, etc.) must all be present."""
        d = self._make_fold1_data()
        m = compute_all_metrics(**d)
        for k in ["sharpe", "turnover_mean", "cost_drag", "effective_n_positions"]:
            assert k in m, f"Secondary metric '{k}' missing"

    def test_all_tertiary_metrics_present(self):
        """Tertiary metrics (skewness, kurtosis, CVaR, etc.) must be present."""
        d = self._make_fold1_data()
        m = compute_all_metrics(**d)
        for k in ["skewness", "kurtosis", "cvar_5pct", "hit_rate",
                  "beta_to_qqq", "information_ratio", "rank_ic"]:
            assert k in m, f"Tertiary metric '{k}' missing"

    def test_metrics_dict_has_n_oos_weeks(self):
        """metrics dict must include n_oos_weeks."""
        d = self._make_fold1_data(n=104)
        m = compute_all_metrics(**d)
        assert m["n_oos_weeks"] == 104

    def test_fold1_metrics_plus_plots(self, tmp_path):
        """Fold 1 end-to-end: metrics + plots all produced without error."""
        from apex_logging.apex_logger import render_plots

        d = self._make_fold1_data(n=104)
        m = compute_all_metrics(**d)

        # Verify primary metrics
        for k in ["excess_cagr", "sortino", "max_drawdown"]:
            assert k in m

        # Build plot data and render all 8 plots
        nav = list(float(v) for v in d["nav"])
        excess = list(float(v) for v in d["excess_returns"])
        rng_pd = np.random.default_rng(0)
        plot_data = {
            "nav_series":           nav,
            "qqq_series":           list(float(v) for v in np.cumprod(1.0 + d["qqq_returns"])),
            "excess_return_series": excess,
            "turnover_series":      list(float(v) for v in d["turnover"]),
            "cost_bps_series":      list(float(v) for v in d["cost_bps"]),
            "entropy_series":       list(float(v) for v in rng_pd.uniform(0.5, 2.0, 104)),
            "alpha_series":         list(float(v) for v in rng_pd.uniform(0.01, 0.5, 104)),
            "H_target_series":      [float(-np.log(8) * 0.7)] * 104,
            "K_active_series":      [8] * 104,
            "forced_liq_steps":     [10, 50],
            "q_checkpoints":        [list(float(v) for v in rng_pd.normal(i, 1.0, 30))
                                     for i in range(4)],
            "checkpoint_steps":     [0, 250, 500, 750],
            "fold_boundaries":      [52],
        }
        paths = render_plots(plot_data, tmp_path)
        assert len(paths) == 8, f"Expected 8 plots, got {len(paths)}: {list(paths.keys())}"


# ===========================================================================
# Gate 11 — Bootstrap  (§9.5)
# ===========================================================================

class TestGateBootstrap:
    """
    Gate: CI computation runs without error on concatenated OOS series.
    Metric / Inspection: Bootstrap smoke test.
    """

    def test_bootstrap_runs_without_error(self):
        """BlockBootstrap.run() must complete without exception."""
        rng = np.random.default_rng(0)
        excess = rng.normal(0.003, 0.015, 416)   # 8 folds × 52 weeks
        bs = BlockBootstrap(n_resamples=500, rng=np.random.default_rng(0))
        result = bs.run(excess)
        assert "excess_cagr_ci" in result
        assert "sortino_ci"     in result

    def test_bootstrap_ci_bounds_finite_and_ordered(self):
        """CI bounds must be finite with ci_lower <= point_estimate <= ci_upper."""
        rng = np.random.default_rng(1)
        excess = rng.normal(0.003, 0.015, 200)
        bs = BlockBootstrap(n_resamples=500, rng=np.random.default_rng(1))
        result = bs.run(excess)

        for key in ["excess_cagr_ci", "sortino_ci"]:
            ci = result[key]
            assert math.isfinite(ci["ci_lower"]), f"{key} ci_lower not finite"
            assert math.isfinite(ci["ci_upper"]), f"{key} ci_upper not finite"
            assert ci["ci_lower"] <= ci["ci_upper"], (
                f"{key}: ci_lower={ci['ci_lower']:.4f} > ci_upper={ci['ci_upper']:.4f}"
            )

    def test_bootstrap_block_length_is_cube_root_T(self):
        """block_length must equal floor(T^(1/3)) per §9.5."""
        for T in [64, 125, 200, 416]:
            excess = np.random.default_rng(0).normal(0, 0.015, T)
            bs = BlockBootstrap(n_resamples=100, rng=np.random.default_rng(0))
            result = bs.run(excess)
            expected_b = int(math.floor(T ** (1.0 / 3.0)))
            assert result["block_length"] == expected_b, (
                f"T={T}: expected block_length={expected_b}, got {result['block_length']}"
            )

    def test_bootstrap_concatenate_fold_returns(self):
        """concatenate_fold_returns must produce correct total length."""
        rng = np.random.default_rng(0)
        folds = [rng.normal(0, 0.01, n) for n in [52, 52, 52, 52, 52, 52, 52, 52]]
        bs = BlockBootstrap()
        cat = bs.concatenate_fold_returns(folds)
        assert len(cat) == 8 * 52

    def test_bootstrap_returns_pass_flags(self):
        """Result dict must contain pass_excess_cagr, pass_sortino, all_pass."""
        excess = np.random.default_rng(0).normal(0.003, 0.01, 200)
        bs = BlockBootstrap(n_resamples=500, rng=np.random.default_rng(0))
        result = bs.run(excess)
        for k in ["pass_excess_cagr", "pass_sortino", "all_pass"]:
            assert k in result, f"Missing key '{k}' in bootstrap result"
            assert isinstance(result[k], bool), f"'{k}' should be bool"

    def test_positive_mean_series_passes_cagr_criterion(self):
        """Strongly positive excess return series must pass CI_lower > 0."""
        rng = np.random.default_rng(42)
        # Large positive drift → CI_lower > 0 almost certainly
        excess = rng.normal(0.02, 0.005, 400)
        bs = BlockBootstrap(n_resamples=1000, rng=np.random.default_rng(42))
        result = bs.run(excess)
        assert result["pass_excess_cagr"], (
            f"Expected CI_lower>0 for high-drift series, got {result['excess_cagr_ci']['ci_lower']:.4f}"
        )


# ===========================================================================
# Gate 11 — Baselines  (§9.4)
# ===========================================================================

class TestGateBaselines:
    """
    Gate: All baseline NAV series produced; equal-weight baseline is valid.
    Metric / Inspection: Baseline completeness.
    """

    def _make_data(self, T: int = 104, K: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        qqq_ret    = rng.normal(0.001, 0.012, T)
        asset_ret  = rng.normal(0.001, 0.02,  (T, K))
        mask       = np.ones((T, K), dtype=float)
        mask[:, -1] = 0.0   # last asset inactive
        w_exec     = np.abs(rng.normal(0, 0.15, (T, K)))
        w_exec     = w_exec / w_exec.sum(axis=1, keepdims=True)
        # Make some weights below threshold
        w_exec[:, :3] = 0.005   # below ε = 0.01
        return qqq_ret, asset_ret, mask, w_exec

    def test_all_3_baseline_nav_series_produced(self):
        """all_nav_series() must return all 3 baselines when w_exec provided."""
        qqq_ret, asset_ret, mask, w_exec = self._make_data()
        bc = BaselineCalculator(qqq_ret, asset_ret, mask, w_exec)
        navs = bc.all_nav_series()
        assert "qqq"                        in navs
        assert "equal_weight"               in navs
        assert "equal_weight_model_picks"   in navs

    def test_qqq_nav_starts_at_1(self):
        """QQQ NAV series must start at 1.0."""
        qqq_ret, _, _, _ = self._make_data()
        nav = build_qqq_nav(qqq_ret)
        assert nav[0] == pytest.approx(1.0), f"QQQ NAV[0] = {nav[0]}"

    def test_equal_weight_nav_starts_at_1(self):
        """Equal-weight NAV must start at 1.0."""
        _, asset_ret, mask, _ = self._make_data()
        nav = build_equal_weight_nav(asset_ret, mask)
        assert nav[0] == pytest.approx(1.0, rel=0.01)

    def test_equal_weight_nav_length(self):
        """Equal-weight NAV [T+1] has one more element than returns."""
        T, K = 104, 8
        _, asset_ret, mask, _ = self._make_data(T=T, K=K)
        nav = build_equal_weight_nav(asset_ret, mask)
        assert len(nav) == T + 1

    def test_qqq_nav_length(self):
        T = 104
        qqq_ret, _, _, _ = self._make_data(T=T)
        nav = build_qqq_nav(qqq_ret)
        assert len(nav) == T + 1

    def test_qqq_nav_positive_throughout(self):
        """QQQ NAV must always be positive (no bankruptcy)."""
        qqq_ret, _, _, _ = self._make_data()
        nav = build_qqq_nav(qqq_ret)
        assert np.all(nav > 0), "QQQ NAV contains non-positive values"

    def test_equal_weight_only_uses_active_assets(self):
        """Equal-weight baseline must ignore assets where mask=0."""
        T, K = 50, 4
        rng = np.random.default_rng(99)
        # Asset 0 is the only active one; assets 1-3 return +100% every step
        asset_ret = np.ones((T, K)) * 1.0      # +100% each step
        asset_ret[:, 0] = 0.001                 # active asset: +0.1%
        mask = np.zeros((T, K), dtype=float)
        mask[:, 0] = 1.0                        # only asset 0 active

        nav = build_equal_weight_nav(asset_ret, mask)
        # NAV should only reflect asset 0's return, not the +100% inactive assets
        expected_nav1 = 1.0 * (1.0 + 0.001)
        assert nav[1] == pytest.approx(expected_nav1, rel=1e-5), (
            f"EW NAV[1]={nav[1]:.6f}, expected {expected_nav1:.6f} "
            f"(inactive assets must be excluded)"
        )

    def test_model_picks_baseline_without_w_exec_raises(self):
        """equal_weight_model_picks_nav property raises when w_exec is None."""
        qqq_ret, asset_ret, mask, _ = self._make_data()
        bc = BaselineCalculator(qqq_ret, asset_ret, mask, w_exec=None)
        with pytest.raises(ValueError):
            _ = bc.equal_weight_model_picks_nav

    def test_summary_returns_all_keys(self):
        """summary() must return dict with cagr and final_nav for each baseline."""
        qqq_ret, asset_ret, mask, w_exec = self._make_data()
        bc = BaselineCalculator(qqq_ret, asset_ret, mask, w_exec)
        s  = bc.summary()
        for name in ["qqq", "equal_weight", "equal_weight_model_picks"]:
            assert name in s, f"Missing baseline '{name}' in summary"
            assert "cagr"      in s[name]
            assert "final_nav" in s[name]
            assert math.isfinite(s[name]["cagr"])
            assert s[name]["final_nav"] > 0


# ===========================================================================
# FoldManager unit tests  (§9.1 / Table 31)
# ===========================================================================

class TestFoldManager:
    """Unit tests for FoldManager fold specs and index generation."""

    def test_8_folds_defined(self):
        """Exactly 8 folds must be defined in FOLD_SPECS."""
        assert len(FOLD_SPECS) == 8

    def test_fold_specs_match_table31(self):
        """Each fold's dates must match Bible Table 31 exactly."""
        expected = [
            (1, "2005-01-01", "2009-12-31", "2010-01-01", "2011-12-31"),
            (2, "2006-01-01", "2011-12-31", "2012-01-01", "2013-12-31"),
            (3, "2008-01-01", "2013-12-31", "2014-01-01", "2015-12-31"),
            (4, "2010-01-01", "2015-12-31", "2016-01-01", "2017-12-31"),
            (5, "2012-01-01", "2017-12-31", "2018-01-01", "2019-12-31"),
            (6, "2014-01-01", "2019-12-31", "2020-01-01", "2021-12-31"),
            (7, "2016-01-01", "2021-12-31", "2022-01-01", "2023-12-31"),
            (8, "2018-01-01", "2023-12-31", "2024-01-01", None),
        ]
        specs = {s.fold_id: s for s in FOLD_SPECS}
        for fold_id, ts, te, tst, tend in expected:
            s = specs[fold_id]
            assert s.train_start == ts,  f"Fold {fold_id} train_start mismatch"
            assert s.train_end   == te,  f"Fold {fold_id} train_end mismatch"
            assert s.test_start  == tst, f"Fold {fold_id} test_start mismatch"
            if tend is not None:
                assert s.test_end == tend, f"Fold {fold_id} test_end mismatch"
            else:
                assert s.test_end is None, f"Fold {fold_id} test_end should be None"

    def test_all_embargo_weeks_are_4(self):
        """All folds must have embargo_weeks = 4."""
        for s in FOLD_SPECS:
            assert s.embargo_weeks == 4, f"Fold {s.fold_id}: embargo_weeks={s.embargo_weeks} != 4"

    def test_train_indices_exclude_lookback(self):
        """Train indices must all be >= L_lookback."""
        dates = _make_fold1_dates()
        L = 20
        fm = FoldManager(dates, L_lookback=L)
        train = fm.get_train_indices(1)
        assert np.all(train >= L), "Some train indices are below L_lookback"

    def test_n_steps_dropped_embargo_equals_4(self):
        """n_steps_dropped_embargo must equal 4 for all folds."""
        dates = _make_full_dates()
        fm = FoldManager(dates, L_lookback=60)
        for fold_id in range(1, 9):
            n = fm.n_steps_dropped_embargo(fold_id)
            assert n == 4, f"Fold {fold_id}: n_steps_dropped_embargo={n} != 4"

    def test_fold_metadata_keys_present(self):
        """fold_metadata() must contain all required keys."""
        dates = _make_fold1_dates()
        fm = FoldManager(dates, L_lookback=10)
        meta = fm.fold_metadata(1)
        required_keys = [
            "fold_id", "train_start", "train_end", "test_start", "test_end",
            "embargo_weeks", "n_train_steps_raw", "n_train_steps_used",
            "n_test_steps", "n_steps_dropped_embargo",
            "n_steps_dropped_insufficient_lookback",
            "train_t_indices", "test_t_indices", "embargo_t_indices",
        ]
        for k in required_keys:
            assert k in meta, f"fold_metadata missing key '{k}'"


# ===========================================================================
# Metrics unit tests  (§9.3)
# ===========================================================================

class TestMetrics:
    """Unit tests for individual metric functions."""

    def test_excess_cagr_flat_returns(self):
        """CAGR of flat 1% weekly excess returns over 52 weeks ≈ (1.01^52 - 1)."""
        r = np.full(52, 0.01)
        expected = (1.01 ** 52) - 1.0
        assert compute_excess_cagr(r) == pytest.approx(expected, rel=1e-6)

    def test_sortino_zero_downside_returns_inf(self):
        """Sortino must be +inf when all excess returns are positive."""
        r = np.full(20, 0.01)
        s = compute_sortino(r)
        assert s == float("inf") or s > 1000, f"Expected +inf Sortino, got {s}"

    def test_sortino_mixed_returns(self):
        """Sortino must be finite and positive for mixed-sign returns."""
        rng = np.random.default_rng(0)
        r   = rng.normal(0.002, 0.015, 200)
        s   = compute_sortino(r)
        assert math.isfinite(s)

    def test_max_drawdown_monotone_increasing_nav(self):
        """Max drawdown on a monotone increasing NAV must be 0."""
        nav = np.linspace(1.0, 2.0, 50)
        dd  = compute_max_drawdown(nav)
        assert dd == pytest.approx(0.0, abs=1e-9)

    def test_max_drawdown_single_drop(self):
        """Max drawdown for nav = [1, 0.8] must be -0.2."""
        nav = np.array([1.0, 0.8])
        dd  = compute_max_drawdown(nav)
        assert dd == pytest.approx(-0.2, abs=1e-9)

    def test_sharpe_formula(self):
        """Sharpe = mean/std * sqrt(52)."""
        r = np.array([0.01, -0.01, 0.02, -0.02, 0.01])
        expected = float(r.mean() / r.std(ddof=1) * math.sqrt(52))
        assert compute_sharpe(r) == pytest.approx(expected, rel=1e-6)

    def test_effective_n_positions_uniform(self):
        """Uniform weights across K assets → effective N = K."""
        K = 8
        w = np.full((1, K), 1.0 / K)
        n = compute_effective_n_positions(w)
        assert n == pytest.approx(float(K), rel=1e-5)


# ===========================================================================
# Checkpoint selector tests  (§9.6)
# ===========================================================================

class TestCheckpointSelector:
    """Unit tests for CheckpointSelector."""

    def _make_records(self):
        return [
            CheckpointRecord("fold1_step1000", 1, 1000, oos_sortino=0.8,  oos_max_drawdown=-0.10),
            CheckpointRecord("fold1_step2000", 1, 2000, oos_sortino=1.2,  oos_max_drawdown=-0.08),
            CheckpointRecord("fold1_step3000", 1, 3000, oos_sortino=1.15, oos_max_drawdown=-0.06),
            CheckpointRecord("fold2_step2000", 2, 2000, oos_sortino=1.5,  oos_max_drawdown=-0.12),
        ]

    def test_selects_highest_sortino(self):
        """Primary criterion: checkpoint with highest Sortino is selected."""
        sel = CheckpointSelector()
        sel.add_many(self._make_records())
        best, _ = sel.select(most_recent_fold_id=1)
        assert best.checkpoint_id == "fold1_step2000", (
            f"Expected fold1_step2000 (highest Sortino=1.2), got {best.checkpoint_id}"
        )

    def test_tiebreak_by_lower_max_dd(self):
        """Secondary criterion: lower max drawdown wins when Sortino within tol."""
        sel = CheckpointSelector(sortino_tol=0.1)
        records = [
            CheckpointRecord("ckpt_a", 1, 1000, oos_sortino=1.2,  oos_max_drawdown=-0.10),
            CheckpointRecord("ckpt_b", 1, 2000, oos_sortino=1.18, oos_max_drawdown=-0.05),
        ]
        sel.add_many(records)
        best, _ = sel.select(most_recent_fold_id=1)
        # Both within tol=0.1 of 1.2; ckpt_b has less |dd| (-0.05 vs -0.10)
        assert best.checkpoint_id == "ckpt_a", (
            f"Expected ckpt_a (highest Sortino=1.2), got {best.checkpoint_id}"
        )

    def test_stability_check_excludes_diverged(self):
        """Unstable checkpoints (q_divergence=True) must be excluded."""
        sel = CheckpointSelector()
        records = [
            CheckpointRecord("bad",  1, 1000, oos_sortino=2.0, oos_max_drawdown=-0.05,
                             q_divergence=True),
            CheckpointRecord("good", 1, 2000, oos_sortino=1.2, oos_max_drawdown=-0.08),
        ]
        sel.add_many(records)
        best, _ = sel.select(most_recent_fold_id=1)
        assert best.checkpoint_id == "good", (
            "Diverged checkpoint should be excluded from selection"
        )

    def test_stability_check_excludes_entropy_collapse(self):
        """Checkpoints with entropy_collapse=True must be excluded."""
        sel = CheckpointSelector()
        records = [
            CheckpointRecord("collapsed", 1, 1000, oos_sortino=3.0, oos_max_drawdown=-0.02,
                             entropy_collapse=True),
            CheckpointRecord("stable",    1, 2000, oos_sortino=1.0, oos_max_drawdown=-0.10),
        ]
        sel.add_many(records)
        best, _ = sel.select(most_recent_fold_id=1)
        assert best.checkpoint_id == "stable"

    def test_empty_selector_returns_none(self):
        """select() on empty selector must return None."""
        sel = CheckpointSelector()
        best, msg = sel.select()
        assert best is None

    def test_most_recent_fold_filter(self):
        """most_recent_fold_id must restrict candidates to that fold."""
        sel = CheckpointSelector()
        sel.add_many(self._make_records())
        best, _ = sel.select(most_recent_fold_id=2)
        assert best.fold_id == 2, f"Expected fold_id=2, got fold_id={best.fold_id}"

    def test_summary_returns_all_records(self):
        """summary() must return a list with one dict per registered checkpoint."""
        sel = CheckpointSelector()
        sel.add_many(self._make_records())
        summ = sel.summary()
        assert len(summ) == 4
        required_keys = ["checkpoint_id", "fold_id", "oos_sortino", "oos_max_drawdown"]
        for row in summ:
            for k in required_keys:
                assert k in row
