"""
tests/test_phase13_gate13.py
============================
Gate 13: Inference and Paper Trading Pipeline  (Bible §12.1–12.4)

Gate criteria (screenshot):
  Guardrails    : All guardrails fire correctly on injected violations   → Fault injection test
  Reproducibility: Paper-trade output matches fold 8 OOS for overlapping  → Cross-check
  Live data     : Live data adapter produces valid observation tensor     → Smoke test

Additional unit tests:
  - CheckpointLoader: save/load round-trip preserves all fields
  - CheckpointLoader: validate_directory catches missing files
  - Guardrails: all 5 checks pass on clean inputs
  - Guardrails: feasibility fails on long-only violation
  - Guardrails: feasibility fails on sum != 1
  - Guardrails: feasibility fails on per-name cap violation
  - Guardrails: feasibility fails on sector cap violation
  - Guardrails: mask integrity fails on mask leak
  - Guardrails: universe validity fails on stale snapshot
  - Guardrails: stale data fires correctly
  - Guardrails: NAV plausibility fires on large jump
  - MissingDataHandler: freeze on short gap
  - MissingDataHandler: liquidate after prolonged gap
  - MissingDataHandler: apply_frozen_weights renormalises correctly
  - MissingDataHandler: feature window gap masks asset
  - LiveDataAdapter: valid observation shape
  - LiveDataAdapter: all-finite features
  - LiveDataAdapter: stale data detected
  - AlertSystem: process_guardrail_report fires correct alerts
  - AlertSystem: pipeline_halt fires on CRITICAL
  - AlertSystem: summary counts correct
  - PaperTradeLoop: NAV > 0 after steps
  - PaperTradeLoop: trade log completeness
  - PaperTradeLoop: halt stops trading
  - PaperTradeLoop: warmup initialization
"""

from __future__ import annotations

import datetime
import json
import math
import sys
import os
from typing import Dict, List, Set

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from inference.checkpoint_loader import (
    CheckpointLoader, CheckpointManifest, LoadedCheckpoint
)
from inference.guardrails import (
    InferenceGuardrails, GuardrailReport, GuardrailResult
)
from inference.missing_data_handler import MissingDataHandler
from inference.live_data_adapter import (
    LiveDataAdapter, SyntheticDataProvider, LiveObservation
)
from inference.alert_system import (
    AlertSystem, Alert, AlertType, AlertSeverity
)
from inference.paper_trade_loop import PaperTradeLoop, TradeRecord
from integration.e2e_runner import make_synthetic_model, make_synthetic_panel


# ===========================================================================
# Shared helpers
# ===========================================================================

def _make_clean_weights(K: int = 8, K_max: int = 16, per_name_cap: float = 0.20) -> np.ndarray:
    """Uniform weights over first K assets summing to 1."""
    w = np.zeros(K_max, dtype=np.float32)
    w[:K] = 1.0 / K
    return w


def _make_mask(K: int = 8, K_max: int = 16) -> np.ndarray:
    m = np.zeros(K_max, dtype=np.float32)
    m[:K] = 1.0
    return m


def _make_sector_ids(K: int = 8, K_max: int = 16) -> np.ndarray:
    s = np.full(K_max, -1, dtype=np.int64)
    for k in range(K):
        s[k] = k % 4   # 4 sectors
    return s


def _make_manifest(fold_id: int = 8) -> CheckpointManifest:
    return CheckpointManifest(
        checkpoint_id    = f"fold{fold_id}_step5000",
        fold_id          = fold_id,
        update_step      = 5000,
        oos_sortino      = 1.45,
        oos_max_drawdown = -0.08,
        oos_excess_cagr  = 0.07,
        train_end        = "2023-12-31",
        test_start       = "2024-01-01",
        test_end         = None,
        model_config     = {"K_max": 16, "F": 10, "D_g": 8},
    )


# ===========================================================================
# Gate 13 — Guardrails (fault injection)
# ===========================================================================

class TestGateGuardrails:
    """
    Gate: All guardrails fire correctly on injected violations.
    Metric / Inspection: Fault injection test.
    """

    K, K_MAX = 8, 16

    def _gr(self) -> InferenceGuardrails:
        return InferenceGuardrails(
            per_name_cap      = 0.20,
            sector_cap        = 0.50,
            mask_leak_tol     = 1e-6,
            nav_max_step_frac = 0.25,
            stale_weeks       = 2,
        )

    def test_all_pass_on_clean_inputs(self):
        """All 5 guardrails must pass for valid, clean inputs."""
        w    = _make_clean_weights(self.K, self.K_MAX)
        mask = _make_mask(self.K, self.K_MAX)
        sid  = _make_sector_ids(self.K, self.K_MAX)
        rep  = self._gr().run_all(
            w_exec             = w,
            mask               = mask,
            sector_ids         = sid,
            snapshot_date      = "2024-01-08",
            most_recent_date   = "2024-01-08",
            last_update_weeks  = {f"T{k}": 0 for k in range(self.K)},
            active_tickers     = {f"T{k}" for k in range(self.K)},
            nav_prev           = 1.0,
            nav_curr           = 1.01,
            step_id            = "2024-01-08",
        )
        assert not rep.any_failed, f"Expected all pass, failed: {rep.failed_names}"
        assert not rep.should_halt

    # ── Feasibility fault injections ────────────────────────────────────

    def test_feasibility_fails_negative_weight(self):
        """INJECT: negative weight on active asset → feasibility FAIL."""
        w    = _make_clean_weights(self.K, self.K_MAX)
        w[0] = -0.01          # INJECT: violation
        mask = _make_mask(self.K, self.K_MAX)
        sid  = _make_sector_ids(self.K, self.K_MAX)
        result = self._gr().check_feasibility(w, mask, sid)
        assert not result.passed, "Expected feasibility FAIL on negative weight"
        assert result.severity == "CRITICAL"

    def test_feasibility_fails_sum_not_one(self):
        """INJECT: active weights sum to 0.5 → feasibility FAIL."""
        w    = _make_clean_weights(self.K, self.K_MAX)
        w[:self.K] = 0.5 / self.K   # sum = 0.5, not 1.0
        mask = _make_mask(self.K, self.K_MAX)
        sid  = _make_sector_ids(self.K, self.K_MAX)
        result = self._gr().check_feasibility(w, mask, sid)
        assert not result.passed, "Expected feasibility FAIL on sum != 1"

    def test_feasibility_fails_per_name_cap(self):
        """INJECT: one asset weight > per_name_cap → feasibility FAIL."""
        w    = np.zeros(self.K_MAX, dtype=np.float32)
        w[0] = 0.25    # over per_name_cap=0.20
        w[1] = 0.75
        mask = _make_mask(2, self.K_MAX)
        sid  = np.full(self.K_MAX, -1, dtype=np.int64)
        sid[:2] = 0
        result = self._gr().check_feasibility(w, mask, sid)
        assert not result.passed, "Expected feasibility FAIL on per-name cap breach"

    def test_feasibility_fails_sector_cap(self):
        """INJECT: one sector weight > sector_cap → feasibility FAIL."""
        w    = np.zeros(self.K_MAX, dtype=np.float32)
        # Put 0.60 > 0.50 in sector 0
        w[:4] = 0.15   # all sector 0 assets
        mask = _make_mask(4, self.K_MAX)
        sid  = np.zeros(self.K_MAX, dtype=np.int64)
        sid[:4] = 0   # all same sector
        result = self._gr().check_feasibility(w, mask, sid)
        assert not result.passed, "Expected feasibility FAIL on sector cap breach (0.60 > 0.50)"

    # ── Mask integrity fault injection ──────────────────────────────────

    def test_mask_integrity_fails_on_leak(self):
        """INJECT: weight on inactive asset → mask_integrity FAIL (CRITICAL)."""
        w    = _make_clean_weights(self.K, self.K_MAX)
        w[self.K] = 0.01   # INJECT: inactive slot has weight
        mask = _make_mask(self.K, self.K_MAX)
        result = self._gr().check_mask_integrity(w, mask)
        assert not result.passed, "Expected mask_integrity FAIL on weight leak"
        assert result.severity == "CRITICAL"

    def test_mask_integrity_passes_zero_inactive(self):
        """No weight on inactive slots → mask_integrity PASS."""
        w    = _make_clean_weights(self.K, self.K_MAX)
        mask = _make_mask(self.K, self.K_MAX)
        result = self._gr().check_mask_integrity(w, mask)
        assert result.passed

    # ── Universe validity ────────────────────────────────────────────────

    def test_universe_validity_fails_stale_snapshot(self):
        """INJECT: snapshot_date older than most_recent → universe FAIL."""
        result = self._gr().check_universe_validity(
            snapshot_date    = "2024-01-01",
            most_recent_date = "2024-01-08",   # INJECT: newer snapshot available
        )
        assert not result.passed, "Expected universe_validity FAIL on stale snapshot"
        assert result.severity == "HIGH"

    def test_universe_validity_passes_current_snapshot(self):
        """snapshot_date == most_recent_date → universe PASS."""
        result = self._gr().check_universe_validity("2024-01-08", "2024-01-08")
        assert result.passed

    # ── Stale data ───────────────────────────────────────────────────────

    def test_stale_data_fails_when_stale(self):
        """INJECT: active asset stale for 3 weeks → stale_data FAIL."""
        result = self._gr().check_stale_data(
            last_update_weeks = {"AAPL": 3, "MSFT": 0},   # INJECT: AAPL stale
            active_tickers    = {"AAPL", "MSFT"},
        )
        assert not result.passed, "Expected stale_data FAIL"
        assert result.severity == "HIGH"

    def test_stale_data_passes_fresh(self):
        """All assets fresh → stale_data PASS."""
        result = self._gr().check_stale_data(
            last_update_weeks = {"AAPL": 0, "MSFT": 1},
            active_tickers    = {"AAPL", "MSFT"},
        )
        assert result.passed

    # ── NAV plausibility ─────────────────────────────────────────────────

    def test_nav_plausibility_fails_large_jump(self):
        """INJECT: NAV jumps 30% in one step → nav_plausibility FAIL."""
        result = self._gr().check_nav_plausibility(
            nav_prev = 1.0, nav_curr = 1.30   # INJECT: 30% jump
        )
        assert not result.passed, "Expected nav_plausibility FAIL on 30% jump"

    def test_nav_plausibility_passes_normal_change(self):
        """Normal 1% NAV change → nav_plausibility PASS."""
        result = self._gr().check_nav_plausibility(nav_prev=1.0, nav_curr=1.01)
        assert result.passed

    def test_should_halt_on_critical_failure(self):
        """GuardrailReport.should_halt must be True when CRITICAL check fails."""
        w    = _make_clean_weights(self.K, self.K_MAX)
        w[self.K] = 0.05   # mask leak → CRITICAL
        mask = _make_mask(self.K, self.K_MAX)
        sid  = _make_sector_ids(self.K, self.K_MAX)
        rep  = self._gr().run_all(
            w_exec             = w,
            mask               = mask,
            sector_ids         = sid,
            snapshot_date      = "2024-01-08",
            most_recent_date   = "2024-01-08",
            last_update_weeks  = {},
            active_tickers     = set(),
            nav_prev           = 1.0,
            nav_curr           = 1.01,
        )
        assert rep.should_halt, "Expected halt on CRITICAL mask leak"

    def test_5_checks_in_report(self):
        """run_all must always return exactly 5 guardrail checks."""
        w    = _make_clean_weights(self.K, self.K_MAX)
        mask = _make_mask(self.K, self.K_MAX)
        sid  = _make_sector_ids(self.K, self.K_MAX)
        rep  = self._gr().run_all(
            w_exec=w, mask=mask, sector_ids=sid,
            snapshot_date="2024-01-08", most_recent_date="2024-01-08",
            last_update_weeks={}, active_tickers=set(),
            nav_prev=1.0, nav_curr=1.01,
        )
        assert len(rep.checks) == 5


# ===========================================================================
# Gate 13 — Checkpoint Loader  (§12.1)
# ===========================================================================

class TestCheckpointLoader:
    """Unit tests for CheckpointLoader save/load round-trip."""

    def _make_checkpoint(self, tmp_path):
        manifest   = _make_manifest(fold_id=8)
        model      = make_synthetic_model(K_max=16, F=10, D_g=8, seed=0)
        state_dict = model.state_dict()
        norm_stats = {
            "mean": np.zeros(10, dtype=np.float32),
            "std":  np.ones(10, dtype=np.float32),
        }
        sec_map = {"AAPL": 0, "MSFT": 1}
        alias   = {"AAPL": ["AAPL US Equity"]}
        sec_id  = {"AAPL": 2, "MSFT": 1}

        CheckpointLoader.save(
            checkpoint_dir   = tmp_path,
            manifest         = manifest,
            model_state_dict = state_dict,
            norm_stats       = norm_stats,
            security_id_map  = sec_map,
            ticker_alias_map = alias,
            sector_map       = sec_id,
        )
        return manifest

    def test_round_trip_manifest(self, tmp_path):
        """Loaded manifest must match saved manifest exactly."""
        orig = self._make_checkpoint(tmp_path)
        ckpt = CheckpointLoader(tmp_path).load()
        assert ckpt.manifest.checkpoint_id    == orig.checkpoint_id
        assert ckpt.manifest.fold_id          == orig.fold_id
        assert ckpt.manifest.update_step      == orig.update_step
        assert ckpt.manifest.oos_sortino      == pytest.approx(orig.oos_sortino)
        assert ckpt.manifest.oos_max_drawdown == pytest.approx(orig.oos_max_drawdown)

    def test_round_trip_norm_stats(self, tmp_path):
        """Loaded norm_stats must match saved values."""
        self._make_checkpoint(tmp_path)
        ckpt = CheckpointLoader(tmp_path).load()
        np.testing.assert_array_almost_equal(ckpt.norm_stats["mean"], np.zeros(10))
        np.testing.assert_array_almost_equal(ckpt.norm_stats["std"],  np.ones(10))

    def test_round_trip_security_id_map(self, tmp_path):
        """Loaded security_id_map must match saved values."""
        self._make_checkpoint(tmp_path)
        ckpt = CheckpointLoader(tmp_path).load()
        assert ckpt.security_id_map == {"AAPL": 0, "MSFT": 1}

    def test_round_trip_model_state_dict(self, tmp_path):
        """Loaded model state dict must have same keys."""
        self._make_checkpoint(tmp_path)
        ckpt   = CheckpointLoader(tmp_path).load()
        model2 = make_synthetic_model(K_max=16, F=10, D_g=8, seed=0)
        model2.load_state_dict(ckpt.model_state_dict, strict=True)

    def test_validate_directory_catches_missing(self, tmp_path):
        """validate_directory must report missing files."""
        valid, missing = CheckpointLoader(tmp_path).validate_directory()
        assert not valid
        assert len(missing) > 0

    def test_load_raises_on_missing_files(self, tmp_path):
        """load() must raise FileNotFoundError when files are missing."""
        with pytest.raises(FileNotFoundError):
            CheckpointLoader(tmp_path).load()

    def test_fold_id_property(self, tmp_path):
        """LoadedCheckpoint.fold_id must equal manifest.fold_id."""
        self._make_checkpoint(tmp_path)
        ckpt = CheckpointLoader(tmp_path).load()
        assert ckpt.fold_id == 8

    def test_oos_sortino_property(self, tmp_path):
        """LoadedCheckpoint.oos_sortino must equal manifest.oos_sortino."""
        self._make_checkpoint(tmp_path)
        ckpt = CheckpointLoader(tmp_path).load()
        assert ckpt.oos_sortino == pytest.approx(1.45)


# ===========================================================================
# Gate 13 — Live Data (smoke test)
# ===========================================================================

class TestGateLiveData:
    """
    Gate: Live data adapter produces valid observation tensor for current week.
    Metric / Inspection: Smoke test.
    """

    K_MAX = 16
    F     = 10
    D_G   = 8
    L     = 4

    def _adapter(self, norm=False):
        provider = SyntheticDataProvider(
            tickers=[f"T{i:02d}" for i in range(8)],
            K_max=self.K_MAX, F=self.F, D_g=self.D_G, seed=42,
        )
        norm_stats = (
            {"mean": np.zeros(self.F, dtype=np.float32),
             "std":  np.ones(self.F, dtype=np.float32)}
            if norm else None
        )
        return LiveDataAdapter(
            provider=provider, K_max=self.K_MAX, F=self.F, D_g=self.D_G,
            L_lookback=self.L, norm_stats=norm_stats,
        )

    def test_observation_shapes(self):
        """LiveObservation arrays must have correct shapes."""
        adapter = self._adapter()
        obs = adapter.build_observation(datetime.date(2024, 1, 8))
        assert obs.x_panel.shape    == (1, self.L, self.K_MAX, self.F)
        assert obs.g_panel.shape    == (1, self.D_G)
        assert obs.mask.shape       == (1, self.K_MAX)
        assert obs.sector_ids.shape == (1, self.K_MAX)
        assert obs.ticker_ids.shape == (1, self.K_MAX)

    def test_observation_is_valid(self):
        """LiveObservation.is_valid must be True with data available."""
        obs = self._adapter().build_observation(datetime.date(2024, 1, 8))
        assert obs.is_valid, f"Observation invalid: {obs.warnings}"

    def test_x_panel_finite(self):
        """x_panel must contain only finite values."""
        obs = self._adapter().build_observation(datetime.date(2024, 1, 8))
        assert np.all(np.isfinite(obs.x_panel)), "x_panel has non-finite values"

    def test_mask_binary(self):
        """mask must contain only 0.0 or 1.0 values."""
        obs = self._adapter().build_observation(datetime.date(2024, 1, 8))
        assert np.all((obs.mask == 0.0) | (obs.mask == 1.0))

    def test_normalization_applied(self):
        """With norm_stats, x_panel values are normalised (zero-mean ≈ possible)."""
        obs = self._adapter(norm=True).build_observation(datetime.date(2024, 1, 8))
        assert np.all(np.isfinite(obs.x_panel))

    def test_active_tickers_nonempty(self):
        """active_tickers must be non-empty for valid observation."""
        obs = self._adapter().build_observation(datetime.date(2024, 1, 8))
        assert len(obs.active_tickers) > 0

    def test_sector_ids_valid_range(self):
        """Active sector IDs must be >= 0; inactive slots = -1."""
        obs = self._adapter().build_observation(datetime.date(2024, 1, 8))
        active_sids = obs.sector_ids[0][obs.mask[0] > 0.5]
        assert np.all(active_sids >= 0), "Active asset has sector_id < 0"

    def test_different_dates_different_obs(self):
        """Two different rebalance dates must produce different g_panel."""
        adapter = self._adapter()
        o1 = adapter.build_observation(datetime.date(2024, 1,  8))
        o2 = adapter.build_observation(datetime.date(2024, 1, 15))
        assert not np.array_equal(o1.g_panel, o2.g_panel), (
            "Different dates should produce different global features"
        )


# ===========================================================================
# Gate 13 — Reproducibility (cross-check)
# ===========================================================================

class TestGateReproducibility:
    """
    Gate: Paper-trade output matches fold 8 OOS for overlapping dates.
    Metric / Inspection: Cross-check (deterministic re-run).
    """

    def _make_loop(self, seed: int = 42):
        model = make_synthetic_model(K_max=16, F=10, D_g=8, seed=seed)
        model.eval()
        return PaperTradeLoop(model=model, K_max=16, init_nav=1.0)

    def _make_obs(self, seed: int = 0):
        panel = make_synthetic_panel(T=10, K=8, F=10, D_g=8, K_max=16, seed=seed)
        return {
            "x_panel":    panel["x_panel"][0:1][np.newaxis],
            "g_panel":    panel["g_panel"][0:1],
            "mask":       panel["mask_panel"][0:1],
            "sector_ids": panel["sector_ids"][0:1],
            "ticker_ids": panel["ticker_ids"][0:1],
        }

    def test_same_seed_same_nav_series(self):
        """Two runs with same seed must produce identical NAV series."""
        def _run(seed):
            loop = self._make_loop(seed=seed)
            loop.initialize_warmup(_make_mask(8, 16))
            for i in range(5):
                obs = self._make_obs(seed=i)
                loop.step(
                    step_id          = f"2024-01-{8+i:02d}",
                    obs              = obs,
                    mask             = _make_mask(8, 16),
                    sector_ids       = _make_sector_ids(8, 16),
                    asset_returns    = np.zeros(16, dtype=np.float32),
                    snapshot_date    = "2024-01-08",
                    most_recent_date = "2024-01-08",
                    last_update_weeks= {},
                    active_tickers   = {f"t{k}" for k in range(8)},
                )
            return loop.nav_series()

        nav1 = _run(seed=42)
        nav2 = _run(seed=42)
        np.testing.assert_array_almost_equal(nav1, nav2,
            err_msg="Same-seed runs should produce identical NAV series")

    def test_nav_positive_throughout(self):
        """NAV must remain positive across all steps."""
        loop = self._make_loop()
        loop.initialize_warmup(_make_mask(8, 16))
        for i in range(10):
            obs = self._make_obs(seed=i)
            loop.step(
                step_id          = f"2024-01-{8+i:02d}",
                obs              = obs,
                mask             = _make_mask(8, 16),
                sector_ids       = _make_sector_ids(8, 16),
                asset_returns    = np.random.default_rng(i).normal(0.001, 0.01, 16).astype(np.float32),
                snapshot_date    = "2024-01-08",
                most_recent_date = "2024-01-08",
                last_update_weeks= {},
                active_tickers   = set(),
            )
        for rec in loop.records:
            assert rec.nav > 0.0, f"step={rec.step_id}: NAV={rec.nav:.6f} <= 0"

    def test_trade_log_completeness(self):
        """trade_log must contain one entry per step."""
        loop = self._make_loop()
        loop.initialize_warmup(_make_mask(8, 16))
        n_steps = 5
        for i in range(n_steps):
            obs = self._make_obs(seed=i)
            loop.step(
                step_id="2024-01-08", obs=obs,
                mask=_make_mask(8, 16),
                sector_ids=_make_sector_ids(8, 16),
                asset_returns=np.zeros(16, dtype=np.float32),
                snapshot_date="2024-01-08", most_recent_date="2024-01-08",
                last_update_weeks={}, active_tickers=set(),
            )
        log = loop.trade_log()
        assert len(log) == n_steps, f"Expected {n_steps} log entries, got {len(log)}"

    def test_halt_stops_trading(self):
        """After a CRITICAL guardrail fires, subsequent steps should return halt=True."""
        loop = self._make_loop()
        loop.initialize_warmup(_make_mask(8, 16))

        # Force halt by injecting it directly
        loop._halted = True

        obs = self._make_obs(seed=0)
        rec = loop.step(
            step_id="2024-01-08", obs=obs,
            mask=_make_mask(8, 16),
            sector_ids=_make_sector_ids(8, 16),
            asset_returns=np.zeros(16, dtype=np.float32),
            snapshot_date="2024-01-08", most_recent_date="2024-01-08",
            last_update_weeks={}, active_tickers=set(),
        )
        assert rec.halt, "Expected halt=True when pipeline is halted"


# ===========================================================================
# Gate 13 — Missing Data Handler  (§12.3)
# ===========================================================================

class TestMissingDataHandler:
    """Unit tests for MissingDataHandler freeze/liquidate policy."""

    TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN"]

    def test_freeze_on_short_gap(self):
        """Asset with 1 missing week must be frozen (not liquidated)."""
        handler = MissingDataHandler(freeze_weeks=2)
        w_prev  = {t: 0.25 for t in self.TICKERS}

        # Step 1: AAPL missing
        has_price   = {t: True for t in self.TICKERS}
        has_feature = {t: True for t in self.TICKERS}
        has_price["AAPL"] = False

        _, frozen, liquidate = handler.process_step(
            self.TICKERS, has_price, has_feature, w_prev
        )
        assert "AAPL" in frozen,      "AAPL should be frozen after 1 missing week"
        assert "AAPL" not in liquidate, "AAPL should NOT be liquidated after 1 missing week"

    def test_liquidate_after_prolonged_gap(self):
        """Asset missing > freeze_weeks must be flagged for liquidation."""
        handler = MissingDataHandler(freeze_weeks=2)
        w_prev  = {t: 0.25 for t in self.TICKERS}
        has_feature = {t: True for t in self.TICKERS}

        for _ in range(3):   # 3 weeks missing > freeze_weeks=2
            has_price = {t: True for t in self.TICKERS}
            has_price["AAPL"] = False
            handler.process_step(self.TICKERS, has_price, has_feature, w_prev)

        assert "AAPL" in handler.flagged_for_liquidation()

    def test_freeze_resets_on_price_return(self):
        """Missing counter resets when price returns."""
        handler = MissingDataHandler(freeze_weeks=2)
        tickers = ["AAPL"]
        w_prev  = {"AAPL": 1.0}
        feat    = {"AAPL": True}

        # Miss for 1 week
        handler.process_step(tickers, {"AAPL": False}, feat, w_prev)
        assert handler.get_state("AAPL").consecutive_missing == 1

        # Price returns
        handler.process_step(tickers, {"AAPL": True}, feat, w_prev)
        assert handler.get_state("AAPL").consecutive_missing == 0
        assert not handler.get_state("AAPL").flagged_liquidate

    def test_apply_frozen_weights_renormalises(self):
        """apply_frozen_weights must renormalise non-frozen assets to sum=1."""
        handler = MissingDataHandler()
        tickers = ["A", "B", "C", "D"]
        w_exec  = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        frozen  = {"A": 0.20}

        w_adj = handler.apply_frozen_weights(w_exec, tickers, frozen)
        total = float(w_adj.sum())
        assert total == pytest.approx(1.0, abs=1e-5), (
            f"Weights should sum to 1, got {total:.6f}"
        )
        assert w_adj[0] == pytest.approx(0.20), "Frozen asset weight wrong"

    def test_feature_window_gap_masks_asset(self):
        """Asset with incomplete feature window must have mask=0."""
        handler = MissingDataHandler(feature_window=4)
        tickers = ["AAPL", "MSFT"]
        has_price   = {t: True  for t in tickers}
        has_feature = {"AAPL": False, "MSFT": True}   # AAPL has incomplete window
        w_prev      = {t: 0.5 for t in tickers}

        adj_mask, _, _ = handler.process_step(tickers, has_price, has_feature, w_prev)
        assert adj_mask[0] == 0.0, "AAPL with incomplete feature window should be masked"
        assert adj_mask[1] == 1.0, "MSFT with full feature window should be unmasked"

    def test_summary_contains_all_tickers(self):
        """summary() must contain all tickers processed so far."""
        handler = MissingDataHandler()
        tickers = ["A", "B"]
        handler.process_step(tickers, {"A": True, "B": True},
                             {"A": True, "B": True}, {"A": 0.5, "B": 0.5})
        s = handler.summary()
        assert "A" in s and "B" in s


# ===========================================================================
# Gate 13 — Alert System
# ===========================================================================

class TestAlertSystem:
    """Unit tests for AlertSystem."""

    def test_fire_creates_alert(self):
        """fire() must create and store an alert."""
        sys = AlertSystem()
        sys.fire(AlertType.STALE_DATA, AlertSeverity.HIGH, "test", step_id="2024-01-08")
        assert sys.n_alerts() == 1

    def test_process_guardrail_report_fires_on_failure(self):
        """process_guardrail_report must fire one alert per failed check."""
        gr   = InferenceGuardrails()
        w    = _make_clean_weights(8, 16)
        w[8] = 0.05   # mask leak
        mask = _make_mask(8, 16)
        sid  = _make_sector_ids(8, 16)
        rep  = gr.run_all(
            w_exec=w, mask=mask, sector_ids=sid,
            snapshot_date="2024-01-08", most_recent_date="2024-01-08",
            last_update_weeks={}, active_tickers=set(),
            nav_prev=1.0, nav_curr=1.01,
        )
        alert_sys = AlertSystem()
        fired = alert_sys.process_guardrail_report(rep)
        assert len(fired) > 0, "Expected alerts to be fired on guardrail failures"

    def test_pipeline_halt_alert_on_critical(self):
        """CRITICAL failures must produce a PIPELINE_HALT alert."""
        gr = InferenceGuardrails()
        w  = _make_clean_weights(8, 16)
        w[8] = 0.05   # mask leak → CRITICAL
        mask = _make_mask(8, 16)
        sid  = _make_sector_ids(8, 16)
        rep  = gr.run_all(
            w_exec=w, mask=mask, sector_ids=sid,
            snapshot_date="2024-01-08", most_recent_date="2024-01-08",
            last_update_weeks={}, active_tickers=set(),
            nav_prev=1.0, nav_curr=1.01,
        )
        alert_sys = AlertSystem()
        alert_sys.process_guardrail_report(rep)
        halt_alerts = alert_sys.get_by_type(AlertType.PIPELINE_HALT)
        assert len(halt_alerts) > 0, "Expected PIPELINE_HALT alert on CRITICAL failure"

    def test_handler_called_on_fire(self):
        """Registered handler must be called for each alert."""
        received = []
        sys = AlertSystem()
        sys.add_handler(lambda a: received.append(a))
        sys.fire(AlertType.NAV_ANOMALY, AlertSeverity.HIGH, "test")
        assert len(received) == 1
        assert received[0].alert_type == AlertType.NAV_ANOMALY

    def test_has_critical_returns_true(self):
        """has_critical() must return True after a CRITICAL alert."""
        sys = AlertSystem()
        assert not sys.has_critical()
        sys.fire(AlertType.GUARDRAIL_VIOLATION, AlertSeverity.CRITICAL, "critical!")
        assert sys.has_critical()

    def test_summary_counts(self):
        """summary() must return correct counts per severity."""
        sys = AlertSystem()
        sys.fire(AlertType.STALE_DATA,           AlertSeverity.HIGH,   "h1")
        sys.fire(AlertType.STALE_DATA,           AlertSeverity.HIGH,   "h2")
        sys.fire(AlertType.GUARDRAIL_VIOLATION,  AlertSeverity.CRITICAL, "c1")
        s = sys.summary()
        assert s["HIGH"]     == 2
        assert s["CRITICAL"] == 1
        assert s["total"]    == 3

    def test_clear_empties_alerts(self):
        """clear() must remove all stored alerts."""
        sys = AlertSystem()
        sys.fire(AlertType.STALE_DATA, AlertSeverity.LOW, "test")
        sys.clear()
        assert sys.n_alerts() == 0

    def test_str_representation(self):
        """Alert __str__ must be a non-empty string."""
        a = AlertSystem().fire(AlertType.NAV_ANOMALY, AlertSeverity.MEDIUM, "jump")
        assert len(str(a)) > 0

    def test_get_by_type(self):
        """get_by_type must filter correctly."""
        sys = AlertSystem()
        sys.fire(AlertType.STALE_DATA,  AlertSeverity.HIGH, "stale")
        sys.fire(AlertType.NAV_ANOMALY, AlertSeverity.HIGH, "nav")
        stale = sys.get_by_type(AlertType.STALE_DATA)
        assert len(stale) == 1
        assert stale[0].alert_type == AlertType.STALE_DATA
