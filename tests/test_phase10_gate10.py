"""
tests/test_phase10_gate10.py
============================
Gate 10: Logging and Observability  (Bible §10 / screenshot gate table)

Gate criteria:
  Completeness : All 7 metric categories from §10.1 emitted in 500-step run
  Alarms       : Injecting NaN into Q-values triggers the NaN alarm
  Plots        : All required plots render without error for synthetic data

Additional unit tests:
  - Per-step log structure (§10.2)
  - Per-fold summary structure (§10.3)
  - Cross-fold summary (§10.4)
  - All 7 individual alarm types trigger on injected bad values
  - Alarm does NOT fire on clean metrics
  - JSONL log file written correctly
  - Plot file names match expected §10.5 list
"""

from __future__ import annotations

import math
import pathlib
import tempfile
from typing import Dict, List

import numpy as np
import pytest

# Import via direct path to avoid any stdlib logging shadowing issues
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from apex_logging.apex_logger import (
    ApexLogger,
    render_plots,
    ALARM_SEVERITIES,
    CATEGORY_REQUIRED_KEYS,
    _compute_drawdown,
)


# ===========================================================================
# Shared helpers
# ===========================================================================

def _clean_update_metrics(step: int = 1) -> Dict:
    """Return a valid, clean metrics dict as SACTrainer.update() would emit."""
    return {
        # Cat 1 – Critic Health
        "q1_mean":            float(np.random.uniform(-5, 5)),
        "q2_mean":            float(np.random.uniform(-5, 5)),
        "q_target_mean":      float(np.random.uniform(-5, 5)),
        "td_error_abs_mean":  float(np.random.uniform(0, 1)),
        "critic_loss":        float(np.random.uniform(0.01, 1.0)),
        "loss_q1":            float(np.random.uniform(0.01, 0.5)),
        "loss_q2":            float(np.random.uniform(0.01, 0.5)),
        # Cat 2 – Target Network
        "polyak_tau":                 0.005,
        "q_online_minus_target_mean": float(np.random.uniform(-0.1, 0.1)),
        "target_param_delta_l2":      float(np.random.uniform(0, 0.01)),
        # Cat 3 – Actor Behavior
        "entropy_mean":      float(np.random.uniform(0.5, 2.0)),
        "H_target_mean":     float(-np.log(8) * 0.7),
        "entropy_gap":       float(np.random.uniform(-0.5, 0.5)),
        "alpha":             float(np.random.uniform(0.01, 0.5)),
        "log_alpha":         float(np.random.uniform(-5, 0)),
        "alpha_loss":        float(np.random.uniform(-0.1, 0.1)),
        "mask_leak_mass":    0.0,
        "effective_n_positions": float(np.random.uniform(3, 8)),
        "max_weight_exec_mean":  float(np.random.uniform(0.1, 0.3)),
        # Cat 4 – Projection
        "projection_l1_dist":         float(np.random.uniform(0, 0.1)),
        "pct_cap_hits_name":          float(np.random.uniform(0, 0.2)),
        "pct_cap_hits_sector":        float(np.random.uniform(0, 0.1)),
        "forced_liquidation_notional": 0.0,
        # Cat 5 – Trading
        "turnover":                   float(np.random.uniform(0.05, 0.3)),
        "cost_bps":                   float(np.random.uniform(1, 10)),
        "cost_fraction_of_gross_pnl": float(np.random.uniform(0.01, 0.1)),
        "holding_period_return":      float(np.random.uniform(-0.02, 0.02)),
        "excess_vs_qqq":              float(np.random.uniform(-0.01, 0.01)),
        # Cat 6 – Gradient Norms
        "actor_grad_norm":        float(np.random.uniform(0.1, 2.0)),
        "actor_grad_norm_post":   float(np.random.uniform(0.1, 1.5)),
        "critic_grad_norm":       float(np.random.uniform(0.1, 2.0)),
        "critic_grad_norm_post":  float(np.random.uniform(0.1, 1.0)),
        "enc_grad_norm_crit":     float(np.random.uniform(0.1, 2.0)),
        "enc_grad_norm_crit_post": float(np.random.uniform(0.1, 1.0)),
        "actor_grad_clipped":     bool(np.random.rand() > 0.8),
        "critic_grad_clipped":    bool(np.random.rand() > 0.8),
        "enc_grad_clipped_crit":  bool(np.random.rand() > 0.8),
        "lr_actor":   3e-4,
        "lr_critic":  3e-4,
        "lr_encoder": 1e-4,
        "lr_alpha":   3e-4,
        # Cat 7 – Data / Masking
        "K_active_mean":  float(np.random.uniform(6, 8)),
        "K_active_min":   6.0,
        "K_active_max":   8.0,
        "nan_count_obs_active": 0,
        "date_idx_mean":  float(step + 100),
        # Alarm inputs
        "constraint_violation": False,
        "warmup_contamination": False,
    }


def _synthetic_plot_data(n_steps: int = 100, n_ckpts: int = 4) -> Dict:
    """Synthetic data for all §10.5 plot types."""
    rng = np.random.default_rng(0)
    nav = list(float(v) for v in np.cumprod(1 + rng.normal(0.001, 0.01, n_steps)))
    qqq = list(float(v) for v in np.cumprod(1 + rng.normal(0.001, 0.012, n_steps)))
    excess = [float(a - b) for a, b in
              zip(rng.normal(0.001, 0.01, n_steps),
                  rng.normal(0.001, 0.012, n_steps))]
    return {
        "nav_series":           nav,
        "qqq_series":           qqq,
        "excess_return_series": excess,
        "turnover_series":      list(float(v) for v in rng.uniform(0.05, 0.3, n_steps)),
        "cost_bps_series":      list(float(v) for v in rng.uniform(1, 10, n_steps)),
        "entropy_series":       list(float(v) for v in rng.uniform(0.5, 2.0, n_steps)),
        "alpha_series":         list(float(v) for v in rng.uniform(0.01, 0.5, n_steps)),
        "H_target_series":      list(float(-np.log(8) * 0.7) for _ in range(n_steps)),
        "K_active_series":      [int(v) for v in rng.integers(6, 9, n_steps)],
        "forced_liq_steps":     [10, 45, 80],
        "q_checkpoints":        [list(float(v) for v in rng.normal(i * 0.5, 1.0, 50))
                                  for i in range(n_ckpts)],
        "checkpoint_steps":     [i * (n_steps // n_ckpts) * 10 for i in range(n_ckpts)],
        "fold_boundaries":      [25, 50, 75],
    }


# ===========================================================================
# Gate 10 — Completeness
# ===========================================================================

class TestGateCompleteness:
    """
    Gate: All 7 metric categories from §10.1 emitted in a 500-step training run.
    Metric / Inspection: Log completeness audit.
    """

    N_STEPS = 500

    def test_all_7_categories_present_in_update_logs(self, tmp_path):
        """After 500 updates with log_every=250, update_logs has 2 records, each
        containing all 7 category dict keys."""
        rng = np.random.default_rng(1)
        logger = ApexLogger(output_dir=tmp_path, log_every_n_updates=250)

        for step in range(1, self.N_STEPS + 1):
            np.random.seed(step)
            logger.log_update(step, _clean_update_metrics(step))

        assert len(logger.update_logs) >= 2, (
            f"Expected ≥2 update log records in {self.N_STEPS} steps "
            f"with cadence 250, got {len(logger.update_logs)}"
        )

        required_cat_keys = [
            "cat1_critic_health",
            "cat2_polyak_sanity",
            "cat3_actor_behavior",
            "cat4_projection",
            "cat5_trading",
            "cat6_grad_norms",
            "cat7_data_masking",
        ]
        for record in logger.update_logs:
            missing = [k for k in required_cat_keys if k not in record]
            assert not missing, (
                f"Update log record at step={record.get('update_step')} "
                f"missing category keys: {missing}"
            )

    def test_each_category_has_required_metric_keys(self, tmp_path):
        """Each category dict must contain the minimum required metric keys."""
        logger = ApexLogger(output_dir=tmp_path, log_every_n_updates=250)
        for step in range(1, self.N_STEPS + 1):
            np.random.seed(step)
            logger.log_update(step, _clean_update_metrics(step))

        for record in logger.update_logs:
            # Cat 1
            cat1 = record["cat1_critic_health"]
            for k in ["q1_mean", "q2_mean", "critic_loss"]:
                assert k in cat1, f"cat1 missing key '{k}'"
            # Cat 2
            cat2 = record["cat2_polyak_sanity"]
            assert "polyak_tau" in cat2, "cat2 missing 'polyak_tau'"
            # Cat 3
            cat3 = record["cat3_actor_behavior"]
            for k in ["entropy_mean", "alpha"]:
                assert k in cat3, f"cat3 missing key '{k}'"
            # Cat 4
            cat4 = record["cat4_projection"]
            assert "projection_l1_dist" in cat4, "cat4 missing 'projection_l1_dist'"
            # Cat 5
            cat5 = record["cat5_trading"]
            assert "turnover" in cat5, "cat5 missing 'turnover'"
            # Cat 6
            cat6 = record["cat6_grad_norms"]
            assert "actor_grad_norm" in cat6, "cat6 missing 'actor_grad_norm'"
            # Cat 7
            cat7 = record["cat7_data_masking"]
            assert "K_active_mean" in cat7, "cat7 missing 'K_active_mean'"

    def test_cadence_exactly_250(self, tmp_path):
        """Log record emitted at steps 250 and 500; NOT at 251 or 499."""
        logger = ApexLogger(output_dir=tmp_path, log_every_n_updates=250)
        for step in range(1, self.N_STEPS + 1):
            logger.log_update(step, _clean_update_metrics(step))

        logged_steps = [r["update_step"] for r in logger.update_logs]
        assert 250 in logged_steps, "No record at step 250"
        assert 500 in logged_steps, "No record at step 500"
        assert len(logged_steps) == 2, (
            f"Expected exactly 2 records for 500 steps / cadence 250, "
            f"got {len(logged_steps)}: {logged_steps}"
        )

    def test_jsonl_file_written(self, tmp_path):
        """JSONL log file must exist and contain parseable lines."""
        logger = ApexLogger(output_dir=tmp_path, log_every_n_updates=250)
        for step in range(1, 251):
            logger.log_update(step, _clean_update_metrics(step))

        logfile = tmp_path / "apex_training.jsonl"
        assert logfile.exists(), "apex_training.jsonl not created"
        lines = logfile.read_text().strip().splitlines()
        assert len(lines) > 0, "Log file is empty"
        for line in lines:
            import json
            record = json.loads(line)   # should not raise
            assert "_event_type" in record
            assert "_ts" in record


# ===========================================================================
# Gate 10 — Alarms
# ===========================================================================

class TestGateAlarms:
    """
    Gate: Injecting NaN into Q-values triggers the NaN alarm.
    Metric / Inspection: Alarm trigger test.
    """

    def test_nan_q_value_triggers_nan_in_obs(self, tmp_path):
        """Injecting NaN into q1_mean must trigger nan_in_obs alarm (CRITICAL)."""
        logger = ApexLogger(output_dir=tmp_path)
        metrics = _clean_update_metrics()
        metrics["q1_mean"] = float("nan")

        alarms = logger.check_alarms(metrics)
        assert alarms["nan_in_obs"]["triggered"], (
            "nan_in_obs alarm must fire when q1_mean is NaN"
        )
        assert alarms["nan_in_obs"]["severity"] == "CRITICAL"

    def test_nan_q_value_triggers_q_divergence(self, tmp_path):
        """NaN Q-value must also trigger q_divergence alarm."""
        logger = ApexLogger(output_dir=tmp_path)
        metrics = _clean_update_metrics()
        metrics["q2_mean"] = float("nan")

        alarms = logger.check_alarms(metrics)
        assert alarms["q_divergence"]["triggered"], (
            "q_divergence alarm must fire when q2_mean is NaN"
        )

    def test_nan_alarm_written_to_log(self, tmp_path):
        """Alarm must be written to the JSONL log file when triggered."""
        import json
        logger = ApexLogger(output_dir=tmp_path, log_every_n_updates=250)
        metrics = _clean_update_metrics()
        metrics["q1_mean"] = float("nan")

        logger.log_update(1, metrics)

        logfile = tmp_path / "apex_training.jsonl"
        alarm_lines = [
            json.loads(line) for line in logfile.read_text().strip().splitlines()
            if json.loads(line).get("_event_type") == "alarm"
        ]
        assert alarm_lines, "No alarm record written to JSONL log"
        triggered_names = list(alarm_lines[0]["alarms"].keys())
        assert "nan_in_obs" in triggered_names or "q_divergence" in triggered_names

    def test_large_q_value_triggers_q_divergence(self, tmp_path):
        """Q-value |q| > 100 must trigger q_divergence alarm (HIGH)."""
        logger = ApexLogger(output_dir=tmp_path)
        metrics = _clean_update_metrics()
        metrics["q1_mean"] = 150.0

        alarms = logger.check_alarms(metrics)
        assert alarms["q_divergence"]["triggered"], (
            f"q_divergence must fire for q1_mean=150, got triggered={alarms['q_divergence']['triggered']}"
        )
        assert alarms["q_divergence"]["severity"] == "HIGH"

    def test_mask_leak_triggers_alarm(self, tmp_path):
        """Non-zero mask_leak_mass triggers mask_leak alarm (CRITICAL)."""
        logger = ApexLogger(output_dir=tmp_path)
        metrics = _clean_update_metrics()
        metrics["mask_leak_mass"] = 0.001

        alarms = logger.check_alarms(metrics)
        assert alarms["mask_leak"]["triggered"]
        assert alarms["mask_leak"]["severity"] == "CRITICAL"

    def test_constraint_violation_triggers_alarm(self, tmp_path):
        """constraint_violation=True triggers alarm (CRITICAL)."""
        logger = ApexLogger(output_dir=tmp_path)
        metrics = _clean_update_metrics()
        metrics["constraint_violation"] = True

        alarms = logger.check_alarms(metrics)
        assert alarms["constraint_violation"]["triggered"]
        assert alarms["constraint_violation"]["severity"] == "CRITICAL"

    def test_warmup_contamination_triggers_alarm(self, tmp_path):
        """warmup_contamination=True triggers alarm (HIGH)."""
        logger = ApexLogger(output_dir=tmp_path)
        metrics = _clean_update_metrics()
        metrics["warmup_contamination"] = True

        alarms = logger.check_alarms(metrics)
        assert alarms["warmup_contamination"]["triggered"]
        assert alarms["warmup_contamination"]["severity"] == "HIGH"

    def test_entropy_collapse_requires_persistence(self, tmp_path):
        """entropy_collapse fires only after > 100 consecutive low-entropy steps."""
        logger = ApexLogger(output_dir=tmp_path)
        metrics = _clean_update_metrics()
        metrics["entropy_mean"] = 0.001   # below threshold

        # After 100 steps it should NOT have fired yet
        for _ in range(100):
            alarms = logger.check_alarms(metrics)
        assert not alarms["entropy_collapse"]["triggered"], (
            "entropy_collapse fired after exactly 100 steps; requires > 100"
        )

        # Step 101 → should fire
        alarms = logger.check_alarms(metrics)
        assert alarms["entropy_collapse"]["triggered"], (
            "entropy_collapse must fire after > 100 consecutive low-entropy steps"
        )

    def test_entropy_collapse_resets_on_recovery(self, tmp_path):
        """Counter resets when entropy recovers above threshold."""
        logger = ApexLogger(output_dir=tmp_path)
        low_metrics  = {**_clean_update_metrics(), "entropy_mean": 0.001}
        high_metrics = {**_clean_update_metrics(), "entropy_mean": 2.0}

        for _ in range(150):
            logger.check_alarms(low_metrics)
        # Now fire
        alarms = logger.check_alarms(low_metrics)
        assert alarms["entropy_collapse"]["triggered"]

        # One high-entropy step resets counter
        logger.check_alarms(high_metrics)
        alarms = logger.check_alarms(high_metrics)
        assert not alarms["entropy_collapse"]["triggered"], (
            "entropy_collapse must reset when entropy recovers"
        )

    def test_clean_metrics_no_alarms(self, tmp_path):
        """Clean metrics must not trigger any alarm."""
        logger = ApexLogger(output_dir=tmp_path)
        metrics = _clean_update_metrics()

        alarms = logger.check_alarms(metrics)
        triggered = {k: v for k, v in alarms.items() if v["triggered"]}
        assert not triggered, (
            f"Clean metrics triggered unexpected alarms: {list(triggered.keys())}"
        )

    def test_all_alarm_keys_present(self, tmp_path):
        """check_alarms must return all 7 alarm keys defined in ALARM_SEVERITIES."""
        logger = ApexLogger(output_dir=tmp_path)
        alarms = logger.check_alarms(_clean_update_metrics())

        for alarm_name in ALARM_SEVERITIES:
            assert alarm_name in alarms, f"Alarm '{alarm_name}' missing from check_alarms output"
            assert "triggered" in alarms[alarm_name]
            assert "severity"  in alarms[alarm_name]
            assert "value"     in alarms[alarm_name]


# ===========================================================================
# Gate 10 — Plots
# ===========================================================================

class TestGatePlots:
    """
    Gate: All required plots render without error for synthetic data.
    Metric / Inspection: Plot smoke test.
    """

    EXPECTED_PLOT_KEYS = {
        "nav_vs_qqq",
        "drawdown",
        "excess_return_bar",
        "turnover_cost",
        "entropy_alpha",
        "k_active",
        "q_distribution",
        "cum_excess_return",
    }

    EXPECTED_FILENAMES = {
        "plot_nav_vs_qqq.png",
        "plot_drawdown.png",
        "plot_excess_return_bar.png",
        "plot_turnover_cost.png",
        "plot_entropy_alpha.png",
        "plot_k_active.png",
        "plot_q_distribution.png",
        "plot_cum_excess_return.png",
    }

    def test_all_8_plots_render_without_error(self, tmp_path):
        """All 8 §10.5 plots must render without raising an exception."""
        plot_data = _synthetic_plot_data()
        paths = render_plots(plot_data, tmp_path)
        assert len(paths) == 8, (
            f"Expected 8 plot files, got {len(paths)}: {list(paths.keys())}"
        )

    def test_all_plot_keys_returned(self, tmp_path):
        """render_plots must return all 8 expected plot keys."""
        plot_data = _synthetic_plot_data()
        paths = render_plots(plot_data, tmp_path)

        missing = self.EXPECTED_PLOT_KEYS - set(paths.keys())
        assert not missing, f"Missing plot keys: {missing}"

    def test_all_plot_files_exist(self, tmp_path):
        """All returned file paths must exist on disk."""
        plot_data = _synthetic_plot_data()
        paths = render_plots(plot_data, tmp_path)

        for name, fpath in paths.items():
            assert pathlib.Path(fpath).exists(), (
                f"Plot '{name}' file not found at: {fpath}"
            )
            assert pathlib.Path(fpath).stat().st_size > 0, (
                f"Plot '{name}' file is empty: {fpath}"
            )

    def test_all_expected_filenames_created(self, tmp_path):
        """All expected PNG filenames must be present in the output directory."""
        plot_data = _synthetic_plot_data()
        render_plots(plot_data, tmp_path)

        created = {f.name for f in tmp_path.iterdir() if f.suffix == ".png"}
        missing = self.EXPECTED_FILENAMES - created
        assert not missing, f"Missing PNG files: {missing}"

    def test_plots_via_logger_render_plots_method(self, tmp_path):
        """ApexLogger.render_plots() method must also produce all 8 files."""
        logger    = ApexLogger(output_dir=tmp_path)
        plot_data = _synthetic_plot_data()
        paths     = logger.render_plots(plot_data)

        assert len(paths) == 8
        for _, fpath in paths.items():
            assert pathlib.Path(fpath).exists()

    def test_plot_with_minimal_data(self, tmp_path):
        """Plots must render even with very short series (10 steps)."""
        plot_data = _synthetic_plot_data(n_steps=10, n_ckpts=2)
        paths = render_plots(plot_data, tmp_path)
        assert len(paths) == 8

    def test_nav_plot_with_no_qqq_series(self, tmp_path):
        """NAV plot must render when qqq_series is absent (optional)."""
        plot_data = _synthetic_plot_data()
        plot_data.pop("qqq_series", None)
        paths = render_plots(plot_data, tmp_path)
        assert "nav_vs_qqq" in paths
        assert pathlib.Path(paths["nav_vs_qqq"]).exists()


# ===========================================================================
# Per-Step Log Structure  (§10.2)
# ===========================================================================

class TestPerStepLogging:
    """Verify §10.2 per-step log records contain all required sub-categories."""

    def _make_step_info(self) -> Dict:
        return {
            # Universe / Masking
            "date": "2023-01-02", "week_idx": 157, "K_active": 8,
            "K_member": 100, "K_valid": 98, "mask_leak_mass_pre": 0.0,
            # Trading
            "turnover": 0.12, "projection_l1_dist": 0.05,
            "max_weight_exec": 0.15, "effective_n_positions": 7.2,
            "pct_cap_hits_name": 0.0, "pct_cap_hits_sector": 0.0,
            # Forced Liquidation
            "forced_liquidation_flag": 0, "forced_liquidation_notional": 0.0,
            "forced_liquidation_count": 0, "invalid_freeze_count": 0,
            "invalid_liquidation_count": 0,
            # Costs
            "cost_total_bps": 3.5, "commission_bps": 1.0,
            "spread_bps": 1.5, "impact_bps": 1.0,
            "cost_fraction_of_gross_pnl": 0.08,
            # Performance
            "nav": 1.02, "weekly_return_net": 0.005,
            "weekly_return_gross": 0.006, "qqq_weekly_return": 0.004,
            "excess_return": 0.001, "reward": 0.002, "drawdown": 0.0,
            # SAC Behavior
            "logp_pre_mean": -1.5, "entropy_mean": 1.8, "entropy_std": 0.3,
            "target_entropy": -1.4, "entropy_gap": 0.4,
            "alpha": 0.2, "log_alpha": -1.6,
        }

    def test_step_log_contains_all_sections(self, tmp_path):
        """log_step must record a dict with all §10.2 sub-category keys."""
        logger = ApexLogger(output_dir=tmp_path)
        logger.log_step(self._make_step_info())

        assert len(logger.step_logs) == 1
        rec = logger.step_logs[0]

        required = [
            "date", "week_idx", "K_active",             # universe
            "turnover", "max_weight_exec",               # trading
            "forced_liquidation_flag",                   # forced liq
            "cost_total_bps",                            # costs
            "nav", "weekly_return_net", "excess_return", # performance
            "entropy_mean", "alpha",                     # SAC behavior
        ]
        for k in required:
            assert k in rec, f"Step log missing key '{k}'"

    def test_step_log_event_field(self, tmp_path):
        """Step log record must have event='step_metrics'."""
        logger = ApexLogger(output_dir=tmp_path)
        logger.log_step(self._make_step_info())
        assert logger.step_logs[0]["event"] == "step_metrics"


# ===========================================================================
# Per-Fold Summary  (§10.3)
# ===========================================================================

class TestPerFoldSummary:
    """Verify §10.3 per-fold summary records."""

    def _make_fold_data(self, fold_id: int = 1) -> Dict:
        return {
            "train_start": "2010-01-04", "train_end": "2015-12-28",
            "test_start": "2016-01-04",  "test_end": "2016-12-26",
            "embargo_weeks": 4,
            "n_train_steps_raw": 260, "n_train_steps_used": 252,
            "n_test_steps": 52,
            "n_steps_dropped_embargo": 4,
            "n_steps_dropped_insufficient_lookback": 60,
            "excess_cagr": 0.03, "sortino": 1.2, "max_drawdown": -0.08, "sharpe": 0.8,
            "excess_cagr_is": 0.04, "sortino_is": 1.5, "max_drawdown_is": -0.06, "sharpe_is": 1.0,
            "entropy_mean_fold": 1.5, "alpha_mean_fold": 0.15,
            "projection_l1_dist_mean": 0.04, "K_active_mean_fold": 7.5,
            "forced_liquidation_count": 0, "cost_bps_mean": 3.2, "cost_bps_max": 8.1,
            "q_mean_fold_end": 0.2, "effective_n_positions_mean": 6.8,
            "pct_steps_clipped_actor": 0.12, "pct_steps_clipped_critic": 0.08,
            "pct_steps_clipped_encoder": 0.10,
            "leakage_normalizer_pass": True, "leakage_membership_pass": True,
            "leakage_embargo_pass": True, "leakage_nstep_boundary_pass": True,
        }

    def test_fold_summary_stored(self, tmp_path):
        """log_fold_summary must append to fold_logs."""
        logger = ApexLogger(output_dir=tmp_path)
        logger.log_fold_summary(1, self._make_fold_data(1))

        assert len(logger.fold_logs) == 1
        rec = logger.fold_logs[0]
        assert rec["fold_id"] == 1
        assert rec["event"] == "fold_summary"

    def test_fold_summary_contains_required_fields(self, tmp_path):
        """Fold summary must include all primary/secondary metrics."""
        logger = ApexLogger(output_dir=tmp_path)
        logger.log_fold_summary(1, self._make_fold_data(1))
        rec = logger.fold_logs[0]

        for k in ["excess_cagr", "sortino", "max_drawdown", "sharpe",
                  "embargo_weeks", "n_train_steps_used", "n_test_steps",
                  "entropy_mean_fold", "alpha_mean_fold", "forced_liquidation_count"]:
            assert k in rec, f"Fold summary missing key '{k}'"

    def test_multiple_folds_accumulated(self, tmp_path):
        """Multiple fold summaries must all be stored."""
        logger = ApexLogger(output_dir=tmp_path)
        for fold_id in range(1, 9):
            logger.log_fold_summary(fold_id, self._make_fold_data(fold_id))

        assert len(logger.fold_logs) == 8


# ===========================================================================
# Cross-Fold Summary  (§10.4)
# ===========================================================================

class TestCrossFoldSummary:
    """Verify §10.4 cross-fold summary."""

    def test_cross_fold_aggregates(self, tmp_path):
        """log_cross_fold must compute mean, std across folds."""
        logger = ApexLogger(output_dir=tmp_path)
        folds = [{"excess_cagr": 0.01 * i, "sortino": 0.5 + 0.1 * i,
                  "max_drawdown": -0.05, "sharpe": 0.8} for i in range(1, 9)]

        rec = logger.log_cross_fold(folds)
        assert rec["n_folds"] == 8
        assert math.isfinite(rec["excess_cagr_mean"])
        assert math.isfinite(rec["excess_cagr_std"])
        assert math.isfinite(rec["sortino_mean"])
        assert "cross_fold_excess_cagr_std" in rec

    def test_cross_fold_per_fold_table(self, tmp_path):
        """Cross-fold record must include the per_fold_table."""
        logger = ApexLogger(output_dir=tmp_path)
        folds  = [{"excess_cagr": 0.02, "sortino": 1.0} for _ in range(4)]

        rec = logger.log_cross_fold(folds)
        assert "per_fold_table" in rec
        assert len(rec["per_fold_table"]) == 4

    def test_cross_fold_stored_in_logger(self, tmp_path):
        """cross_fold_log attribute must be set after log_cross_fold."""
        logger = ApexLogger(output_dir=tmp_path)
        logger.log_cross_fold([{"excess_cagr": 0.01} for _ in range(3)])
        assert logger.cross_fold_log is not None
        assert logger.cross_fold_log["n_folds"] == 3


# ===========================================================================
# Drawdown utility
# ===========================================================================

class TestDrawdownUtility:
    def test_flat_nav_zero_drawdown(self):
        nav = [1.0] * 10
        dd  = _compute_drawdown(nav)
        assert all(abs(v) < 1e-9 for v in dd), "Flat NAV should have zero drawdown"

    def test_declining_nav(self):
        nav = [1.0, 0.9, 0.8, 0.7]
        dd  = _compute_drawdown(nav)
        assert dd[0] == pytest.approx(0.0,  abs=1e-9)
        assert dd[1] == pytest.approx(-0.1, abs=1e-6)
        assert dd[-1] == pytest.approx(-0.3, abs=1e-6)

    def test_recovery_resets_drawdown(self):
        nav = [1.0, 0.8, 1.2, 1.0]
        dd  = _compute_drawdown(nav)
        assert dd[2] == pytest.approx(0.0,  abs=1e-9)   # new peak
        assert dd[3] < 0                                  # drawdown from 1.2
