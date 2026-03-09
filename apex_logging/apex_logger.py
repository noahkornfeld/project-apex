"""
logging/apex_logger.py
======================
Structured logging and observability for Project Apex (Bible §10).

Logging cadences (§10.7 / Table 52):
  - Every 250 gradient updates : Categories 1-4, 6, 7  (§10.1)
  - Every environment step     : Category 5 + all §10.2 metrics
  - Per fold completion        : §10.3 summary + §10.5 plots
  - Cross-fold                 : §10.4 summary + CIs

Regression alarms (§10.6 / Table 51):
  mask_leak, q_divergence, entropy_collapse, alpha_pinned_max,
  nan_in_obs, constraint_violation, warmup_contamination
"""

from __future__ import annotations

import datetime
import json
import math
import pathlib
from typing import Dict, List, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# Alarm severity table  (§10.6 / Table 51)
# ---------------------------------------------------------------------------

ALARM_SEVERITIES: Dict[str, str] = {
    "mask_leak":            "CRITICAL",
    "q_divergence":         "HIGH",
    "entropy_collapse":     "HIGH",
    "alpha_pinned_max":     "MEDIUM",
    "nan_in_obs":           "CRITICAL",
    "constraint_violation": "CRITICAL",
    "warmup_contamination": "HIGH",
}

# Minimum keys that constitute each category being "present" (completeness check)
CATEGORY_REQUIRED_KEYS: Dict[int, List[str]] = {
    1: ["q1_mean", "q2_mean", "critic_loss"],
    2: ["polyak_tau"],
    3: ["entropy_mean", "alpha"],
    4: ["projection_l1_dist"],
    5: ["turnover"],
    6: ["actor_grad_norm"],
    7: ["K_active_mean"],
}


# ---------------------------------------------------------------------------
# ApexLogger
# ---------------------------------------------------------------------------

class ApexLogger:
    """
    Structured logger for Project Apex (Bible §10).

    Parameters
    ----------
    output_dir          : directory where log files and plots are written
    log_every_n_updates : gradient-update cadence for per-update log records
                          (default 250, per §10.7)
    alpha_max           : alpha_max value from config (used in alarm)
    q_divergence_thresh : |Q| threshold for q_divergence alarm (default 100)
    """

    def __init__(
        self,
        output_dir:          Union[str, pathlib.Path],
        log_every_n_updates: int   = 250,
        alpha_max:           float = 1.0,
        q_divergence_thresh: float = 100.0,
    ) -> None:
        self.output_dir          = pathlib.Path(output_dir)
        self.log_every_n_updates = int(log_every_n_updates)
        self._alpha_max          = float(alpha_max)
        self._q_div_thresh       = float(q_divergence_thresh)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # In-memory record stores (for test inspection)
        self.update_logs:    List[Dict] = []
        self.step_logs:      List[Dict] = []
        self.fold_logs:      List[Dict] = []
        self.cross_fold_log: Optional[Dict] = None
        self.alarm_log:      List[Dict] = []

        # Rolling buffer: raw update metrics between log cadence points
        self._update_buffer: List[Dict] = []

        # Persistent alarm counters
        self._entropy_collapse_steps = 0
        self._alpha_pinned_steps     = 0

        # JSONL log file
        self._logfile = self.output_dir / "apex_training.jsonl"

    # ======================================================================
    # Public API
    # ======================================================================

    def log_update(self, update_step: int, sac_metrics: Dict) -> Dict:
        """
        Ingest per-update metrics from SACTrainer.update() (§10.1).

        Buffers raw metrics; emits a structured 7-category record every
        log_every_n_updates steps.

        Returns the alarm dict evaluated on this update.
        """
        self._update_buffer.append({"step": update_step, **sac_metrics})

        # Evaluate alarms on every update
        alarms = self.check_alarms(sac_metrics)
        triggered = {k: v for k, v in alarms.items() if v["triggered"]}
        if triggered:
            alarm_record = {
                "event":    "alarm",
                "step":     update_step,
                "alarms":   triggered,
            }
            self._emit("alarm", alarm_record)
            self.alarm_log.append(alarm_record)

        # Emit 7-category summary at cadence
        if update_step % self.log_every_n_updates == 0:
            record = self._build_update_log_record(update_step)
            self._emit("update_metrics", record)
            self.update_logs.append(record)
            self._update_buffer.clear()

        return alarms

    def log_step(self, step_info: Dict) -> None:
        """
        Log per-environment-step metrics (§10.2).

        Called every rebalance week (every env step).
        """
        record = self._build_step_log_record(step_info)
        self._emit("step_metrics", record)
        self.step_logs.append(record)

    def log_fold_summary(self, fold_id: int, fold_data: Dict) -> Dict:
        """
        Emit per-fold summary (§10.3 / Table 49).

        Parameters
        ----------
        fold_id   : integer fold index (1-based)
        fold_data : dict of fold-level aggregated metrics
        """
        _keys = [
            "train_start", "train_end", "test_start", "test_end",
            "embargo_weeks", "n_train_steps_raw", "n_train_steps_used",
            "n_test_steps", "n_steps_dropped_embargo",
            "n_steps_dropped_insufficient_lookback",
            # Performance
            "excess_cagr", "sortino", "max_drawdown", "sharpe",
            "excess_cagr_is", "sortino_is", "max_drawdown_is", "sharpe_is",
            # Stability
            "entropy_mean_fold", "alpha_mean_fold", "projection_l1_dist_mean",
            "K_active_mean_fold", "forced_liquidation_count",
            # Cost
            "cost_bps_mean", "cost_bps_max",
            # Critic/Actor
            "q_mean_fold_end", "effective_n_positions_mean",
            # Grad clipping rates
            "pct_steps_clipped_actor", "pct_steps_clipped_critic",
            "pct_steps_clipped_encoder",
            # Leakage validation results
            "leakage_normalizer_pass", "leakage_membership_pass",
            "leakage_embargo_pass", "leakage_nstep_boundary_pass",
        ]
        record = {
            "event":   "fold_summary",
            "fold_id": fold_id,
            **{k: fold_data.get(k) for k in _keys},
        }
        self._emit("fold_summary", record)
        self.fold_logs.append(record)
        return record

    def log_cross_fold(self, all_fold_data: List[Dict]) -> Dict:
        """
        Emit cross-fold summary (§10.4).

        Parameters
        ----------
        all_fold_data : list of fold_data dicts (one per fold, same format as
                        log_fold_summary input)
        """
        def _nanmean(vals): return float(np.nanmean(vals)) if vals else float("nan")
        def _nanstd(vals):  return float(np.nanstd(vals))  if vals else float("nan")

        excess_cagrs = [float(f.get("excess_cagr", float("nan"))) for f in all_fold_data]
        sortinos     = [float(f.get("sortino",     float("nan"))) for f in all_fold_data]
        drawdowns    = [float(f.get("max_drawdown", float("nan"))) for f in all_fold_data]
        sharpes      = [float(f.get("sharpe",       float("nan"))) for f in all_fold_data]

        record = {
            "event":                "cross_fold_summary",
            "n_folds":              len(all_fold_data),
            # Concatenated OOS series
            "excess_cagr_per_fold": excess_cagrs,
            "sortino_per_fold":     sortinos,
            "max_drawdown_per_fold": drawdowns,
            "sharpe_per_fold":       sharpes,
            # Cross-fold aggregates
            "excess_cagr_mean":     _nanmean(excess_cagrs),
            "excess_cagr_std":      _nanstd(excess_cagrs),
            "sortino_mean":         _nanmean(sortinos),
            "sortino_std":          _nanstd(sortinos),
            "sharpe_mean":          _nanmean(sharpes),
            "max_drawdown_mean":    _nanmean(drawdowns),
            # Consistency measure (§10.4)
            "cross_fold_excess_cagr_std": _nanstd(excess_cagrs),
            # Per-fold table (one row per fold)
            "per_fold_table": all_fold_data,
        }
        self._emit("cross_fold_summary", record)
        self.cross_fold_log = record
        return record

    def check_alarms(self, metrics: Dict) -> Dict[str, Dict]:
        """
        Evaluate all §10.6 regression alarms (Table 51).

        Returns
        -------
        dict : alarm_name → {"triggered": bool, "severity": str, "value": ...}
        """
        # ------------------------------------------------------------------ #
        # 1. nan_in_obs  (also triggers on NaN Q-values — §10.6)
        nan_obs = int(metrics.get("nan_count_obs_active", 0))
        q1_raw  = metrics.get("q1_mean", 0.0)
        q2_raw  = metrics.get("q2_mean", 0.0)
        nan_in_q = (
            (isinstance(q1_raw, float) and math.isnan(q1_raw)) or
            (isinstance(q2_raw, float) and math.isnan(q2_raw))
        )
        nan_any = nan_obs > 0 or nan_in_q

        # ------------------------------------------------------------------ #
        # 2. q_divergence
        q1_abs = 0.0 if nan_in_q else abs(float(q1_raw))
        q2_abs = 0.0 if nan_in_q else abs(float(q2_raw))
        q_div  = q1_abs > self._q_div_thresh or q2_abs > self._q_div_thresh or nan_in_q

        # ------------------------------------------------------------------ #
        # 3. entropy_collapse  (persistent counter)
        ent = metrics.get("entropy_mean", float("nan"))
        if isinstance(ent, float) and not math.isnan(ent):
            self._entropy_collapse_steps = (
                self._entropy_collapse_steps + 1 if ent < 0.01 else 0
            )

        # ------------------------------------------------------------------ #
        # 4. alpha_pinned_max  (persistent counter)
        alpha_cur = metrics.get("alpha", float("nan"))
        if isinstance(alpha_cur, float) and not math.isnan(alpha_cur):
            if abs(alpha_cur - self._alpha_max) < 1e-4:
                self._alpha_pinned_steps += 1
            else:
                self._alpha_pinned_steps = 0

        # ------------------------------------------------------------------ #
        # 5. mask_leak
        mask_leak_val = float(metrics.get("mask_leak_mass", 0.0))

        # ------------------------------------------------------------------ #
        # 6. constraint_violation
        cv = bool(metrics.get("constraint_violation", False))

        # ------------------------------------------------------------------ #
        # 7. warmup_contamination
        wc = bool(metrics.get("warmup_contamination", False))

        return {
            "nan_in_obs": {
                "triggered": nan_any,
                "severity":  "CRITICAL",
                "value":     nan_obs,
            },
            "q_divergence": {
                "triggered": q_div,
                "severity":  "HIGH",
                "value":     max(q1_abs, q2_abs),
            },
            "entropy_collapse": {
                "triggered": self._entropy_collapse_steps > 100,
                "severity":  "HIGH",
                "value":     self._entropy_collapse_steps,
            },
            "alpha_pinned_max": {
                "triggered": self._alpha_pinned_steps > 200,
                "severity":  "MEDIUM",
                "value":     self._alpha_pinned_steps,
            },
            "mask_leak": {
                "triggered": mask_leak_val > 0.0,
                "severity":  "CRITICAL",
                "value":     mask_leak_val,
            },
            "constraint_violation": {
                "triggered": cv,
                "severity":  "CRITICAL",
                "value":     cv,
            },
            "warmup_contamination": {
                "triggered": wc,
                "severity":  "HIGH",
                "value":     wc,
            },
        }

    def render_plots(
        self,
        plot_data:  Dict,
        output_dir: Optional[Union[str, pathlib.Path]] = None,
    ) -> Dict[str, str]:
        """
        Render all §10.5 required plots.

        Parameters
        ----------
        plot_data : dict with the following optional keys:
            nav_series           : List[float]  portfolio NAV per OOS step
            qqq_series           : List[float]  QQQ NAV per OOS step
            excess_return_series : List[float]  weekly excess return
            turnover_series      : List[float]  weekly one-way turnover
            cost_bps_series      : List[float]  weekly cost in bps
            entropy_series       : List[float]  entropy_mean per update/step
            alpha_series         : List[float]  alpha per update/step
            H_target_series      : List[float]  H_target per update/step
            K_active_series      : List[int]    K_active per step
            forced_liq_steps     : List[int]    step indices of forced liq events
            q_checkpoints        : List[List[float]]  Q-value samples at ckpts
            checkpoint_steps     : List[int]    update steps of checkpoints
            fold_boundaries      : List[int]    step indices for fold separators

        Returns
        -------
        dict : plot_name → absolute file path
        """
        return render_plots(plot_data, output_dir or self.output_dir)

    # ======================================================================
    # Private helpers
    # ======================================================================

    def _build_update_log_record(self, update_step: int) -> Dict:
        """Build 7-category structured record from buffered update metrics."""
        buf = self._update_buffer if self._update_buffer else [{}]

        def _vals(key):
            return [
                m[key] for m in buf
                if key in m
                and m[key] is not None
                and not (isinstance(m[key], float) and math.isnan(m[key]))
            ]

        def safe_mean(key):
            v = _vals(key)
            return float(np.mean(v)) if v else float("nan")

        def safe_std(key):
            v = _vals(key)
            return float(np.std(v)) if v else float("nan")

        def safe_pct(key, pct):
            v = _vals(key)
            return float(np.percentile(v, pct)) if v else float("nan")

        def pct_true(key):
            v = [bool(m.get(key, False)) for m in buf]
            return float(sum(v) / len(v)) if v else 0.0

        latest = buf[-1] if buf else {}

        # ---- Category 1: Critic Health (Table 36) ----
        cat1 = {
            "q1_mean":            safe_mean("q1_mean"),
            "q2_mean":            safe_mean("q2_mean"),
            "q_min_mean":         safe_mean("q1_mean"),
            "q1_std":             safe_std("q1_mean"),
            "q2_std":             safe_std("q2_mean"),
            "q_gap_mean":         safe_mean("q_gap_mean"),
            "q_target_mean":      safe_mean("q_target_mean"),
            "td_error_abs_mean":  safe_mean("td_error_abs_mean"),
            "critic_loss":        safe_mean("critic_loss"),
            "critic_loss_p50":    safe_pct("critic_loss", 50),
            "critic_loss_p90":    safe_pct("critic_loss", 90),
            "critic_loss_p99":    safe_pct("critic_loss", 99),
            "quantile_spread_mean": safe_mean("quantile_spread_mean"),
        }

        # ---- Category 2: Target Network / Polyak Sanity (Table 37) ----
        cat2 = {
            "polyak_tau":             latest.get("polyak_tau", float("nan")),
            "q_online_minus_target_mean": safe_mean("q_online_minus_target_mean"),
            "target_param_delta_l2":  safe_mean("target_param_delta_l2"),
        }

        # ---- Category 3: Actor Behavior (Table 38) ----
        cat3 = {
            "entropy_mean":           safe_mean("entropy_mean"),
            "H_target_mean":          safe_mean("H_target_mean"),
            "entropy_gap":            safe_mean("entropy_gap"),
            "alpha":                  latest.get("alpha", float("nan")),
            "log_alpha":              latest.get("log_alpha", float("nan")),
            "alpha_loss":             safe_mean("alpha_loss"),
            "effective_n_positions":  safe_mean("effective_n_positions"),
            "max_weight_exec_mean":   safe_mean("max_weight_exec_mean"),
            "max_weight_exec_p95":    safe_pct("max_weight_exec_mean", 95),
            "mask_leak_mass":         safe_mean("mask_leak_mass"),
        }

        # ---- Category 4: Projection and Constraints (Table 39) ----
        cat4 = {
            "projection_l1_dist":         safe_mean("projection_l1_dist"),
            "pct_cap_hits_name":          safe_mean("pct_cap_hits_name"),
            "pct_cap_hits_sector":        safe_mean("pct_cap_hits_sector"),
            "excess_mass_redistributed":  safe_mean("excess_mass_redistributed"),
            "forced_liquidation_notional": safe_mean("forced_liquidation_notional"),
        }

        # ---- Category 5: Trading / Turnover / Costs (Table 40) ----
        cat5 = {
            "turnover":                    safe_mean("turnover"),
            "turnover_p95":                safe_pct("turnover", 95),
            "cost_bps":                    safe_mean("cost_bps"),
            "cost_fraction_of_gross_pnl":  safe_mean("cost_fraction_of_gross_pnl"),
            "holding_period_return":       safe_mean("holding_period_return"),
            "excess_vs_qqq":               safe_mean("excess_vs_qqq"),
        }

        # ---- Category 6: Gradient Norms (Table 41) ----
        lr_actor   = latest.get("lr_actor",   float("nan"))
        lr_critic  = latest.get("lr_critic",  float("nan"))
        lr_encoder = latest.get("lr_encoder", float("nan"))
        lr_alpha   = latest.get("lr_alpha",   float("nan"))

        cat6 = {
            "actor_grad_norm":             safe_mean("actor_grad_norm"),
            "actor_grad_norm_post":        safe_mean("actor_grad_norm_post"),
            "critic_grad_norm":            safe_mean("critic_grad_norm"),
            "critic_grad_norm_post":       safe_mean("critic_grad_norm_post"),
            "enc_grad_norm_crit":          safe_mean("enc_grad_norm_crit"),
            "enc_grad_norm_crit_post":     safe_mean("enc_grad_norm_crit_post"),
            "pct_steps_clipped_actor":     pct_true("actor_grad_clipped"),
            "pct_steps_clipped_critic":    pct_true("critic_grad_clipped"),
            "pct_steps_clipped_encoder":   pct_true("enc_grad_clipped_crit"),
            "lr_actor":                    lr_actor,
            "lr_critic":                   lr_critic,
            "lr_encoder":                  lr_encoder,
            "lr_alpha":                    lr_alpha,
            "param_norm_actor":            safe_mean("param_norm_actor"),
            "param_norm_critic":           safe_mean("param_norm_critic"),
            "param_norm_encoder":          safe_mean("param_norm_encoder"),
        }

        # ---- Category 7: Data / Masking / K Variability (Table 42) ----
        cat7 = {
            "K_active_mean":                         safe_mean("K_active_mean"),
            "K_active_min":                          safe_mean("K_active_min"),
            "K_active_max":                          safe_mean("K_active_max"),
            "nan_count_obs_active":                  safe_mean("nan_count_obs_active"),
            "date_idx_mean":                         safe_mean("date_idx_mean"),
            "pct_transitions_with_membership_change": safe_mean("pct_transitions_with_membership_change"),
        }

        return {
            "event":       "update_metrics",
            "update_step": update_step,
            "n_buffered":  len(buf),
            "cat1_critic_health":   cat1,
            "cat2_polyak_sanity":   cat2,
            "cat3_actor_behavior":  cat3,
            "cat4_projection":      cat4,
            "cat5_trading":         cat5,
            "cat6_grad_norms":      cat6,
            "cat7_data_masking":    cat7,
        }

    def _build_step_log_record(self, step_info: Dict) -> Dict:
        """Build structured per-step log record (§10.2 Tables 43-48)."""
        return {
            "event": "step_metrics",
            # Universe and Masking Sanity (Table 43)
            "date":               step_info.get("date"),
            "week_idx":           step_info.get("week_idx"),
            "K_active":           step_info.get("K_active"),
            "K_member":           step_info.get("K_member"),
            "K_valid":            step_info.get("K_valid"),
            "mask_leak_mass_pre": step_info.get("mask_leak_mass_pre", 0.0),
            # Trading and Constraints (Table 44)
            "turnover":              step_info.get("turnover"),
            "projection_l1_dist":    step_info.get("projection_l1_dist"),
            "max_weight_exec":       step_info.get("max_weight_exec"),
            "effective_n_positions": step_info.get("effective_n_positions"),
            "pct_cap_hits_name":     step_info.get("pct_cap_hits_name"),
            "pct_cap_hits_sector":   step_info.get("pct_cap_hits_sector"),
            # Forced Liquidation (Table 45)
            "forced_liquidation_flag":     step_info.get("forced_liquidation_flag", 0),
            "forced_liquidation_notional": step_info.get("forced_liquidation_notional", 0.0),
            "forced_liquidation_count":    step_info.get("forced_liquidation_count", 0),
            "invalid_freeze_count":        step_info.get("invalid_freeze_count", 0),
            "invalid_liquidation_count":   step_info.get("invalid_liquidation_count", 0),
            # Costs (Table 46)
            "cost_total_bps":            step_info.get("cost_total_bps"),
            "commission_bps":            step_info.get("commission_bps"),
            "spread_bps":                step_info.get("spread_bps"),
            "impact_bps":                step_info.get("impact_bps"),
            "cost_fraction_of_gross_pnl": step_info.get("cost_fraction_of_gross_pnl"),
            # Performance (Table 47)
            "nav":                step_info.get("nav"),
            "weekly_return_net":  step_info.get("weekly_return_net"),
            "weekly_return_gross": step_info.get("weekly_return_gross"),
            "qqq_weekly_return":  step_info.get("qqq_weekly_return"),
            "excess_return":      step_info.get("excess_return"),
            "reward":             step_info.get("reward"),
            "drawdown":           step_info.get("drawdown"),
            # SAC Behavior (Table 48)
            "logp_pre_mean":   step_info.get("logp_pre_mean"),
            "entropy_mean":    step_info.get("entropy_mean"),
            "entropy_std":     step_info.get("entropy_std"),
            "target_entropy":  step_info.get("target_entropy"),
            "entropy_gap":     step_info.get("entropy_gap"),
            "alpha":           step_info.get("alpha"),
            "log_alpha":       step_info.get("log_alpha"),
        }

    def _emit(self, event_type: str, record: Dict) -> None:
        """Append a JSONL record to the log file."""
        out = {
            "_event_type": event_type,
            "_ts":         datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **record,
        }
        with open(self._logfile, "a") as f:
            f.write(json.dumps(out, default=_json_default) + "\n")


# ---------------------------------------------------------------------------
# Standalone plot renderer  (§10.5)
# ---------------------------------------------------------------------------

def render_plots(
    plot_data:  Dict,
    output_dir: Union[str, pathlib.Path],
) -> Dict[str, str]:
    """
    Render all §10.5 required plots to output_dir.

    Required §10.5 plots (Table 50):
      1. Portfolio NAV vs QQQ NAV          (per fold OOS)
      2. Drawdown Curve                    (per fold OOS)
      3. Weekly Excess Return Bar Chart    (per fold OOS)
      4. Turnover + Cost bps Over Time     (dual-axis)
      5. Entropy + Alpha Over Time         (per fold training)
      6. K_active + Forced Liquidation Markers
      7. Q-value Distribution Evolution   (violin at checkpoints)
      8. Cumulative Excess Return          (cross-fold)

    Parameters
    ----------
    plot_data : dict — see ApexLogger.render_plots docstring for expected keys

    Returns
    -------
    dict : plot_name → absolute file path  (only successfully rendered plots)
    """
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend — safe in tests / CI
    import matplotlib.pyplot as plt

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    def _steps(series):
        return list(range(len(series)))

    # ------------------------------------------------------------------
    # 1. Portfolio NAV vs QQQ NAV
    # ------------------------------------------------------------------
    nav = plot_data.get("nav_series", [])
    qqq = plot_data.get("qqq_series", [])
    if nav:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(_steps(nav), nav, label="Portfolio", linewidth=1.5, color="steelblue")
        if qqq:
            ax.plot(_steps(qqq), qqq, label="QQQ", linewidth=1.5,
                    linestyle="--", color="darkorange")
        ax.set_title("Portfolio NAV vs QQQ NAV")
        ax.set_ylabel("NAV")
        ax.set_xlabel("Step")
        ax.legend()
        p = str(output_dir / "plot_nav_vs_qqq.png")
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(fig)
        paths["nav_vs_qqq"] = p

    # ------------------------------------------------------------------
    # 2. Drawdown Curve
    # ------------------------------------------------------------------
    if nav:
        port_dd = _compute_drawdown(nav)
        qqq_dd  = _compute_drawdown(qqq) if qqq else []
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(_steps(port_dd), port_dd, 0, alpha=0.5,
                        color="steelblue", label="Portfolio DD")
        if qqq_dd:
            ax.fill_between(_steps(qqq_dd), qqq_dd, 0, alpha=0.35,
                            color="darkorange", label="QQQ DD")
        ax.set_title("Drawdown Curve")
        ax.set_ylabel("Drawdown")
        ax.set_xlabel("Step")
        ax.legend()
        p = str(output_dir / "plot_drawdown.png")
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(fig)
        paths["drawdown"] = p

    # ------------------------------------------------------------------
    # 3. Weekly Excess Return Bar Chart
    # ------------------------------------------------------------------
    excess = plot_data.get("excess_return_series", [])
    if excess:
        colors = ["#2ecc71" if r >= 0 else "#e74c3c" for r in excess]
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(_steps(excess), excess, color=colors, width=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Weekly Excess Return")
        ax.set_ylabel("Excess Return")
        ax.set_xlabel("Week")
        p = str(output_dir / "plot_excess_return_bar.png")
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(fig)
        paths["excess_return_bar"] = p

    # ------------------------------------------------------------------
    # 4. Turnover + Cost bps Over Time  (dual-axis)
    # ------------------------------------------------------------------
    turnover = plot_data.get("turnover_series", [])
    cost_bps = plot_data.get("cost_bps_series", [])
    if turnover or cost_bps:
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax2 = ax1.twinx()
        if turnover:
            ax1.plot(_steps(turnover), turnover, color="steelblue",
                     label="Turnover", linewidth=1.5)
            ax1.set_ylabel("Turnover", color="steelblue")
        if cost_bps:
            ax2.plot(_steps(cost_bps), cost_bps, color="darkorange",
                     label="Cost bps", linewidth=1.5, linestyle="--")
            ax2.set_ylabel("Cost bps", color="darkorange")
        ax1.set_title("Turnover + Cost bps Over Time")
        ax1.set_xlabel("Step")
        p = str(output_dir / "plot_turnover_cost.png")
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(fig)
        paths["turnover_cost"] = p

    # ------------------------------------------------------------------
    # 5. Entropy + Alpha Over Time  (joint dual-axis)
    # ------------------------------------------------------------------
    entropy  = plot_data.get("entropy_series", [])
    alpha_s  = plot_data.get("alpha_series", [])
    h_target = plot_data.get("H_target_series", [])
    if entropy or alpha_s:
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax2 = ax1.twinx()
        if entropy:
            ax1.plot(_steps(entropy), entropy, color="steelblue",
                     label="Entropy", linewidth=1.5)
        if h_target:
            ax1.plot(_steps(h_target), h_target, color="steelblue",
                     label="H_target", linewidth=1.0, linestyle=":")
        if alpha_s:
            ax2.plot(_steps(alpha_s), alpha_s, color="darkorange",
                     label="Alpha", linewidth=1.5, linestyle="--")
        ax1.set_ylabel("Entropy", color="steelblue")
        ax2.set_ylabel("Alpha",   color="darkorange")
        ax1.set_title("Entropy + Alpha Over Time")
        ax1.set_xlabel("Update Step")
        p = str(output_dir / "plot_entropy_alpha.png")
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(fig)
        paths["entropy_alpha"] = p

    # ------------------------------------------------------------------
    # 6. K_active Over Time + Forced Liquidation Markers
    # ------------------------------------------------------------------
    K_active_s = plot_data.get("K_active_series", [])
    forced_liq = plot_data.get("forced_liq_steps", [])
    if K_active_s:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(_steps(K_active_s), K_active_s, color="teal",
                linewidth=1.5, label="K_active")
        if forced_liq:
            y_vals = [
                K_active_s[s] if s < len(K_active_s) else K_active_s[-1]
                for s in forced_liq
            ]
            ax.scatter(forced_liq, y_vals, color="red", zorder=5,
                       label="Forced Liq.", marker="x", s=80)
        ax.set_title("K_active Over Time + Forced Liquidation Markers")
        ax.set_ylabel("K_active")
        ax.set_xlabel("Step")
        ax.legend()
        p = str(output_dir / "plot_k_active.png")
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(fig)
        paths["k_active"] = p

    # ------------------------------------------------------------------
    # 7. Q-value Distribution Evolution  (violin plot at checkpoints)
    # ------------------------------------------------------------------
    q_checkpoints = plot_data.get("q_checkpoints", [])
    ckpt_steps    = plot_data.get("checkpoint_steps", [])
    if q_checkpoints and all(len(qc) > 0 for qc in q_checkpoints):
        labels = (
            [str(s) for s in ckpt_steps]
            if ckpt_steps else
            [str(i) for i in range(len(q_checkpoints))]
        )
        fig, ax = plt.subplots(figsize=(max(6, len(q_checkpoints) * 1.5), 4))
        positions = list(range(len(q_checkpoints)))
        ax.violinplot(q_checkpoints, positions=positions, showmedians=True)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title("Q-value Distribution Evolution")
        ax.set_ylabel("Q-value")
        ax.set_xlabel("Update Step")
        p = str(output_dir / "plot_q_distribution.png")
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(fig)
        paths["q_distribution"] = p

    # ------------------------------------------------------------------
    # 8. Cumulative Excess Return  (cross-fold)
    # ------------------------------------------------------------------
    if excess:
        cum        = list(float(np.cumsum(excess[:i+1])[-1]) for i in range(len(excess)))
        fold_bnd   = plot_data.get("fold_boundaries", [])
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(_steps(cum), cum, color="steelblue", linewidth=1.5,
                label="Cumulative Excess Return")
        ax.axhline(0, color="black", linewidth=0.8)
        for fb in fold_bnd:
            ax.axvline(fb, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_title("Cumulative Excess Return (Cross-Fold)")
        ax.set_ylabel("Cumulative Excess Return")
        ax.set_xlabel("Week")
        ax.legend()
        p = str(output_dir / "plot_cum_excess_return.png")
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(fig)
        paths["cum_excess_return"] = p

    return paths


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _compute_drawdown(nav_series: List[float]) -> List[float]:
    """Compute drawdown series (0 to -1) from a NAV series."""
    if not nav_series:
        return []
    peak = nav_series[0]
    dd: List[float] = []
    for v in nav_series:
        if v > peak:
            peak = v
        dd.append((v - peak) / peak if peak != 0 else 0.0)
    return dd
