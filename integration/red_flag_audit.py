"""
integration/red_flag_audit.py
==============================
Red-flag stability diagnostic audit for Project Apex (Bible §8.12 / Table 30).

Inspects metrics history from a training run and raises alarms for:
  - Q-value divergence         : |q1_mean| or |q2_mean| > 100
  - Q-gap persistently high    : q_gap_mean > 0.5 consistently
  - Entropy collapse           : entropy_mean < 0.01 for > 100 consecutive steps
  - Alpha saturation           : alpha pinned to alpha_max for > 200 steps
  - Grad norm always clipped   : actor grad_norm_post always == actor_clip (5.0)
  - High TD error plateau      : td_error_abs_mean plateaued at high value
  - High projection distance   : projection_l1_dist > 0.3 on average
  - Reward distribution anomalies: reward std near-zero or NaN-contaminated
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Alarm result
# ---------------------------------------------------------------------------

@dataclass
class RedFlagResult:
    """Single red-flag check result."""
    flag_name:  str
    triggered:  bool
    severity:   str           # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
    message:    str
    details:    Optional[Dict] = None


@dataclass
class AuditReport:
    """Complete audit report for a training run."""
    flags:           List[RedFlagResult]
    n_updates_audited: int
    summary:         str = ""

    @property
    def any_critical(self) -> bool:
        return any(f.triggered and f.severity == "CRITICAL" for f in self.flags)

    @property
    def any_high(self) -> bool:
        return any(f.triggered and f.severity == "HIGH" for f in self.flags)

    @property
    def any_triggered(self) -> bool:
        return any(f.triggered for f in self.flags)

    @property
    def n_triggered(self) -> int:
        return sum(1 for f in self.flags if f.triggered)

    @property
    def triggered_names(self) -> List[str]:
        return [f.flag_name for f in self.flags if f.triggered]

    def __str__(self) -> str:
        lines = [
            f"AuditReport ({self.n_updates_audited} updates): "
            f"{self.n_triggered}/{len(self.flags)} flags triggered"
        ]
        for f in self.flags:
            status = "🚨" if f.triggered else "✓"
            lines.append(f"  {status} [{f.severity}] {f.flag_name}: {f.message}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# RedFlagAuditor
# ---------------------------------------------------------------------------

class RedFlagAuditor:
    """
    §8.12 Stability Diagnostics and Red Flags (Table 30).

    Parameters
    ----------
    q_divergence_threshold      : |q| > this → HIGH flag (default 100)
    q_gap_threshold             : q_gap_mean > this consistently → HIGH
    q_gap_consecutive_frac      : fraction of steps where gap must exceed threshold
    entropy_collapse_threshold  : entropy < this → collapse (default 0.01)
    entropy_collapse_steps      : consecutive steps required (default 100)
    alpha_pinned_steps          : consecutive steps at alpha_max (default 200)
    projection_l1_threshold     : mean > this → flag (default 0.3)
    grad_always_clipped_frac    : fraction of steps clipped to flag (default 0.95)
    td_plateau_window           : look-back window for TD error plateau
    td_plateau_improvement_tol  : min fractional improvement to NOT flag
    """

    def __init__(
        self,
        q_divergence_threshold:    float = 100.0,
        q_gap_threshold:           float = 0.5,
        q_gap_consecutive_frac:    float = 0.5,
        entropy_collapse_threshold: float = 0.01,
        entropy_collapse_steps:    int   = 100,
        alpha_pinned_steps:        int   = 200,
        projection_l1_threshold:   float = 0.3,
        grad_always_clipped_frac:  float = 0.95,
        td_plateau_window:         int   = 50,
        td_plateau_improvement_tol: float = 0.05,
    ) -> None:
        self.q_div_thr        = float(q_divergence_threshold)
        self.q_gap_thr        = float(q_gap_threshold)
        self.q_gap_frac       = float(q_gap_consecutive_frac)
        self.ent_thr          = float(entropy_collapse_threshold)
        self.ent_steps        = int(entropy_collapse_steps)
        self.alpha_pin_steps  = int(alpha_pinned_steps)
        self.proj_thr         = float(projection_l1_threshold)
        self.grad_clip_frac   = float(grad_always_clipped_frac)
        self.td_window        = int(td_plateau_window)
        self.td_tol           = float(td_plateau_improvement_tol)

    # ======================================================================
    # Individual checks
    # ======================================================================

    def check_q_divergence(self, metrics_history: List[Dict]) -> RedFlagResult:
        """
        Q-value divergence (Table 30 / Table 51): |q1_mean| or |q2_mean| > 100.
        Also checks if q_divergence_flag was raised by SACTrainer.
        """
        any_diverged = False
        max_q_abs    = 0.0
        n_violations = 0

        for m in metrics_history:
            # Check explicit flag from SACTrainer
            if m.get("q_divergence_flag", False):
                any_diverged = True
                n_violations += 1
            # Also check raw values
            q1 = m.get("q1_mean", 0.0)
            q2 = m.get("q2_mean", 0.0)
            for q in (q1, q2):
                if math.isfinite(q):
                    max_q_abs = max(max_q_abs, abs(q))
                    if abs(q) > self.q_div_thr:
                        any_diverged = True
                        n_violations += 1
            # NaN check
            if math.isnan(q1) or math.isnan(q2):
                any_diverged = True
                n_violations += 1

        return RedFlagResult(
            flag_name = "q_divergence",
            triggered = any_diverged,
            severity  = "HIGH",
            message   = (
                f"Q divergence DETECTED: {n_violations} violations, max |q|={max_q_abs:.2f}"
                if any_diverged
                else f"Q values stable (max |q|={max_q_abs:.2f})"
            ),
            details   = {"n_violations": n_violations, "max_q_abs": max_q_abs},
        )

    def check_q_gap(self, metrics_history: List[Dict]) -> RedFlagResult:
        """q_gap_mean = mean(Q1 - Q2) > 0.5 consistently (Table 30)."""
        vals = [
            m.get("q_gap_mean", float("nan"))
            for m in metrics_history
            if math.isfinite(m.get("q_gap_mean", float("nan")))
        ]
        if not vals:
            return RedFlagResult("q_gap_persistent", False, "HIGH",
                                  "q_gap: no data", {"n_valid": 0})

        above_thr = sum(1 for v in vals if v > self.q_gap_thr)
        frac      = above_thr / len(vals)
        triggered = frac >= self.q_gap_frac
        mean_gap  = float(sum(vals) / len(vals))

        return RedFlagResult(
            flag_name = "q_gap_persistent",
            triggered = triggered,
            severity  = "HIGH",
            message   = (
                f"Q gap persistently high: {frac:.1%} of steps > {self.q_gap_thr} "
                f"(mean={mean_gap:.3f})"
                if triggered
                else f"Q gap OK (mean={mean_gap:.3f}, {frac:.1%} above threshold)"
            ),
            details   = {"frac_above": frac, "mean_q_gap": mean_gap, "n_valid": len(vals)},
        )

    def check_entropy_collapse(self, metrics_history: List[Dict]) -> RedFlagResult:
        """
        Entropy collapse (Table 51): entropy_mean < 0.01 for > 100 consecutive steps.
        Also checks entropy_collapse_flag from SACTrainer.
        """
        # Check explicit flag first
        if any(m.get("entropy_collapse_flag", False) for m in metrics_history):
            return RedFlagResult(
                flag_name = "entropy_collapse",
                triggered = True,
                severity  = "HIGH",
                message   = "Entropy collapse flag raised by SACTrainer",
            )

        # Count max consecutive collapse steps
        max_consec = 0
        consec     = 0
        for m in metrics_history:
            ent = m.get("entropy_mean", float("nan"))
            if math.isfinite(ent) and ent < self.ent_thr:
                consec += 1
                max_consec = max(max_consec, consec)
            else:
                consec = 0

        triggered = max_consec > self.ent_steps
        return RedFlagResult(
            flag_name = "entropy_collapse",
            triggered = triggered,
            severity  = "HIGH",
            message   = (
                f"Entropy collapse: {max_consec} consecutive steps with entropy < {self.ent_thr}"
                if triggered
                else f"Entropy OK (max consecutive collapse steps: {max_consec})"
            ),
            details   = {"max_consecutive_collapse": max_consec},
        )

    def check_alpha_saturation(
        self,
        metrics_history: List[Dict],
        alpha_max:       float = 1.0,
    ) -> RedFlagResult:
        """
        Alpha saturation (Table 51): alpha pinned to alpha_max for > 200 consecutive steps.
        Also checks alpha_pinned_max_flag from SACTrainer.
        """
        if any(m.get("alpha_pinned_max_flag", False) for m in metrics_history):
            return RedFlagResult(
                flag_name = "alpha_saturation",
                triggered = True,
                severity  = "MEDIUM",
                message   = "Alpha pinned-max flag raised by SACTrainer",
            )

        max_consec = 0
        consec     = 0
        for m in metrics_history:
            a = m.get("alpha", float("nan"))
            if math.isfinite(a) and abs(a - alpha_max) < 1e-4:
                consec += 1
                max_consec = max(max_consec, consec)
            else:
                consec = 0

        triggered = max_consec > self.alpha_pin_steps
        return RedFlagResult(
            flag_name = "alpha_saturation",
            triggered = triggered,
            severity  = "MEDIUM",
            message   = (
                f"Alpha saturation: {max_consec} consecutive steps at alpha_max={alpha_max}"
                if triggered
                else f"Alpha OK (max consecutive at alpha_max: {max_consec})"
            ),
            details   = {"max_consecutive_pinned": max_consec, "alpha_max": alpha_max},
        )

    def check_grad_always_clipped(self, metrics_history: List[Dict]) -> RedFlagResult:
        """
        Actor gradients always clipped (Table 30):
        grad_norm_actor_post always == actor_clip (5.0).
        """
        vals = [
            m.get("actor_grad_norm_post", float("nan"))
            for m in metrics_history
            if math.isfinite(m.get("actor_grad_norm_post", float("nan")))
        ]
        if not vals:
            return RedFlagResult("grad_always_clipped", False, "MEDIUM",
                                  "grad_norm_actor: no data", {"n_valid": 0})

        clipped = sum(1 for v in vals if abs(v - 5.0) < 0.01)
        frac    = clipped / len(vals)
        triggered = frac >= self.grad_clip_frac

        return RedFlagResult(
            flag_name = "grad_always_clipped",
            triggered = triggered,
            severity  = "MEDIUM",
            message   = (
                f"Actor grad always clipped: {frac:.1%} of steps at clip bound 5.0"
                if triggered
                else f"Actor grad clipping OK ({frac:.1%} at clip bound)"
            ),
            details   = {"frac_clipped": frac, "n_valid": len(vals)},
        )

    def check_td_error_plateau(self, metrics_history: List[Dict]) -> RedFlagResult:
        """
        TD error plateau (Table 30): td_error_abs_mean plateaued at very high value.
        """
        vals = [
            m.get("td_error_abs_mean", float("nan"))
            for m in metrics_history
            if math.isfinite(m.get("td_error_abs_mean", float("nan")))
        ]
        if len(vals) < self.td_window * 2:
            return RedFlagResult("td_plateau", False, "MEDIUM",
                                  "TD error: insufficient data for plateau check",
                                  {"n_valid": len(vals)})

        first_half  = float(sum(vals[:self.td_window]) / self.td_window)
        second_half = float(sum(vals[-self.td_window:]) / self.td_window)

        if first_half < 1e-8:
            return RedFlagResult("td_plateau", False, "MEDIUM",
                                  f"TD error near zero (mean={first_half:.4f})")

        improvement = (first_half - second_half) / first_half
        triggered   = improvement < self.td_tol and second_half > 0.1

        return RedFlagResult(
            flag_name = "td_plateau",
            triggered = triggered,
            severity  = "MEDIUM",
            message   = (
                f"TD error plateau detected: first={first_half:.4f}, "
                f"last={second_half:.4f}, improvement={improvement:.1%}"
                if triggered
                else f"TD error improving: {improvement:.1%} reduction"
            ),
            details   = {
                "first_window_mean":  first_half,
                "last_window_mean":   second_half,
                "fractional_improvement": improvement,
            },
        )

    def check_projection_distance(self, metrics_history: List[Dict]) -> RedFlagResult:
        """
        Projection L1 distance (Table 30): projection_l1_dist > 0.3 on average.
        """
        vals = [
            m.get("projection_l1_dist", float("nan"))
            for m in metrics_history
            if math.isfinite(m.get("projection_l1_dist", float("nan")))
        ]
        if not vals:
            return RedFlagResult("projection_l1_high", False, "LOW",
                                  "projection_l1_dist: no data", {"n_valid": 0})

        mean_dist = float(sum(vals) / len(vals))
        triggered = mean_dist > self.proj_thr

        return RedFlagResult(
            flag_name = "projection_l1_high",
            triggered = triggered,
            severity  = "LOW",
            message   = (
                f"Projection L1 distance too high: mean={mean_dist:.3f} > {self.proj_thr}"
                if triggered
                else f"Projection L1 OK: mean={mean_dist:.3f}"
            ),
            details   = {"mean_projection_l1": mean_dist, "threshold": self.proj_thr},
        )

    def check_reward_anomalies(self, metrics_history: List[Dict]) -> RedFlagResult:
        """
        Reward distribution anomalies: reward_std near-zero or NaN-contaminated.
        """
        rewards = [
            m.get("reward_mean", float("nan"))
            for m in metrics_history
        ]
        n_nan   = sum(1 for r in rewards if math.isnan(r))
        finite  = [r for r in rewards if math.isfinite(r)]

        nan_contaminated = n_nan > len(rewards) * 0.1
        std_degenerate   = False
        r_std            = float("nan")

        if len(finite) >= 2:
            r_std          = float((sum((x - sum(finite)/len(finite))**2 for x in finite) / (len(finite)-1)) ** 0.5)
            std_degenerate = r_std < 1e-8

        triggered = nan_contaminated or std_degenerate

        return RedFlagResult(
            flag_name = "reward_anomaly",
            triggered = triggered,
            severity  = "HIGH" if nan_contaminated else "MEDIUM",
            message   = (
                f"Reward anomaly: nan_frac={n_nan/max(1,len(rewards)):.1%}, "
                f"std={r_std:.4f}"
                if triggered
                else f"Reward distribution OK (std={r_std:.4f})"
            ),
            details   = {
                "n_nan":           n_nan,
                "n_total":         len(rewards),
                "reward_std":      r_std,
                "std_degenerate":  std_degenerate,
            },
        )

    # ======================================================================
    # Full audit
    # ======================================================================

    def audit(
        self,
        metrics_history: List[Dict],
        alpha_max:       float = 1.0,
    ) -> AuditReport:
        """
        Run all §8.12 stability checks and return an AuditReport.

        Parameters
        ----------
        metrics_history : list of metrics dicts from SACTrainer.update()
        alpha_max       : upper bound on alpha (for saturation check)
        """
        flags = [
            self.check_q_divergence(metrics_history),
            self.check_q_gap(metrics_history),
            self.check_entropy_collapse(metrics_history),
            self.check_alpha_saturation(metrics_history, alpha_max),
            self.check_grad_always_clipped(metrics_history),
            self.check_td_error_plateau(metrics_history),
            self.check_projection_distance(metrics_history),
            self.check_reward_anomalies(metrics_history),
        ]

        n_triggered = sum(1 for f in flags if f.triggered)
        summary = (
            f"Audit complete: {n_triggered}/{len(flags)} red flags triggered "
            f"on {len(metrics_history)} update steps."
        )
        if n_triggered == 0:
            summary += " No red flags detected — training appears healthy."
        else:
            names = [f.flag_name for f in flags if f.triggered]
            summary += f" Triggered: {names}"

        return AuditReport(
            flags             = flags,
            n_updates_audited = len(metrics_history),
            summary           = summary,
        )
