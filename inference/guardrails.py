"""
inference/guardrails.py
========================
Inference-time guardrails for Project Apex (Bible §12.2).

Enforced at every inference step:
  1. Feasibility check   — all constraints satisfied post-projection
  2. Mask integrity      — mask_leak_mass == 0 (no weight on inactive assets)
  3. Universe validity   — NDX membership snapshot is most-recent as-of
  4. Stale data guard    — no active asset exceeds the stale data window
  5. NAV plausibility    — NAV change between steps within plausible range
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Guardrail result
# ---------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    """Result of a single guardrail check."""
    name:      str
    passed:    bool
    severity:  str              # "CRITICAL" | "HIGH" | "MEDIUM"
    message:   str
    details:   Optional[Dict]  = None


@dataclass
class GuardrailReport:
    """Aggregated report from all guardrail checks at one inference step."""
    checks:      List[GuardrailResult]
    step_id:     Optional[str] = None   # e.g. "2024-01-08"

    @property
    def any_critical(self) -> bool:
        return any(not r.passed and r.severity == "CRITICAL" for r in self.checks)

    @property
    def any_failed(self) -> bool:
        return any(not r.passed for r in self.checks)

    @property
    def failed_names(self) -> List[str]:
        return [r.name for r in self.checks if not r.passed]

    @property
    def should_halt(self) -> bool:
        """True if the pipeline should halt (any CRITICAL failure)."""
        return self.any_critical

    def __str__(self) -> str:
        lines = [f"GuardrailReport (step={self.step_id}): "
                 f"{len(self.failed_names)}/{len(self.checks)} failed"]
        for r in self.checks:
            icon = "✓" if r.passed else "✗"
            lines.append(f"  {icon} [{r.severity}] {r.name}: {r.message}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# InferenceGuardrails
# ---------------------------------------------------------------------------

class InferenceGuardrails:
    """
    §12.2 Guardrail checks enforced at every inference step.

    Parameters
    ----------
    per_name_cap      : max weight per asset (default 0.15 per §4.5)
    sector_cap        : max weight per GICS sector (default 0.35 per §4.5)
    mask_leak_tol     : tolerance for mask_leak_mass == 0 (default 1e-6)
    nav_max_step_frac : NAV change > this fraction triggers plausibility flag
    stale_weeks       : asset is stale if no price update for this many weeks
    """

    def __init__(
        self,
        per_name_cap:       float = 0.15,
        sector_cap:         float = 0.35,
        mask_leak_tol:      float = 1e-6,
        nav_max_step_frac:  float = 0.25,
        stale_weeks:        int   = 2,
    ) -> None:
        self.per_name_cap      = float(per_name_cap)
        self.sector_cap        = float(sector_cap)
        self.mask_leak_tol     = float(mask_leak_tol)
        self.nav_max_step_frac = float(nav_max_step_frac)
        self.stale_weeks       = int(stale_weeks)

    # ======================================================================
    # Individual checks
    # ======================================================================

    def check_feasibility(
        self,
        w_exec:     np.ndarray,   # [K] executed weights
        mask:       np.ndarray,   # [K] float; 1=active, 0=inactive
        sector_ids: np.ndarray,   # [K] int; GICS sector code, -1=inactive
    ) -> GuardrailResult:
        """
        Feasibility check (§12.2): after ConstraintProjector, assert all
        constraints are satisfied. Halt if any violation detected.

        Checks:
          C1 — Long-only: w_i >= 0 for all active i
          C2 — Simplex:  sum_active(w_i) ≈ 1.0
          C3 — Per-name: w_i <= per_name_cap for all i
          C4 — Sector:   sum_sector(w_i) <= sector_cap for each sector
        """
        violations: List[str] = []
        active = mask > 0.5

        # C1: Long-only
        neg = w_exec[active][w_exec[active] < -1e-6]
        if len(neg) > 0:
            violations.append(f"C1 long-only violated: min={float(neg.min()):.6f}")

        # C2: Simplex
        w_sum = float(np.sum(w_exec[active]))
        if not (0.99 <= w_sum <= 1.01):
            violations.append(f"C2 simplex: sum={w_sum:.6f} (expected ≈1.0)")

        # C3: Per-name cap
        over_cap = np.where(active & (w_exec > self.per_name_cap + 1e-6))[0]
        if len(over_cap) > 0:
            violations.append(
                f"C3 per-name cap: {len(over_cap)} assets over {self.per_name_cap}"
            )

        # C4: Sector cap
        unique_sectors = set(sector_ids[active].tolist()) - {-1}
        for sec in unique_sectors:
            in_sec  = active & (sector_ids == sec)
            sec_sum = float(np.sum(w_exec[in_sec]))
            if sec_sum > self.sector_cap + 1e-6:
                violations.append(
                    f"C4 sector_cap: sector={sec} sum={sec_sum:.4f} > {self.sector_cap}"
                )

        passed = len(violations) == 0
        return GuardrailResult(
            name     = "feasibility",
            passed   = passed,
            severity = "CRITICAL",
            message  = "All constraints satisfied" if passed else "; ".join(violations),
            details  = {"violations": violations} if not passed else None,
        )

    def check_mask_integrity(
        self,
        w_exec: np.ndarray,   # [K] executed weights
        mask:   np.ndarray,   # [K] float; 1=active, 0=inactive
    ) -> GuardrailResult:
        """
        Mask integrity (§12.2): mask_leak_mass == 0.
        No weight must be assigned to non-member or inactive assets.
        """
        inactive = mask < 0.5
        leak_mass = float(np.sum(np.abs(w_exec[inactive])))
        passed    = leak_mass <= self.mask_leak_tol

        return GuardrailResult(
            name     = "mask_integrity",
            passed   = passed,
            severity = "CRITICAL",
            message  = (
                f"mask_leak_mass={leak_mass:.2e} (tol={self.mask_leak_tol:.0e})"
                if not passed
                else f"mask_leak_mass=0 ✓"
            ),
            details  = {"mask_leak_mass": leak_mass} if not passed else None,
        )

    def check_universe_validity(
        self,
        snapshot_date:      str,      # ISO date of membership snapshot used
        most_recent_date:   str,      # ISO date of most-recent available snapshot
    ) -> GuardrailResult:
        """
        Universe validity (§12.2): verify the NDX membership snapshot is the
        most-recent as-of snapshot available at decision time.
        """
        passed  = snapshot_date >= most_recent_date
        message = (
            f"snapshot_date={snapshot_date} >= most_recent={most_recent_date} ✓"
            if passed
            else (
                f"STALE SNAPSHOT: used={snapshot_date}, "
                f"most_recent={most_recent_date}"
            )
        )
        return GuardrailResult(
            name     = "universe_validity",
            passed   = passed,
            severity = "HIGH",
            message  = message,
            details  = {
                "snapshot_date":    snapshot_date,
                "most_recent_date": most_recent_date,
            } if not passed else None,
        )

    def check_stale_data(
        self,
        last_update_weeks: Dict[str, int],   # ticker → weeks since last price
        active_tickers:    Set[str],
    ) -> GuardrailResult:
        """
        Stale data guard (§12.2): if any active asset has not received a price
        update within the expected window (stale_weeks), flag as missing data.
        """
        stale = {
            t: w for t, w in last_update_weeks.items()
            if t in active_tickers and w > self.stale_weeks
        }
        passed = len(stale) == 0
        return GuardrailResult(
            name     = "stale_data",
            passed   = passed,
            severity = "HIGH",
            message  = (
                f"Stale assets: {list(stale.keys())} "
                f"(threshold={self.stale_weeks} weeks)"
                if not passed
                else f"All {len(active_tickers)} active assets have fresh data ✓"
            ),
            details  = {"stale_assets": stale} if not passed else None,
        )

    def check_nav_plausibility(
        self,
        nav_prev:  float,
        nav_curr:  float,
        step_desc: str = "",
    ) -> GuardrailResult:
        """
        NAV plausibility (§12.2): assert NAV change is within a plausible
        range. Large unexplained NAV jumps trigger an alert.
        """
        if nav_prev <= 0.0:
            return GuardrailResult(
                name     = "nav_plausibility",
                passed   = False,
                severity = "HIGH",
                message  = f"nav_prev={nav_prev:.6f} is non-positive",
            )

        frac_change = (nav_curr - nav_prev) / nav_prev
        passed      = abs(frac_change) <= self.nav_max_step_frac

        return GuardrailResult(
            name     = "nav_plausibility",
            passed   = passed,
            severity = "HIGH",
            message  = (
                f"NAV jump detected: {frac_change:+.2%} "
                f"({nav_prev:.4f} → {nav_curr:.4f}) {step_desc}"
                if not passed
                else f"NAV change {frac_change:+.2%} within bounds ✓"
            ),
            details  = {
                "nav_prev":    nav_prev,
                "nav_curr":    nav_curr,
                "frac_change": frac_change,
            } if not passed else None,
        )

    # ======================================================================
    # Run all guardrails
    # ======================================================================

    def run_all(
        self,
        w_exec:              np.ndarray,
        mask:                np.ndarray,
        sector_ids:          np.ndarray,
        snapshot_date:       str,
        most_recent_date:    str,
        last_update_weeks:   Dict[str, int],
        active_tickers:      Set[str],
        nav_prev:            float,
        nav_curr:            float,
        step_id:             Optional[str] = None,
    ) -> GuardrailReport:
        """
        Run all 5 guardrails and return a GuardrailReport.

        If report.should_halt is True → halt pipeline and alert.
        """
        checks = [
            self.check_feasibility(w_exec, mask, sector_ids),
            self.check_mask_integrity(w_exec, mask),
            self.check_universe_validity(snapshot_date, most_recent_date),
            self.check_stale_data(last_update_weeks, active_tickers),
            self.check_nav_plausibility(nav_prev, nav_curr, step_id or ""),
        ]
        return GuardrailReport(checks=checks, step_id=step_id)
