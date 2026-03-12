"""
inference/alert_system.py
==========================
Alert system for Project Apex inference (Bible §12.2 / Phase 13).

Triggers alerts on:
  - Guardrail violations (mask leak, constraint breach, stale data)
  - NAV anomalies (plausibility bounds exceeded)
  - Stale data events
  - Missing data handler liquidation events
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Alert severity and type
# ---------------------------------------------------------------------------

class AlertSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"


class AlertType(str, Enum):
    GUARDRAIL_VIOLATION  = "GUARDRAIL_VIOLATION"
    NAV_ANOMALY          = "NAV_ANOMALY"
    STALE_DATA           = "STALE_DATA"
    MISSING_DATA         = "MISSING_DATA"
    FORCED_LIQUIDATION   = "FORCED_LIQUIDATION"
    UNIVERSE_STALE       = "UNIVERSE_STALE"
    PIPELINE_HALT        = "PIPELINE_HALT"


@dataclass
class Alert:
    """A single alert event."""
    alert_type:  AlertType
    severity:    AlertSeverity
    message:     str
    timestamp:   datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    step_id:     Optional[str]    = None   # e.g. "2024-01-08"
    details:     Optional[Dict]   = None

    def __str__(self) -> str:
        ts = self.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        return (
            f"[{ts}] [{self.severity.value}] {self.alert_type.value} "
            f"(step={self.step_id}): {self.message}"
        )


# ---------------------------------------------------------------------------
# AlertSystem
# ---------------------------------------------------------------------------

class AlertSystem:
    """
    Collects and dispatches alerts for the paper-trading pipeline.

    Usage
    -----
    alert_sys = AlertSystem()
    alert_sys.add_handler(lambda a: print(a))

    # Raise an alert
    alert_sys.fire(AlertType.GUARDRAIL_VIOLATION, AlertSeverity.CRITICAL,
                   "mask leak detected", step_id="2024-01-08")

    # Inspect
    critical = alert_sys.get_by_severity(AlertSeverity.CRITICAL)
    """

    def __init__(self) -> None:
        self._alerts:   List[Alert]                    = []
        self._handlers: List[Callable[[Alert], None]]  = []

    # ======================================================================
    # Handler registration
    # ======================================================================

    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        """Register a callback that receives every Alert as it is fired."""
        self._handlers.append(handler)

    # ======================================================================
    # Alert firing
    # ======================================================================

    def fire(
        self,
        alert_type:  AlertType,
        severity:    AlertSeverity,
        message:     str,
        step_id:     Optional[str]  = None,
        details:     Optional[Dict] = None,
    ) -> Alert:
        """Create, store and dispatch an alert."""
        alert = Alert(
            alert_type = alert_type,
            severity   = severity,
            message    = message,
            step_id    = step_id,
            details    = details,
        )
        self._alerts.append(alert)
        for handler in self._handlers:
            handler(alert)
        return alert

    # ======================================================================
    # Guardrail-aware helpers
    # ======================================================================

    def process_guardrail_report(self, report) -> List[Alert]:
        """
        Convert a GuardrailReport into alerts.

        Parameters
        ----------
        report : GuardrailReport from inference.guardrails
        """
        fired: List[Alert] = []
        for check in report.checks:
            if not check.passed:
                sev = (
                    AlertSeverity.CRITICAL
                    if check.severity == "CRITICAL"
                    else AlertSeverity.HIGH
                    if check.severity == "HIGH"
                    else AlertSeverity.MEDIUM
                )
                a = self.fire(
                    alert_type = AlertType.GUARDRAIL_VIOLATION,
                    severity   = sev,
                    message    = f"{check.name}: {check.message}",
                    step_id    = report.step_id,
                    details    = check.details,
                )
                fired.append(a)

        if report.should_halt:
            a = self.fire(
                alert_type = AlertType.PIPELINE_HALT,
                severity   = AlertSeverity.CRITICAL,
                message    = (
                    f"Pipeline halted due to CRITICAL guardrail failures: "
                    f"{report.failed_names}"
                ),
                step_id    = report.step_id,
            )
            fired.append(a)

        return fired

    def fire_stale_data(
        self,
        tickers:  List[str],
        step_id:  Optional[str] = None,
    ) -> Alert:
        return self.fire(
            alert_type = AlertType.STALE_DATA,
            severity   = AlertSeverity.HIGH,
            message    = f"Stale data for {len(tickers)} tickers: {tickers}",
            step_id    = step_id,
            details    = {"stale_tickers": tickers},
        )

    def fire_forced_liquidation(
        self,
        tickers:  List[str],
        step_id:  Optional[str] = None,
    ) -> Alert:
        return self.fire(
            alert_type = AlertType.FORCED_LIQUIDATION,
            severity   = AlertSeverity.HIGH,
            message    = f"Forced liquidation for {len(tickers)} tickers: {tickers}",
            step_id    = step_id,
            details    = {"liquidation_tickers": tickers},
        )

    def fire_nav_anomaly(
        self,
        nav_prev:    float,
        nav_curr:    float,
        frac_change: float,
        step_id:     Optional[str] = None,
    ) -> Alert:
        return self.fire(
            alert_type = AlertType.NAV_ANOMALY,
            severity   = AlertSeverity.HIGH,
            message    = (
                f"NAV anomaly: {nav_prev:.4f} → {nav_curr:.4f} "
                f"({frac_change:+.2%})"
            ),
            step_id    = step_id,
            details    = {
                "nav_prev":    nav_prev,
                "nav_curr":    nav_curr,
                "frac_change": frac_change,
            },
        )

    # ======================================================================
    # Inspection
    # ======================================================================

    def get_all(self) -> List[Alert]:
        return list(self._alerts)

    def get_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        return [a for a in self._alerts if a.severity == severity]

    def get_by_type(self, alert_type: AlertType) -> List[Alert]:
        return [a for a in self._alerts if a.alert_type == alert_type]

    def has_critical(self) -> bool:
        return any(a.severity == AlertSeverity.CRITICAL for a in self._alerts)

    def n_alerts(self) -> int:
        return len(self._alerts)

    def clear(self) -> None:
        self._alerts.clear()

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {s.value: 0 for s in AlertSeverity}
        for a in self._alerts:
            counts[a.severity.value] += 1
        counts["total"] = len(self._alerts)
        return counts
