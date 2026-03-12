"""
inference/paper_trade_loop.py
==============================
Paper-trading loop for Project Apex (Bible §12.4 / Phase 13).

Weekly cadence:
  Monday close : generate new w_exec signal from model
  Tuesday open : simulate execution (apply training cost model §5.4)

Initialisation: portfolio starts at equal-weight among active assets,
mirroring the warmup-step initialization (§12.4).

Records per-step: date, w_exec, w_prev, cost_bps, nav, trade_log.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from inference.guardrails import InferenceGuardrails, GuardrailReport
from inference.missing_data_handler import MissingDataHandler
from inference.alert_system import AlertSystem, AlertType, AlertSeverity


# ---------------------------------------------------------------------------
# Step record
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """Single paper-trade step record."""
    step_id:      str             # ISO date (Monday)
    w_prev:       np.ndarray      # [K_max] pre-trade weights
    w_exec:       np.ndarray      # [K_max] post-signal weights
    cost_frac:    float           # transaction cost fraction
    cost_bps:     float           # total cost in bps
    nav:          float           # portfolio NAV after cost
    nav_pre_cost: float           # NAV before cost
    active_k:     int             # number of active assets
    guardrail_ok: bool
    halt:         bool = False    # True if pipeline was halted this step
    notes:        List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Simple cost model (used when TradingEnvironment is not available)
# ---------------------------------------------------------------------------

def _simple_cost_model(
    delta_w:  np.ndarray,   # [K] absolute weight changes
    nav:      float,
    vix:      float  = 20.0,
    adv63:    Optional[np.ndarray] = None,
    mask:     Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Simplified 4-term cost model matching §5.4 for paper-trade simulation.
    Returns (cost_fraction, cost_bps_total).
    """
    K      = len(delta_w)
    active = np.ones(K, dtype=bool) if mask is None else (mask > 0.5)
    adv    = np.ones(K) * 1e8 if adv63 is None else adv63

    trade_dollars = np.abs(delta_w) * nav
    M             = max(1.0, vix / 20.0)
    adv_safe      = np.where(active & (adv > 1e-3), adv, 1e-3)
    adv_median    = float(np.median(adv[active])) if active.any() else 1e-3

    term1 = 1.0 * np.ones(K)
    term2 = M * 2.0 * np.power(adv_median / adv_safe, 0.3)
    term3 = M * 10.0 * np.power(np.maximum(trade_dollars / adv_safe, 0.0), 0.5)
    term4 = 1.5 * 0.1 * 0.7979 * np.ones(K)

    cost_bps_per = trade_dollars / 1e4 * (term1 + term2 + term3 + term4)
    cost_bps_per = np.where(active & (trade_dollars > 1e-9), cost_bps_per, 0.0)

    total_bps      = float(cost_bps_per.sum())
    cost_fraction  = total_bps / 1e4   # bps → fraction

    return cost_fraction, total_bps


# ---------------------------------------------------------------------------
# PaperTradeLoop
# ---------------------------------------------------------------------------

class PaperTradeLoop:
    """
    §12.4 Weekly paper-trading loop.

    Parameters
    ----------
    model          : ApexActorCritic (eval mode) — produces w_exec from obs
    K_max          : universe size (must match model)
    guardrails     : InferenceGuardrails instance
    missing_handler: MissingDataHandler instance
    alert_system   : AlertSystem instance
    init_nav       : starting NAV (default 1.0)
    device         : torch device
    """

    def __init__(
        self,
        model,
        K_max:           int,
        guardrails:      InferenceGuardrails = None,
        missing_handler: MissingDataHandler  = None,
        alert_system:    AlertSystem          = None,
        init_nav:        float               = 1.0,
        device:          torch.device        = None,
    ) -> None:
        self._model   = model
        self._K_max   = K_max
        self._gr      = guardrails   or InferenceGuardrails()
        self._md      = missing_handler or MissingDataHandler()
        self._alerts  = alert_system or AlertSystem()
        self._nav     = float(init_nav)
        self._device  = device or torch.device("cpu")

        # Running state
        self._w_cur:   np.ndarray = np.zeros(K_max, dtype=np.float32)
        self._records: List[TradeRecord] = []
        self._halted:  bool = False

    # ======================================================================
    # Properties
    # ======================================================================

    @property
    def nav(self) -> float:
        return self._nav

    @property
    def records(self) -> List[TradeRecord]:
        return list(self._records)

    @property
    def is_halted(self) -> bool:
        return self._halted

    # ======================================================================
    # Initialization (§12.4)
    # ======================================================================

    def initialize_warmup(
        self,
        mask: np.ndarray,   # [K_max] active mask
    ) -> None:
        """
        Initialize portfolio to equal-weight among active assets,
        matching the warmup initialization (§12.4).
        """
        active  = mask > 0.5
        n_act   = int(active.sum())
        self._w_cur = np.zeros(self._K_max, dtype=np.float32)
        if n_act > 0:
            self._w_cur[active] = 1.0 / n_act

    # ======================================================================
    # Single step
    # ======================================================================

    def step(
        self,
        step_id:          str,             # Monday date string
        obs:              Dict,            # observation dict for model
        mask:             np.ndarray,      # [K_max] float
        sector_ids:       np.ndarray,      # [K_max] int
        asset_returns:    np.ndarray,      # [K_max] float (realized this week)
        snapshot_date:    str,
        most_recent_date: str,
        last_update_weeks: Dict[str, int],
        active_tickers:   set,
        vix:              float            = 20.0,
        adv63:            Optional[np.ndarray] = None,
    ) -> TradeRecord:
        """
        Execute one paper-trade step (Monday signal → Tuesday execution).

        1. Run model → w_exec
        2. Apply missing-data policy
        3. Apply guardrails
        4. Compute cost
        5. Update NAV
        6. Record step

        Returns TradeRecord for this step.
        """
        if self._halted:
            rec = TradeRecord(
                step_id      = step_id,
                w_prev       = self._w_cur.copy(),
                w_exec       = self._w_cur.copy(),
                cost_frac    = 0.0,
                cost_bps     = 0.0,
                nav          = self._nav,
                nav_pre_cost = self._nav,
                active_k     = int((mask > 0.5).sum()),
                guardrail_ok = False,
                halt         = True,
                notes        = ["Pipeline halted — not executing new trades"],
            )
            self._records.append(rec)
            return rec

        notes: List[str] = []

        # ── 1. Model inference ───────────────────────────────────────────
        w_pre = self._run_model(obs)

        # ── 2. Missing data policy ───────────────────────────────────────
        tickers = [f"t{k}" for k in range(self._K_max)]
        has_price   = {t: True for t in tickers}
        has_feature = {t: True for t in tickers}
        w_prev_dict = {f"t{k}": float(self._w_cur[k]) for k in range(self._K_max)}

        adj_mask, frozen_weights, liquidate_list = self._md.process_step(
            tickers, has_price, has_feature, w_prev_dict
        )
        effective_mask = mask * adj_mask

        if liquidate_list:
            self._alerts.fire_forced_liquidation(liquidate_list, step_id)
            notes.append(f"Forced liquidation: {liquidate_list}")

        # Apply frozen weights
        if frozen_weights:
            frozen_idx = {k: float(v) for t, v in frozen_weights.items()
                          for k, t2 in enumerate(tickers) if t2 == t}
            w_pre_np = w_pre.copy()
            tickers_list = list(tickers)
            w_exec_arr = self._md.apply_frozen_weights(w_pre_np, tickers_list, frozen_weights)
        else:
            w_exec_arr = w_pre.copy()

        # Apply effective mask
        w_exec_arr = w_exec_arr * effective_mask
        w_sum = float(w_exec_arr.sum())
        if w_sum > 1e-9:
            w_exec_arr = w_exec_arr / w_sum

        # ── 3. Guardrails ────────────────────────────────────────────────
        nav_pre = self._nav * float(1.0 + np.dot(self._w_cur, asset_returns))

        gr_report = self._gr.run_all(
            w_exec            = w_exec_arr,
            mask              = effective_mask,
            sector_ids        = sector_ids,
            snapshot_date     = snapshot_date,
            most_recent_date  = most_recent_date,
            last_update_weeks = last_update_weeks,
            active_tickers    = active_tickers,
            nav_prev          = self._nav,
            nav_curr          = nav_pre,
            step_id           = step_id,
        )

        fired = self._alerts.process_guardrail_report(gr_report)
        guardrail_ok = not gr_report.any_failed

        if gr_report.should_halt:
            self._halted = True
            notes.append("CRITICAL guardrail violation — pipeline halted")

        # ── 4. Transaction cost (§5.4) ───────────────────────────────────
        delta_w    = w_exec_arr - self._w_cur
        cost_frac, cost_bps = _simple_cost_model(delta_w, nav_pre, vix, adv63, effective_mask)

        # ── 5. NAV update ────────────────────────────────────────────────
        nav_after_cost = nav_pre * (1.0 - cost_frac)
        self._nav      = nav_after_cost
        self._w_cur    = w_exec_arr.copy()

        # ── 6. Record ────────────────────────────────────────────────────
        rec = TradeRecord(
            step_id      = step_id,
            w_prev       = w_exec_arr.copy(),
            w_exec       = w_exec_arr.copy(),
            cost_frac    = cost_frac,
            cost_bps     = cost_bps,
            nav          = self._nav,
            nav_pre_cost = nav_pre,
            active_k     = int((effective_mask > 0.5).sum()),
            guardrail_ok = guardrail_ok,
            halt         = self._halted,
            notes        = notes,
        )
        self._records.append(rec)
        return rec

    # ======================================================================
    # Helpers
    # ======================================================================

    def _run_model(self, obs: Dict) -> np.ndarray:
        """
        Run actor_forward from observation dict.
        obs must have: x_panel, g_panel, mask, sector_ids, ticker_ids.
        Returns w_exec [K_max] numpy array.
        """
        x   = torch.tensor(obs["x_panel"],   dtype=torch.float32,  device=self._device)
        g   = torch.tensor(obs["g_panel"],   dtype=torch.float32,  device=self._device)
        mk  = torch.tensor(obs["mask"],      dtype=torch.float32,  device=self._device)
        sid = torch.tensor(obs["sector_ids"],dtype=torch.int64,    device=self._device)
        tid = torch.tensor(obs["ticker_ids"],dtype=torch.int64,    device=self._device)

        with torch.no_grad():
            w_pre, _ = self._model.actor_forward(x, g, mk, sid, tid)

        return w_pre.squeeze(0).cpu().numpy()

    # ======================================================================
    # Reporting
    # ======================================================================

    def nav_series(self) -> List[float]:
        return [r.nav for r in self._records]

    def cost_series(self) -> List[float]:
        return [r.cost_bps for r in self._records]

    def trade_log(self) -> List[Dict]:
        return [
            {
                "step_id":      r.step_id,
                "nav":          r.nav,
                "cost_bps":     r.cost_bps,
                "active_k":     r.active_k,
                "guardrail_ok": r.guardrail_ok,
                "halt":         r.halt,
                "notes":        r.notes,
            }
            for r in self._records
        ]
