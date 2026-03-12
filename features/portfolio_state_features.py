"""
Portfolio-State Features — Bible §3.4.1
=========================================
Portfolio-state features capture the agent's current position and
performance state.  They are broadcast into g_t (global context vector)
alongside macro and benchmark features.

8 features (slots 12-19 of the 20-dim g_t vector):
    0  turnover_last_step          — sum |w_exec_t − w_exec_{t-1}|
    1  realized_port_vol           — std of last 13w returns × √52  (annualised)
    2  current_drawdown            — (NAV_t − peak_NAV) / peak_NAV, in [-1, 0]
    3  gross_exposure              — sum of w_exec  (≈1.0 for long-only)
    4  rolling_excess_ret_qqq      — portfolio 13w cumret minus QQQ 13w cumret
    5  estimated_cost_next_step    — cost model with current weights as trade
    6  effective_n_positions       — 1 / Σ w_exec_i²  (inverse Herfindahl)
    7  market_vol_regime           — std of last 52w daily QQQ rets × √252

Normalisation (applied in compute_portfolio_state):
    Features 0,1,4,5,6,7  → global z-score with fixed-scale priors
    Feature  2             → raw; already in [-1, 0]
    Feature  3             → raw; bounded, ≈1.0
All outputs clipped to [-5, 5].

Fixed-scale priors (derived from typical equity long-only portfolio behaviour):
    turnover          : mean=0.15, std=0.15
    realized_vol      : mean=0.15, std=0.08
    excess_ret_qqq    : mean=0.00, std=0.10
    estimated_cost    : mean=0.003, std=0.003
    eff_n_positions   : mean=20.0,  std=15.0
    market_vol_regime : mean=0.18,  std=0.08

These priors are intentionally conservative.  The model will adapt to the
actual distribution via gradient descent; the normalization just ensures
the features enter the network at a reasonable scale rather than as zeros.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Sequence

PORTFOLIO_STATE_FEATURE_NAMES = [
    "turnover_last_step",
    "realized_port_vol",
    "current_drawdown",
    "gross_exposure",
    "rolling_excess_ret_qqq",
    "estimated_cost_next_step",
    "effective_n_positions",
    "market_vol_regime",
]

N_PORTFOLIO_STATE_FEATURES = len(PORTFOLIO_STATE_FEATURE_NAMES)

_EPS = 1e-8

# Fixed-scale normalisation priors
_NORM_MEAN = np.array([0.15, 0.15, 0.0,  1.0,  0.0,   0.003, 20.0, 0.18], dtype=np.float64)
_NORM_STD  = np.array([0.15, 0.08, 1.0,  1.0,  0.10,  0.003, 15.0, 0.08], dtype=np.float64)
# (std=1 for drawdown and gross_exposure → pass-through, no z-score)

_CLIP = 5.0  # symmetric clip after z-scoring


def compute_portfolio_state(
    w_exec:         np.ndarray,
    w_exec_prev:    np.ndarray,
    nav:            float,
    peak_nav:       float,
    ret_port_hist:  Sequence[float],
    ret_qqq_hist:   Sequence[float],
    qqq_daily_rets: np.ndarray,
    mask:           np.ndarray,
    estimated_cost: float = 0.0,
) -> np.ndarray:
    """
    Compute all 8 portfolio-state features and return a normalised [8] vector.

    Parameters
    ----------
    w_exec         : [K] executed weights at current step
    w_exec_prev    : [K] executed weights from previous step (for turnover)
    nav            : current portfolio NAV
    peak_nav       : running peak NAV since episode start
    ret_port_hist  : list / deque of recent weekly portfolio net returns
    ret_qqq_hist   : list / deque of recent weekly QQQ returns
    qqq_daily_rets : array of recent daily QQQ log-returns (≥260 for full 52w)
    mask           : [K] active-slot mask
    estimated_cost : pre-computed cost fraction from the cost model

    Returns
    -------
    np.ndarray of shape [8], dtype float32, clipped to [-5, 5].
    """
    raw = np.empty(N_PORTFOLIO_STATE_FEATURES, dtype=np.float64)

    # ------------------------------------------------------------------
    # 0  turnover_last_step  =  Σ |w_exec_t − w_exec_{t-1}|
    # ------------------------------------------------------------------
    raw[0] = float(np.abs(w_exec - w_exec_prev).sum())

    # ------------------------------------------------------------------
    # 1  realized_port_vol  =  std(last 13w port returns) × √52
    # ------------------------------------------------------------------
    port_window = list(ret_port_hist)[-13:]
    if len(port_window) >= 2:
        raw[1] = float(np.std(port_window, ddof=1)) * np.sqrt(52)
    else:
        raw[1] = 0.0

    # ------------------------------------------------------------------
    # 2  current_drawdown  =  clip((nav − peak_nav) / peak_nav, −1, 0)
    # ------------------------------------------------------------------
    peak_safe = max(float(peak_nav), _EPS)
    raw[2]    = float(np.clip((nav - peak_safe) / peak_safe, -1.0, 0.0))

    # ------------------------------------------------------------------
    # 3  gross_exposure  =  Σ w_exec_i  (should be ≈1.0 for long-only)
    # ------------------------------------------------------------------
    raw[3] = float(w_exec.sum())

    # ------------------------------------------------------------------
    # 4  rolling_excess_ret_qqq  =  Σ r_port[−13:] − Σ r_qqq[−13:]
    # ------------------------------------------------------------------
    port_13 = list(ret_port_hist)[-13:]
    qqq_13  = list(ret_qqq_hist)[-13:]
    n13     = min(len(port_13), len(qqq_13))
    if n13 >= 1:
        raw[4] = float(np.sum(port_13[-n13:])) - float(np.sum(qqq_13[-n13:]))
    else:
        raw[4] = 0.0

    # ------------------------------------------------------------------
    # 5  estimated_cost_next_step  (pre-computed by caller)
    # ------------------------------------------------------------------
    raw[5] = float(estimated_cost)

    # ------------------------------------------------------------------
    # 6  effective_n_positions  =  1 / Σ w_exec_i²  (inverse Herfindahl)
    # ------------------------------------------------------------------
    w_sq_sum = float(np.dot(w_exec, w_exec))
    raw[6]   = 1.0 / w_sq_sum if w_sq_sum > _EPS else 0.0

    # ------------------------------------------------------------------
    # 7  market_vol_regime  =  std(last 52w daily QQQ rets) × √252
    # ------------------------------------------------------------------
    if len(qqq_daily_rets) >= 2:
        window = qqq_daily_rets[-260:]          # up to 52w of trading days
        raw[7] = float(np.std(window, ddof=1)) * np.sqrt(252)
    else:
        raw[7] = 0.0

    # ------------------------------------------------------------------
    # Normalise with fixed-scale priors; clip to [-_CLIP, +_CLIP]
    # (features 2 and 3 have std=1 in _NORM_STD so they pass through
    #  after subtracting their mean=0/1 respectively)
    # ------------------------------------------------------------------
    z = (raw - _NORM_MEAN) / (_NORM_STD + _EPS)
    z = np.clip(z, -_CLIP, _CLIP)

    return z.astype(np.float32)


# ---------------------------------------------------------------------------
# Backward-compatible stubs (kept for unit tests that pre-date Phase 5)
# ---------------------------------------------------------------------------

def compute_portfolio_state_stub(
    n_dates: Optional[int] = None,
    dates: Optional[pd.DatetimeIndex] = None,
) -> np.ndarray:
    """Return placeholder zeros (legacy stub — used only by feature_panel
    precomputation where live portfolio state is unavailable)."""
    if dates is not None:
        T = len(dates)
    elif n_dates is not None:
        T = n_dates
    else:
        return np.zeros(N_PORTFOLIO_STATE_FEATURES, dtype=np.float32)
    return np.zeros((T, N_PORTFOLIO_STATE_FEATURES), dtype=np.float32)


def compute_portfolio_state_stub_single() -> np.ndarray:
    """Return a single-step zero vector (legacy stub)."""
    return np.zeros(N_PORTFOLIO_STATE_FEATURES, dtype=np.float32)
