"""
Portfolio-State Features — Bible §3.4  [STUB]
===============================================
Portfolio-state features are broadcast into g_t alongside macro and benchmark.

These require the environment loop (Phase 5) to compute properly:
    - Executed portfolio weights w_exec_(t-1)
    - Portfolio NAV series for drawdown and volatility
    - Transaction cost model for Estimated Cost Next Step

Per the Phase 3 roadmap (Stub vs Full):
    "Stub: portfolio-state features (need environment loop; wire placeholder
     zeros until Phase 5)."

§3.4.1 Portfolio-State Features (8 features):
    Turnover Last Step              — global z-score
    Realized Portfolio Volatility   — global z-score (13-week rolling)
    Current Drawdown                — scaled [-1, 0]
    Gross Exposure                  — none (≈1.0 for long-only)
    Rolling Excess Return vs QQQ    — global z-score (13-week)
    Estimated Cost Next Step        — global z-score
    Effective # of Positions        — global z-score
    Market Volatility Regime        — global z-score (52-week QQQ vol)
"""

import numpy as np
import pandas as pd
from typing import Optional

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


def compute_portfolio_state_stub(
    n_dates: Optional[int] = None,
    dates: Optional[pd.DatetimeIndex] = None,
) -> np.ndarray:
    """
    Return placeholder zeros for all 8 portfolio-state features.

    Args:
        n_dates : Number of time steps (if dates not provided).
        dates   : DatetimeIndex (preferred). If provided, n_dates is ignored.

    Returns:
        ndarray of shape [T, 8] filled with zeros.
        At inference time, shape is [8] (a single time step).

    TODO (Phase 5): Replace with real portfolio-state computation using
        executed weights, NAV series, and cost model.
    """
    if dates is not None:
        T = len(dates)
    elif n_dates is not None:
        T = n_dates
    else:
        return np.zeros(N_PORTFOLIO_STATE_FEATURES, dtype=np.float32)

    return np.zeros((T, N_PORTFOLIO_STATE_FEATURES), dtype=np.float32)


def compute_portfolio_state_stub_single() -> np.ndarray:
    """
    Return a single-step portfolio-state stub (shape [8]).
    Used at inference time before Phase 5 is implemented.
    """
    return np.zeros(N_PORTFOLIO_STATE_FEATURES, dtype=np.float32)
