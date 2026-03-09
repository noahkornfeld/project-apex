"""
evaluation/baselines.py
=======================
Baseline NAV series and metrics for Project Apex (Bible §9.4 / Table 35).

3 baselines:
  1. QQQ buy-and-hold          — primary benchmark
  2. Equal-weight (all NDX)    — equal-weight all K_active constituents weekly
  3. Equal-weight (model picks) — equal-weight assets where w_exec_i > ε (=0.01)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


# §9.4: threshold for "model picks" in Equal Weight baseline
EQUAL_WEIGHT_THRESHOLD: float = 0.01


# ---------------------------------------------------------------------------
# NAV series builders
# ---------------------------------------------------------------------------

def build_qqq_nav(qqq_returns: np.ndarray) -> np.ndarray:
    """
    Build QQQ buy-and-hold NAV series starting at 1.0.

    Parameters
    ----------
    qqq_returns : [T] QQQ weekly gross returns (fractions, e.g. 0.01 = +1%)

    Returns
    -------
    nav : [T+1] NAV series starting at 1.0; nav[t+1] = nav[t] * (1 + r_qqq_t)
    """
    qqq_returns = np.asarray(qqq_returns, dtype=float)
    nav = np.empty(len(qqq_returns) + 1)
    nav[0] = 1.0
    for t, r in enumerate(qqq_returns):
        nav[t + 1] = nav[t] * (1.0 + r)
    return nav


def build_equal_weight_nav(
    asset_returns: np.ndarray,    # [T, K] per-asset weekly gross returns
    mask:          np.ndarray,    # [T, K] float; 1 = active, 0 = inactive
) -> np.ndarray:
    """
    Equal-weight all active NDX constituents each week.

    At each step t the portfolio holds equal weight in all K_active_t assets.
    Portfolio return = mean of active asset gross returns (after masking).

    Returns [T+1] NAV series starting at 1.0.
    """
    asset_returns = np.asarray(asset_returns, dtype=float)
    mask          = np.asarray(mask,          dtype=float)
    T = len(asset_returns)

    nav = np.empty(T + 1)
    nav[0] = 1.0

    for t in range(T):
        active = mask[t] > 0
        k_active = active.sum()
        if k_active == 0:
            r_port = 0.0
        else:
            r_port = float(asset_returns[t][active].mean())
        nav[t + 1] = nav[t] * (1.0 + r_port)

    return nav


def build_equal_weight_model_picks_nav(
    asset_returns: np.ndarray,    # [T, K] per-asset weekly gross returns
    w_exec:        np.ndarray,    # [T, K] model executed weights
    mask:          np.ndarray,    # [T, K] float; 1 = active
    threshold:     float = EQUAL_WEIGHT_THRESHOLD,
) -> np.ndarray:
    """
    Equal-weight only assets where the model allocates w_exec_i > threshold (ε).

    Tests whether selection skill exceeds weighting skill (§9.4).
    If no asset clears the threshold, falls back to equal-weight all active.

    Returns [T+1] NAV series starting at 1.0.
    """
    asset_returns = np.asarray(asset_returns, dtype=float)
    w_exec        = np.asarray(w_exec,        dtype=float)
    mask          = np.asarray(mask,          dtype=float)
    T = len(asset_returns)

    nav = np.empty(T + 1)
    nav[0] = 1.0

    for t in range(T):
        picks  = (w_exec[t] > threshold) & (mask[t] > 0)
        if picks.sum() == 0:
            # Fallback: equal-weight all active
            active = mask[t] > 0
            picks  = active

        if picks.sum() == 0:
            r_port = 0.0
        else:
            r_port = float(asset_returns[t][picks].mean())

        nav[t + 1] = nav[t] * (1.0 + r_port)

    return nav


# ---------------------------------------------------------------------------
# BaselineCalculator
# ---------------------------------------------------------------------------

class BaselineCalculator:
    """
    Compute all 3 §9.4 baseline NAV series and their performance metrics.

    Parameters
    ----------
    qqq_returns    : [T] QQQ weekly gross returns
    asset_returns  : [T, K] per-asset weekly gross returns
    mask           : [T, K] float; 1 = active constituent, 0 = inactive
    w_exec         : [T, K] model executed weights (for model-picks baseline)
    threshold      : weight threshold ε for model-picks (default 0.01)
    """

    def __init__(
        self,
        qqq_returns:   np.ndarray,
        asset_returns: np.ndarray,
        mask:          np.ndarray,
        w_exec:        Optional[np.ndarray] = None,
        threshold:     float = EQUAL_WEIGHT_THRESHOLD,
    ) -> None:
        self.qqq_returns   = np.asarray(qqq_returns,   dtype=float)
        self.asset_returns = np.asarray(asset_returns, dtype=float)
        self.mask          = np.asarray(mask,          dtype=float)
        self.w_exec        = (
            np.asarray(w_exec, dtype=float) if w_exec is not None else None
        )
        self.threshold     = float(threshold)

        # Built lazily
        self._qqq_nav:    Optional[np.ndarray] = None
        self._ew_nav:     Optional[np.ndarray] = None
        self._ew_mp_nav:  Optional[np.ndarray] = None

    # ---- NAV builders ----

    @property
    def qqq_nav(self) -> np.ndarray:
        if self._qqq_nav is None:
            self._qqq_nav = build_qqq_nav(self.qqq_returns)
        return self._qqq_nav

    @property
    def equal_weight_nav(self) -> np.ndarray:
        if self._ew_nav is None:
            self._ew_nav = build_equal_weight_nav(self.asset_returns, self.mask)
        return self._ew_nav

    @property
    def equal_weight_model_picks_nav(self) -> np.ndarray:
        if self._ew_mp_nav is None:
            if self.w_exec is None:
                raise ValueError(
                    "w_exec must be provided to compute equal-weight model-picks baseline."
                )
            self._ew_mp_nav = build_equal_weight_model_picks_nav(
                self.asset_returns, self.w_exec, self.mask, self.threshold
            )
        return self._ew_mp_nav

    # ---- Summary ----

    def all_nav_series(self) -> Dict[str, np.ndarray]:
        """Return dict of all 3 baseline NAV series."""
        result: Dict[str, np.ndarray] = {
            "qqq":          self.qqq_nav,
            "equal_weight": self.equal_weight_nav,
        }
        if self.w_exec is not None:
            result["equal_weight_model_picks"] = self.equal_weight_model_picks_nav
        return result

    def summary(self) -> Dict[str, Dict]:
        """Return CAGR and final NAV for each baseline."""
        T = len(self.qqq_returns)
        weeks_per_year = 52.0

        def _cagr(nav_arr):
            final = float(nav_arr[-1])
            if final <= 0 or T == 0:
                return float("nan")
            return final ** (weeks_per_year / T) - 1.0

        out: Dict[str, Dict] = {
            "qqq":          {"cagr": _cagr(self.qqq_nav),          "final_nav": float(self.qqq_nav[-1])},
            "equal_weight": {"cagr": _cagr(self.equal_weight_nav), "final_nav": float(self.equal_weight_nav[-1])},
        }
        if self.w_exec is not None:
            mp_nav = self.equal_weight_model_picks_nav
            out["equal_weight_model_picks"] = {
                "cagr":      _cagr(mp_nav),
                "final_nav": float(mp_nav[-1]),
            }
        return out
