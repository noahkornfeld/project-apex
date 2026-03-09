"""
environment/reward_fn.py
Reward Function — Phase 6 (Bible §6).

5-term per-step reward:
    r_t = (e_t / σ_mkt,t)              Term 1: market-adjusted excess return
        - λ_slow  × σ_t                Term 2: slow volatility penalty
        - λ_tail  × max(0, -e_t/σ)²   Term 3: quadratic tail / downside penalty
        - λ_cost  × costs_t            Term 4: explicit transaction cost penalty
        - λ_cv    × violations_t       Term 5: soft constraint violation penalty

    clipped to [−5, +5] after summation (§6.5).

Key design choices aligned with Bible §6:
  • e_t uses net portfolio return (after cost deduction), so costs enter Term 1
    implicitly AND Term 4 explicitly — intentional double-cost per §6.3 note.
  • Rolling σ_mkt,t and σ_t use causal trailing windows (no future leakage).
  • Cold-start (§6.4): priors used when fewer than W weeks of in-episode
    history are available; linearly blended toward in-episode std as the
    window fills.
  • Full float32 throughout; no Q-value clipping (§6.5).
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional

import math
import numpy as np

EPS = 1e-8

# Default cold-start priors (typical weekly return volatility)
# QQQ weekly vol ≈ 20% ann / sqrt(52) ≈ 2.77% → round to 2.0% conservatively
# Portfolio weekly vol typically slightly higher
_DEFAULT_SIGMA_MKT_PRIOR  = 0.020   # 2.0% per week
_DEFAULT_SIGMA_PORT_PRIOR = 0.025   # 2.5% per week


class RewardFunction:
    """
    Stateful 5-term reward function (Bible §6).

    Must call reset() at the start of each episode to clear rolling buffers.

    Parameters
    ----------
    lambda_slow : float
        Weight on the slow vol penalty term (§6.3, default 0.75).
    lambda_tail : float
        Weight on the quadratic downside penalty (§6.3, default 0.4).
    lambda_cost : float
        Weight on explicit cost penalty (§6.3, default 1.0).
    lambda_cv : float
        Weight on soft constraint violation penalty (§6.3, default 1.0).
    sigma_mkt_window : int
        Causal rolling window for σ_mkt,t in *weeks* (§6.2, default 13).
    sigma_port_window : int
        Causal rolling window for σ_t in *weeks* (§6.2, default 52).
    clip_low / clip_high : float
        Reward clipping bounds (§6.5, default ±5).
    sigma_mkt_prior : float
        Cold-start prior for σ_mkt,t (§6.4); used when insufficient
        in-episode history has accumulated.
    sigma_port_prior : float
        Cold-start prior for σ_t (§6.4).
    """

    def __init__(
        self,
        lambda_slow:  float = 0.75,
        lambda_tail:  float = 0.40,
        lambda_cost:  float = 1.0,
        lambda_cv:    float = 1.0,
        sigma_mkt_window:  int   = 13,
        sigma_port_window: int   = 52,
        clip_low:  float = -5.0,
        clip_high: float =  5.0,
        sigma_mkt_prior:  float = _DEFAULT_SIGMA_MKT_PRIOR,
        sigma_port_prior: float = _DEFAULT_SIGMA_PORT_PRIOR,
    ):
        self._lam_slow  = float(lambda_slow)
        self._lam_tail  = float(lambda_tail)
        self._lam_cost  = float(lambda_cost)
        self._lam_cv    = float(lambda_cv)

        self._w_mkt  = int(sigma_mkt_window)
        self._w_port = int(sigma_port_window)

        self._clip_low  = float(clip_low)
        self._clip_high = float(clip_high)

        self._prior_mkt  = float(sigma_mkt_prior)
        self._prior_port = float(sigma_port_prior)

        # Rolling return buffers (one entry per env step = one week)
        self._qqq_buf:  deque = deque(maxlen=self._w_mkt)
        self._port_buf: deque = deque(maxlen=self._w_port)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear rolling buffers.  Call at the start of each episode (§6.4)."""
        self._qqq_buf.clear()
        self._port_buf.clear()

    def compute(
        self,
        r_port_gross: float,
        r_qqq:        float,
        cost_t:       float,
        violations_t: float,
    ) -> Dict[str, float]:
        """Compute the 5-term reward for one weekly step.

        Parameters
        ----------
        r_port_gross : float
            Gross portfolio return = Σ w_exec_i × (close_{t+1}/close_t − 1).
            Price-change component only; cost deduction applied internally.
        r_qqq : float
            QQQ close-to-close return for the same weekly period.
        cost_t : float
            Total transaction cost fraction from §5.4 (summed across assets,
            divided by NAV).
        violations_t : float
            ||w_pre − w_exec||₂ from the ConstraintProjector output.

        Returns
        -------
        dict with keys:
            reward           – clipped scalar (float32)
            reward_unclipped – pre-clip scalar
            term1 .. term5   – individual terms (positive = reward contribution)
            e_t              – excess return (net)
            r_port_net       – net return after cost deduction
            sigma_mkt        – σ_mkt,t used this step
            sigma_port       – σ_t used this step
            norm_excess      – e_t / σ_mkt,t  (normalised excess)
            was_clipped      – bool
        """
        # ------------------------------------------------------------------
        # Net return  (§6.2: "costs reduce r_port_t" → double-cost §6.3)
        # ------------------------------------------------------------------
        r_port_net = (1.0 + r_port_gross) * max(0.0, 1.0 - cost_t) - 1.0

        # Excess return over QQQ benchmark
        e_t = r_port_net - r_qqq

        # ------------------------------------------------------------------
        # Update rolling buffers BEFORE computing σ so the current step's
        # return is included in the window (causal, end-of-period update).
        # ------------------------------------------------------------------
        self._qqq_buf.append(float(r_qqq))
        self._port_buf.append(float(r_port_net))

        # ------------------------------------------------------------------
        # Rolling volatility estimates with cold-start blending (§6.4)
        # ------------------------------------------------------------------
        sigma_mkt  = self._rolling_std(self._qqq_buf,  self._w_mkt,  self._prior_mkt)
        sigma_port = self._rolling_std(self._port_buf, self._w_port, self._prior_port)

        # Guard against degenerate zero σ (can happen with synthetic flat data)
        sigma_mkt  = max(sigma_mkt,  EPS)
        sigma_port = max(sigma_port, EPS)

        # ------------------------------------------------------------------
        # 5-term reward formula  (§6.1)
        # ------------------------------------------------------------------

        # Term 1: market-volatility-adjusted excess return
        norm_excess = e_t / sigma_mkt
        term1 = norm_excess

        # Term 2: slow vol penalty (discourages high absolute portfolio risk)
        term2 = self._lam_slow * sigma_port

        # Term 3: quadratic tail / downside penalty (Sortino-like; §6.2)
        #   activates only when e_t/σ < 0 (portfolio underperforms QQQ)
        tail  = max(0.0, -norm_excess)
        term3 = self._lam_tail * tail * tail

        # Term 4: explicit transaction cost penalty
        term4 = self._lam_cost * cost_t

        # Term 5: soft constraint violation penalty
        term5 = self._lam_cv * violations_t

        # Sum
        reward_raw = term1 - term2 - term3 - term4 - term5

        # Clip to ±5 (§6.5)
        reward = float(np.clip(reward_raw, self._clip_low, self._clip_high))

        return dict(
            reward           = reward,
            reward_unclipped = float(reward_raw),
            term1            = float(term1),
            term2            = float(term2),
            term3            = float(term3),
            term4            = float(term4),
            term5            = float(term5),
            e_t              = float(e_t),
            r_port_net       = float(r_port_net),
            sigma_mkt        = float(sigma_mkt),
            sigma_port       = float(sigma_port),
            norm_excess      = float(norm_excess),
            was_clipped      = bool(abs(reward_raw - reward) > 1e-9),
        )

    def compute_from_env(self, reward_components: dict) -> Dict[str, float]:
        """Convenience wrapper: takes the raw dict from env.step().

        Extracts r_port_t (gross), r_qqq_t, cost_t, violations_t and
        delegates to compute().
        """
        return self.compute(
            r_port_gross = float(reward_components["r_port_t"]),
            r_qqq        = float(reward_components["r_qqq_t"]),
            cost_t       = float(reward_components["cost_t"]),
            violations_t = float(reward_components["violations_t"]),
        )

    # ------------------------------------------------------------------
    # Properties for inspection / testing
    # ------------------------------------------------------------------

    @property
    def lambdas(self) -> Dict[str, float]:
        return dict(
            lambda_slow = self._lam_slow,
            lambda_tail = self._lam_tail,
            lambda_cost = self._lam_cost,
            lambda_cv   = self._lam_cv,
        )

    @property
    def n_steps_in_buffer(self) -> Dict[str, int]:
        return dict(qqq=len(self._qqq_buf), port=len(self._port_buf))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _rolling_std(self, buf: deque, window: int, prior: float) -> float:
        """Causal rolling std with cold-start blending (§6.4).

        At n < 2            : return prior (no variance estimate possible).
        At 2 ≤ n < window   : blend prior and in-episode std linearly.
        At n >= window       : return pure in-episode rolling std.

        Linear blend: alpha = (n-1) / (window-1), 0 at n=1, 1 at n=window.
        """
        n = len(buf)
        if n < 2:
            return prior

        vals     = list(buf)
        sigma_ep = float(np.std(vals, ddof=1))

        if n >= window:
            return sigma_ep

        # Cold-start blend: transition from prior to in-episode estimate
        alpha = (n - 1.0) / max(1.0, window - 1.0)
        return (1.0 - alpha) * prior + alpha * sigma_ep


# ---------------------------------------------------------------------------
# Module-level convenience constructor from config dataclass / dict
# ---------------------------------------------------------------------------

def from_config(cfg) -> RewardFunction:
    """Construct RewardFunction from a RewardConfig dataclass or dict.

    Accepts either the config_schema.RewardConfig dataclass or a plain dict
    with the same keys (matching master_config.yaml reward: section).
    """
    if hasattr(cfg, "__dict__"):
        d = vars(cfg)
    else:
        d = dict(cfg)

    return RewardFunction(
        lambda_slow       = d.get("lambda_slow",            0.75),
        lambda_tail       = d.get("lambda_tail",            0.40),
        lambda_cost       = d.get("lambda_cost",            1.0),
        lambda_cv         = d.get("lambda_cv",              1.0),
        sigma_mkt_window  = d.get("sigma_mkt_window_weeks", 13),
        sigma_port_window = d.get("sigma_port_window_weeks",52),
    )
