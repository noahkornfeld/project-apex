"""
environment/trading_env.py
Gym-style Trading Environment — Phase 5 (Bible §5).

Implements:
  §5.1  Decision/execution timeline  (observe close d_t, execute open d_{t+1})
  §5.2  Weekly rebalance mechanics
  §5.3  NAV evolution  NAV_{t+1} = NAV_t × (1 + r_port_t) × (1 - cost_t)
  §5.4  Transaction cost model  (4-term: commission, spread, impact, gap)
  §5.5  QQQ benchmark NAV tracking
  §5.6  Episode lifecycle  (reset / termination)
  §5.7  Forced-liquidation mechanics  (NDX removal + missingness → mask=0)
  §4.6  Counterfactual w_pre NAV tracking

step() returns raw reward components (r_port_t, r_qqq_t, cost_t, violations_t).
Full 5-term reward formula (§6) is implemented in environment/reward_fn.py.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Dict, Optional, Tuple, Any

from features.portfolio_state_features import (
    compute_portfolio_state,
    N_PORTFOLIO_STATE_FEATURES,
)

# Offset of portfolio-state slice inside g  (9 macro + 3 benchmark = 12)
_PORT_STATE_OFFSET = 9 + 3

EPS = 1e-8

# §5.4 cost model constants (also in master_config.yaml)
COMMISSION_BPS     = 1.0
SPREAD_COEFF       = 2.0
SPREAD_ADV_EXP     = 0.3
SPREAD_VOL_FLOOR   = 0.20
IMPACT_COEFF       = 10.0
IMPACT_SIZE_EXP    = 0.5
GAP_COEFF          = 1.5
GAP_SCALING        = 0.7979     # √(2/π)

# §5.7 forced-liquidation thresholds
MISSING_L_THRESH   = 2          # missing days in 60-day window
STREAK_THRESH      = 3          # consecutive missing days


class TradingEnvironment:
    """
    Gym-compatible trading environment for Project Apex.

    All price / cost inputs are pre-built numpy arrays indexed by the same
    daily row index as the feature panel (see environment/market_data.py).
    The environment selects weekly rebalance dates via `weekly_idx`.

    Parameters
    ----------
    md : dict
        Market-data dict from market_data.build_market_data() or
        market_data.make_synthetic_market_data().  Required keys:
          adj_close, adj_open, adv63, vol_252, gap_vol_252,
          sector_ids, qqq_close, vix, weekly_idx, dates_str,
          active_ids, mask_panel, x_panel, g_panel
    projector : ConstraintProjector or None
        If None, w_pre is used as-is (for tests that bypass projection).
    L_lookback : int
        Lookback window in trading days for observations returned by reset/step.
    cost_config : dict, optional
        Override default cost constants (keys mirror COMMISSION_BPS etc.).
    missingness_config : dict or MissingnessConfig-like, optional
        Override missingness thresholds.  Recognised keys:
          missing_in_window_threshold  (default MISSING_L_THRESH = 2)
          consecutive_missing_threshold (default STREAK_THRESH    = 3)
          temporal_window_days          (default L_lookback)
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        md: Dict[str, np.ndarray],
        projector=None,
        L_lookback: int = 60,
        cost_config: Optional[Dict] = None,
        missingness_config=None,
    ):
        # Market data arrays (all indexed by daily panel row)
        self._adj_close    = md["adj_close"]     # [T, K]
        self._adj_open     = md["adj_open"]      # [T, K]
        self._adv63        = md["adv63"]         # [T, K]
        self._vol_252      = md["vol_252"]       # [T, K]
        self._gap_vol_252  = md["gap_vol_252"]   # [T, K]
        self._sector_ids   = md["sector_ids"]    # [T, K]
        self._qqq_close    = md["qqq_close"]     # [T]
        self._vix          = md["vix"]           # [T]
        self._weekly_idx   = md["weekly_idx"]    # [W] panel row indices
        self._dates_str    = md["dates_str"]     # [T] str dates
        self._active_ids   = md["active_ids"]    # [T, K]
        self._mask_panel   = md["mask_panel"]    # [T, K]
        self._x_panel      = md["x_panel"]       # [T, K, F]
        self._g_panel      = md["g_panel"]       # [T, D]

        self._T, self._K  = self._adj_close.shape
        self._W           = len(self._weekly_idx)  # number of weekly steps
        self._L           = L_lookback

        self._projector   = projector

        # Cost config overrides
        self._cc = dict(
            commission_bps   = COMMISSION_BPS,
            spread_coeff     = SPREAD_COEFF,
            spread_adv_exp   = SPREAD_ADV_EXP,
            spread_vol_floor = SPREAD_VOL_FLOOR,
            impact_coeff     = IMPACT_COEFF,
            impact_size_exp  = IMPACT_SIZE_EXP,
            gap_coeff        = GAP_COEFF,
            gap_scaling      = GAP_SCALING,
        )
        if cost_config:
            self._cc.update(cost_config)

        # §5.7 Missingness thresholds — read from config if supplied
        _mc = {}
        if missingness_config is not None:
            _mc = vars(missingness_config) if hasattr(missingness_config, "__dict__") else dict(missingness_config)
        self._miss_L_thresh      = int(_mc.get("missing_in_window_threshold",  MISSING_L_THRESH))
        self._miss_streak_thresh = int(_mc.get("consecutive_missing_threshold", STREAK_THRESH))
        self._miss_window        = int(_mc.get("temporal_window_days",          L_lookback))

        # Episode state (initialised in reset)
        self._ep_step      = 0          # step within episode (0-based)
        self._nav          = 1.0        # portfolio NAV
        self._peak_nav     = 1.0        # running peak NAV (for drawdown)
        self._nav_pre      = 1.0        # counterfactual w_pre NAV  (§4.6)
        self._nav_qqq      = 1.0        # QQQ benchmark NAV
        self._w_exec       = np.zeros(self._K, dtype=np.float32)
        self._w_exec_prev  = np.zeros(self._K, dtype=np.float32)  # for turnover
        self._w_pre_saved  = np.zeros(self._K, dtype=np.float32)

        # §5.7 Missingness counters (reset each episode)
        self._missing_L      = np.zeros(self._K, dtype=np.int32)   # count in window
        self._streak_missing = np.zeros(self._K, dtype=np.int32)   # consecutive streak

        # Rolling return history for reward computation (Phase 6)
        self._ret_port_hist: list = []
        self._ret_qqq_hist:  list = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[Dict[str, Any], Dict]:
        """Initialise episode.  Returns (obs, info).

        §5.6.1: NAV=1.0, w_exec=0, rolling stats zeroed.
        First valid step = first weekly index with sufficient lookback.
        """
        # First valid step: need at least L weeks of panel history
        self._ep_step = self._first_valid_step()

        self._nav          = 1.0
        self._peak_nav     = 1.0
        self._nav_pre      = 1.0
        self._nav_qqq      = 1.0
        self._w_exec       = np.zeros(self._K, dtype=np.float32)
        self._w_exec_prev    = np.zeros(self._K, dtype=np.float32)
        self._w_pre_saved    = np.zeros(self._K, dtype=np.float32)
        self._missing_L      = np.zeros(self._K, dtype=np.int32)
        self._streak_missing = np.zeros(self._K, dtype=np.int32)
        self._ret_port_hist  = []
        self._ret_qqq_hist   = []

        obs  = self._get_obs(self._ep_step)
        info = {
            "date":    self._step_date(self._ep_step),
            "ep_step": self._ep_step,
            "nav":     self._nav,
            "nav_qqq": self._nav_qqq,
        }
        return obs, info

    def step(
        self,
        w_pre: np.ndarray,
    ) -> Tuple[Dict[str, Any], Dict[str, float], bool, Dict]:
        """Execute one weekly rebalance step.

        §5.1 Timeline
        -------------
        Observe at close of d_t  (self._ep_step)
        Execute  at open  of d_{t+1}  (self._ep_step + 1)
        NAV marked at close of d_{t+1}

        Parameters
        ----------
        w_pre : [K] float32 – pre-projection weight vector from actor.

        Returns
        -------
        obs              : dict – observation at d_{t+1}
        reward_components: dict – stub components for Phase 6
        done             : bool
        info             : dict
        """
        t_cur  = self._ep_step          # current weekly-step index
        t_next = t_cur + 1              # next weekly-step index

        # ----------------------------------------------------------
        # §5.1 Enforce temporal ordering: t_cur < t_next
        # ----------------------------------------------------------
        assert t_next < self._W, "step() called past episode end"

        p_cur  = self._weekly_idx[t_cur]   # panel row at d_t
        p_next = self._weekly_idx[t_next]  # panel row at d_{t+1}

        # Sanity: observation date < execution date
        assert self._dates_str[p_cur] < self._dates_str[p_next], (
            f"Timeline violation: obs {self._dates_str[p_cur]} "
            f">= exec {self._dates_str[p_next]}"
        )

        # ----------------------------------------------------------
        # §5.7 Forced-liquidation mask update
        # Rule 1 (NDX removal): mask_panel transition 1→0.
        # Rule 2 (Missingness): missing_L[i] > threshold OR
        #                        streak_missing[i] > threshold.
        # ----------------------------------------------------------
        mask_cur  = self._mask_panel[p_cur].copy()   # [K]
        mask_next = self._mask_panel[p_next].copy()  # [K] (mask at d_{t+1})

        # Update rolling missingness counters using the current panel row
        miss_force = self._update_missingness_counters(p_cur)  # [K] bool

        # Apply missingness overrides to both current and next masks
        mask_cur[miss_force]  = 0.0
        mask_next[miss_force] = 0.0

        # Forced-liquidation: any asset currently held that is now inactive
        # (either from NDX removal at p_next or missingness at p_cur)
        forced_liq_slots = np.where(
            (self._w_exec > 0) & (mask_next < 0.5)
        )[0]

        # ----------------------------------------------------------
        # Project w_pre → w_exec  (§4.4 Step 8)
        # ----------------------------------------------------------
        w_pre_np = np.asarray(w_pre, dtype=np.float32).copy()
        w_pre_np = w_pre_np * mask_cur        # zero inactive

        if self._projector is not None:
            import torch
            sector_ids_cur = self._sector_ids[p_cur]
            mask_t  = torch.from_numpy(mask_cur).float()
            sid_t   = torch.from_numpy(sector_ids_cur).long()
            w_pre_t = torch.from_numpy(w_pre_np).float()
            with torch.no_grad():
                w_exec_t = self._projector(w_pre_t, mask_t, sid_t)
            w_exec = w_exec_t.numpy().astype(np.float32)
        else:
            # No projector: normalise manually (for tests)
            w_exec = self._simple_normalize(w_pre_np, mask_cur)

        # Force-liquidate slots that become inactive
        if len(forced_liq_slots) > 0:
            forced_liq_weight = float(w_exec[forced_liq_slots].sum())
            n_active_next = int(mask_next.sum())
            
            if n_active_next > 0:
                # Normal case: redistribute to remaining active assets
                w_exec[forced_liq_slots] = 0.0
                w_exec += (forced_liq_weight / n_active_next) * mask_next
            else:
                # Edge case: no active assets next period
                # Keep current weights (will be liquidated at market close)
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"[ENV] No active assets in next period, keeping current weights")
                # Don't zero out - let the portfolio ride until assets become active again
        else:
            forced_liq_weight = 0.0

        # ----------------------------------------------------------
        # Transaction costs  (§5.4)  – executed at open of d_{t+1}
        # ----------------------------------------------------------
        adj_open_next = self._adj_open[p_next]         # [K] execution price
        delta_w       = w_exec - self._w_exec          # [K] weight change
        nav_cur       = self._nav

        cost_t, cost_per_asset = self._compute_cost(
            delta_w        = delta_w,
            nav            = nav_cur,
            adv63          = self._adv63[p_cur],
            vol_252        = self._vol_252[p_cur],
            gap_vol_252    = self._gap_vol_252[p_cur],
            vix            = float(self._vix[p_cur]),
            mask           = mask_cur,
        )

        # ----------------------------------------------------------
        # Portfolio return  (§5.3)  close d_t → close d_{t+1}
        # ----------------------------------------------------------
        close_cur  = self._adj_close[p_cur]    # [K]
        close_next = self._adj_close[p_next]   # [K]

        ret_asset = np.where(
            close_cur > EPS,
            close_next / (close_cur + EPS) - 1.0,
            0.0,
        ).astype(np.float32)
        ret_asset = np.clip(ret_asset, -0.99, 1.0)   # §5.3 guard: cap weekly asset returns at physically plausible range

        r_port_t = float(np.dot(self._w_exec, ret_asset))   # uses PREV w_exec

        # Counterfactual w_pre return (§4.6)
        r_pre_t  = float(np.dot(self._w_pre_saved, ret_asset))

        # ----------------------------------------------------------
        # NAV evolution  NAV_{t+1} = NAV_t × (1 + r_port) × (1 - cost)
        # ----------------------------------------------------------
        self._nav     = nav_cur * (1.0 + r_port_t) * max(0.0, 1.0 - cost_t)
        self._nav_pre = self._nav_pre * (1.0 + r_pre_t)   # no cost for cfact

        # ----------------------------------------------------------
        # QQQ NAV  (§5.5)
        # ----------------------------------------------------------
        qqq_cur  = float(self._qqq_close[p_cur])
        qqq_next = float(self._qqq_close[p_next])
        r_qqq_t  = (qqq_next / (qqq_cur + EPS)) - 1.0 if qqq_cur > EPS else 0.0
        self._nav_qqq = self._nav_qqq * (1.0 + r_qqq_t)

        # ----------------------------------------------------------
        # Accumulate rolling histories
        # ----------------------------------------------------------
        self._ret_port_hist.append(r_port_t)
        self._ret_qqq_hist.append(r_qqq_t)

        # Update state for next step
        self._w_exec_prev  = self._w_exec.copy()   # save before overwriting
        self._w_exec       = w_exec.copy()
        self._w_pre_saved  = w_pre_np.copy()
        self._peak_nav     = max(self._peak_nav, self._nav)
        self._ep_step      = t_next

        # ----------------------------------------------------------
        # Termination  (§5.6.2)
        # ----------------------------------------------------------
        done = (t_next >= self._W - 1)

        # ----------------------------------------------------------
        # Observation at d_{t+1}
        # ----------------------------------------------------------
        obs = {} if done else self._get_obs(t_next)

        # ----------------------------------------------------------
        # Reward components stub  (full reward in Phase 6)
        # ----------------------------------------------------------
        violations_t = float(np.linalg.norm(w_pre_np - w_exec))
        reward_components = dict(
            r_port_t          = r_port_t,
            r_qqq_t           = r_qqq_t,
            excess_t          = r_port_t - r_qqq_t,
            cost_t            = cost_t,
            violations_t      = violations_t,
            nav               = self._nav,
            nav_qqq           = self._nav_qqq,
            nav_pre           = self._nav_pre,
            forced_liq_weight = forced_liq_weight,
            forced_liq_count  = int(len(forced_liq_slots)),
        )

        info = dict(
            date_obs    = self._dates_str[p_cur],
            date_exec   = self._dates_str[p_next],
            ep_step     = t_next,
            nav         = self._nav,
            nav_qqq     = self._nav_qqq,
            nav_pre     = self._nav_pre,
            cost_bps    = cost_t * 1e4,
            forced_liq  = len(forced_liq_slots) > 0,
        )

        return obs, reward_components, done, info

    # ------------------------------------------------------------------
    # Cost model  §5.4
    # ------------------------------------------------------------------

    def compute_cost_public(
        self,
        delta_w: np.ndarray,
        nav: float,
        adv63: np.ndarray,
        vol_252: np.ndarray,
        gap_vol_252: np.ndarray,
        vix: float,
        mask: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Public wrapper for unit-testing the cost model."""
        return self._compute_cost(delta_w, nav, adv63, vol_252,
                                  gap_vol_252, vix, mask)

    def _compute_cost(
        self,
        delta_w: np.ndarray,
        nav: float,
        adv63: np.ndarray,
        vol_252: np.ndarray,
        gap_vol_252: np.ndarray,
        vix: float,
        mask: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """§5.4  4-term transaction cost model.

        total_cost(i,t) = |trade_$| / 10_000
            × [ 1.0                                                  Term1
              + M × 2.0 × (ADV_med/ADV_i)^0.3 × max(1,σ/0.20)      Term2
              + M × 10  × (|trade_$|/ADV_i)^0.5 × σ/0.20           Term3
              + 1.5 × σ_gap × 0.7979 ]                               Term4

        Returns (total_cost_fraction, per_asset_cost_array).
        """
        cc = self._cc
        K  = len(delta_w)

        trade_dollars = np.abs(delta_w) * nav          # [K]
        active        = mask > 0.5

        adv_safe      = np.where(active & (adv63 > EPS), adv63, EPS)
        vol_safe      = np.where(active, np.clip(vol_252, EPS, None), cc["spread_vol_floor"])
        gap_safe      = np.where(active, np.clip(gap_vol_252, 0.0, None), 0.0)

        # ADV_median across active assets (causal, cross-sectional)
        adv_active    = adv63[active]
        adv_median    = float(np.median(adv_active)) if len(adv_active) > 0 else EPS

        # Market conditions multiplier M(i,t) = max(1, VIX/20)  §5.4
        M             = max(1.0, vix / 20.0)

        # Term 1: Commission (1 bps flat)
        term1 = cc["commission_bps"] * np.ones(K, dtype=np.float64)

        # Term 2: Bid-ask spread
        adv_ratio     = adv_median / adv_safe
        vol_factor    = np.maximum(1.0, vol_safe / cc["spread_vol_floor"])
        term2         = (M * cc["spread_coeff"]
                        * np.power(adv_ratio, cc["spread_adv_exp"])
                        * vol_factor)

        # Term 3: Market impact (square-root model)
        size_ratio    = trade_dollars / adv_safe
        term3         = (M * cc["impact_coeff"]
                        * np.power(np.maximum(size_ratio, 0.0), cc["impact_size_exp"])
                        * vol_safe / cc["spread_vol_floor"])

        # Term 4: Gap risk
        term4         = cc["gap_coeff"] * gap_safe * cc["gap_scaling"]

        # Total cost per asset in bps, then convert to fraction
        cost_bps_per_asset = trade_dollars / 1e4 * (term1 + term2 + term3 + term4)

        # Zero out inactive or zero-trade slots
        cost_bps_per_asset = np.where(active & (trade_dollars > EPS),
                                       cost_bps_per_asset, 0.0)

        total_cost_fraction = float(cost_bps_per_asset.sum() / (nav + EPS))

        return total_cost_fraction, cost_bps_per_asset.astype(np.float32)

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _get_obs(self, ep_step: int) -> Dict[str, Any]:
        """Build observation dict at weekly step `ep_step`.

        Returns a window of x_panel features [L_days, K, F],
        the global vector g[D], the mask [K], sector_ids [K],
        and active_ids [K] at the current panel row.

        The last 8 slots of g (portfolio-state features) are computed
        live from current episode state and patched in here, replacing the
        zeros written during panel precomputation (§3.4.1).
        """
        p_idx = self._weekly_idx[ep_step]

        # Lookback window is L trading days (not weeks)
        p_start = max(0, p_idx - self._L)
        x_win   = self._x_panel[p_start : p_idx + 1]   # [<=L_days, K, F]

        # ------------------------------------------------------------------
        # Build live g by patching portfolio-state slice (§3.4.1 fix)
        # ------------------------------------------------------------------
        g = self._g_panel[p_idx].copy()   # copy so we don't mutate the panel

        # Estimated cost: simulate cost of current holdings as proxy for
        # "expected cost next rebalance" (§3.4.1)
        est_cost = 0.0
        w_sq = float(np.dot(self._w_exec, self._w_exec))
        if w_sq > 1e-8:
            est_cost, _ = self._compute_cost(
                delta_w     = self._w_exec,
                nav         = self._nav,
                adv63       = self._adv63[p_idx],
                vol_252     = self._vol_252[p_idx],
                gap_vol_252 = self._gap_vol_252[p_idx],
                vix         = float(self._vix[p_idx]),
                mask        = self._mask_panel[p_idx],
            )

        # 52-week daily QQQ log-returns window (up to 260 trading days back)
        qqq_start = max(0, p_idx - 260)
        qqq_prices = self._qqq_close[qqq_start : p_idx + 1]
        if len(qqq_prices) >= 2:
            qqq_daily_rets = np.diff(np.log(np.maximum(qqq_prices, 1e-8)))
        else:
            qqq_daily_rets = np.array([], dtype=np.float32)

        port_state = compute_portfolio_state(
            w_exec         = self._w_exec,
            w_exec_prev    = self._w_exec_prev,
            nav            = self._nav,
            peak_nav       = self._peak_nav,
            ret_port_hist  = self._ret_port_hist,
            ret_qqq_hist   = self._ret_qqq_hist,
            qqq_daily_rets = qqq_daily_rets,
            mask           = self._mask_panel[p_idx],
            estimated_cost = est_cost,
        )

        g[_PORT_STATE_OFFSET : _PORT_STATE_OFFSET + N_PORTFOLIO_STATE_FEATURES] = port_state

        return dict(
            x           = x_win,
            g           = g,
            mask        = self._mask_panel[p_idx],
            sector_ids  = self._sector_ids[p_idx],
            active_ids  = self._active_ids[p_idx],
            date        = self._dates_str[p_idx],
            panel_idx   = int(p_idx),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_missingness_counters(self, p_cur: int) -> np.ndarray:
        """Update missing_L and streak_missing counters for all assets.

        §5.7: An asset is treated as missing on a daily row when its
        adj_close is ≤ 0 or NaN.

        Parameters
        ----------
        p_cur : int
            Current panel row index (daily).

        Returns
        -------
        force_inactive : np.ndarray [K] bool
            True where missing_L > threshold OR streak > threshold.
            These assets should be overridden to inactive in mask_cur/next.
        """
        win  = self._miss_window
        p0   = max(0, p_cur - win + 1)

        close_win = self._adj_close[p0 : p_cur + 1]            # [w, K]
        missing   = (close_win <= EPS) | np.isnan(close_win)   # [w, K] bool

        # --- missing_L[i]: count of missing days in the rolling window ---
        self._missing_L = missing.sum(axis=0).astype(np.int32)  # [K]

        # --- streak_missing[i]: consecutive missing days ending at p_cur ---
        # Reverse the window so index 0 = most recent day.
        # Find the first non-missing day from the end; the streak = that index.
        miss_rev      = missing[::-1]          # [w, K] reversed
        not_miss_rev  = ~miss_rev              # True = not missing
        # argmax returns index of first True; if all False (all missing),
        # argmax returns 0 — so we guard with an all-missing flag.
        all_missing   = not_miss_rev.sum(axis=0) == 0   # [K] bool
        first_ok      = np.argmax(not_miss_rev, axis=0)  # [K] int
        self._streak_missing = np.where(
            all_missing,
            len(close_win),          # entire window is missing
            first_ok,
        ).astype(np.int32)

        force_inactive = (
            (self._missing_L      > self._miss_L_thresh) |
            (self._streak_missing > self._miss_streak_thresh)
        )
        return force_inactive

    def _first_valid_step(self) -> int:
        """Find first weekly step with ≥ L trading days of panel history (§5.6.1)."""
        for i, pidx in enumerate(self._weekly_idx):
            if pidx >= self._L:
                return i
        return 0   # fallback: start from first available step

    def _step_date(self, ep_step: int) -> str:
        p_idx = self._weekly_idx[ep_step]
        return self._dates_str[p_idx]

    @staticmethod
    def _simple_normalize(w: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fallback normalisation when no projector is provided (tests only)."""
        w = np.maximum(w, 0.0) * mask
        s = w.sum()
        if s > EPS:
            return w / s
        n = int(mask.sum())
        if n > 0:
            return mask / n
        return w

    # ------------------------------------------------------------------
    # Properties for diagnostics
    # ------------------------------------------------------------------

    @property
    def nav(self) -> float:
        return self._nav

    @property
    def nav_qqq(self) -> float:
        return self._nav_qqq

    @property
    def nav_pre(self) -> float:
        return self._nav_pre

    @property
    def n_steps(self) -> int:
        """Total number of steps in this episode (= W - 1)."""
        return max(0, self._W - 1 - self._first_valid_step())

    @property
    def ep_step(self) -> int:
        return self._ep_step

    @property
    def w_exec(self) -> np.ndarray:
        return self._w_exec.copy()

    @property
    def ret_history(self) -> list:
        return list(self._ret_port_hist)

    @property
    def qqq_ret_history(self) -> list:
        return list(self._ret_qqq_hist)
