"""
evaluation/bootstrap.py
=======================
Block bootstrap confidence intervals for Project Apex (Bible §9.5).

Procedure:
  1. Concatenate all OOS weekly excess return series across folds → length T
  2. Block length b = floor(T^(1/3))   (typically 8–12 weeks)
  3. Resample 10,000 times by drawing blocks with replacement (moving block)
  4. For each sample compute metric (Excess CAGR, Sortino)
  5. 95% CI = [2.5th, 97.5th] percentiles

Pass criterion (§9.5):
  CI_lower(Excess CAGR) > 0
  CI_lower(Sortino)     > QQQ_Sortino
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .metrics import compute_excess_cagr, compute_sortino


# ---------------------------------------------------------------------------
# Core moving-block bootstrap
# ---------------------------------------------------------------------------

def moving_block_bootstrap(
    series:     np.ndarray,
    n_resamples: int                          = 10_000,
    block_length: Optional[int]               = None,
    metric_fn:  Callable[[np.ndarray], float] = compute_excess_cagr,
    rng:        Optional[np.random.Generator] = None,
    ci_levels:  Tuple[float, float]           = (2.5, 97.5),
) -> Dict:
    """
    Moving-block bootstrap for a 1-D return series.

    Parameters
    ----------
    series       : [T] weekly excess return series
    n_resamples  : number of bootstrap resamples (default 10,000 per §9.5)
    block_length : b; if None, uses floor(T^(1/3))
    metric_fn    : callable(returns) → scalar metric
    rng          : numpy Generator for reproducibility
    ci_levels    : (lower_pct, upper_pct) for confidence interval

    Returns
    -------
    dict with keys:
      point_estimate, ci_lower, ci_upper, ci_lower_pct, ci_upper_pct,
      n_resamples, block_length, T
    """
    series = np.asarray(series, dtype=float)
    T = len(series)
    if T == 0:
        return {"point_estimate": float("nan"), "ci_lower": float("nan"),
                "ci_upper": float("nan"), "block_length": 0, "T": 0}

    if block_length is None:
        block_length = max(1, int(math.floor(T ** (1.0 / 3.0))))

    if rng is None:
        rng = np.random.default_rng(0)

    n_blocks = math.ceil(T / block_length)
    max_start = T - block_length          # last valid block start index

    if max_start < 0:
        # Series shorter than one block — degenerate: each resample = full series
        metrics = np.array([metric_fn(series) for _ in range(n_resamples)])
    else:
        metrics = np.empty(n_resamples)
        for i in range(n_resamples):
            starts   = rng.integers(0, max_start + 1, size=n_blocks)
            blocks   = [series[s: s + block_length] for s in starts]
            resampled = np.concatenate(blocks)[:T]    # trim to exact length T
            metrics[i] = metric_fn(resampled)

    # Remove non-finite
    finite = metrics[np.isfinite(metrics)]
    if len(finite) == 0:
        return {"point_estimate": float("nan"), "ci_lower": float("nan"),
                "ci_upper": float("nan"), "block_length": block_length, "T": T,
                "n_resamples": n_resamples}

    return {
        "point_estimate": float(metric_fn(series)),
        "ci_lower":       float(np.percentile(finite, ci_levels[0])),
        "ci_upper":       float(np.percentile(finite, ci_levels[1])),
        "ci_lower_pct":   ci_levels[0],
        "ci_upper_pct":   ci_levels[1],
        "n_resamples":    n_resamples,
        "block_length":   block_length,
        "T":              T,
    }


# ---------------------------------------------------------------------------
# BlockBootstrap  (high-level API)
# ---------------------------------------------------------------------------

class BlockBootstrap:
    """
    §9.5 Block bootstrap CI computation for concatenated OOS excess return series.

    Parameters
    ----------
    n_resamples : number of bootstrap resamples (default 10,000)
    rng         : numpy Generator for reproducibility
    """

    def __init__(
        self,
        n_resamples: int                          = 10_000,
        rng:         Optional[np.random.Generator] = None,
    ) -> None:
        self.n_resamples = int(n_resamples)
        self.rng         = rng if rng is not None else np.random.default_rng(42)

    def run(
        self,
        excess_returns_oos:  np.ndarray,   # concatenated OOS excess return series
        qqq_returns_oos:     Optional[np.ndarray] = None,   # for QQQ Sortino criterion
        weeks_per_year:      float = 52.0,
    ) -> Dict:
        """
        Compute bootstrap CIs for Excess CAGR and Sortino (§9.5).

        Parameters
        ----------
        excess_returns_oos : [T] concatenated OOS weekly excess return series
        qqq_returns_oos    : [T] QQQ OOS returns (used to compute QQQ Sortino
                             for the pass criterion)
        weeks_per_year     : annualisation factor

        Returns
        -------
        dict with:
          excess_cagr_ci, sortino_ci,
          pass_excess_cagr, pass_sortino, all_pass,
          block_length, T
        """
        excess_returns_oos = np.asarray(excess_returns_oos, dtype=float)
        T = len(excess_returns_oos)

        block_length = max(1, int(math.floor(T ** (1.0 / 3.0))))

        cagr_ci = moving_block_bootstrap(
            excess_returns_oos,
            n_resamples=self.n_resamples,
            block_length=block_length,
            metric_fn=lambda r: compute_excess_cagr(r, weeks_per_year),
            rng=np.random.default_rng(self.rng.integers(0, 2**31)),
        )

        sortino_ci = moving_block_bootstrap(
            excess_returns_oos,
            n_resamples=self.n_resamples,
            block_length=block_length,
            metric_fn=lambda r: compute_sortino(r, weeks_per_year),
            rng=np.random.default_rng(self.rng.integers(0, 2**31)),
        )

        # §9.5 Pass criteria
        pass_cagr    = float(cagr_ci.get("ci_lower", float("nan"))) > 0.0
        qqq_sortino  = float("nan")
        pass_sortino = False

        if qqq_returns_oos is not None:
            # QQQ Sortino for comparison: QQQ excess return = 0 vs QQQ (self-referential)
            # In practice, QQQ excess return series ≈ 0, so its Sortino ≈ 0.
            # The criterion: CI_lower(Sortino) > QQQ_Sortino.
            qqq_r       = np.asarray(qqq_returns_oos, dtype=float)
            qqq_sortino = compute_sortino(qqq_r, weeks_per_year)
            pass_sortino = float(sortino_ci.get("ci_lower", float("nan"))) > qqq_sortino
        else:
            # Without QQQ series, check CI_lower > 0
            pass_sortino = float(sortino_ci.get("ci_lower", float("nan"))) > 0.0

        return {
            "excess_cagr_ci":   cagr_ci,
            "sortino_ci":       sortino_ci,
            "pass_excess_cagr": pass_cagr,
            "pass_sortino":     pass_sortino,
            "qqq_sortino":      qqq_sortino,
            "all_pass":         pass_cagr and pass_sortino,
            "block_length":     block_length,
            "T":                T,
        }

    def concatenate_fold_returns(
        self,
        fold_excess_returns: List[np.ndarray],
    ) -> np.ndarray:
        """
        Concatenate OOS excess return series from all folds (§9.5 step 1).

        Parameters
        ----------
        fold_excess_returns : list of [T_fold_i] arrays, one per fold

        Returns
        -------
        [T_total] concatenated series
        """
        if not fold_excess_returns:
            return np.array([], dtype=float)
        return np.concatenate([np.asarray(r, dtype=float) for r in fold_excess_returns])
