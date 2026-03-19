"""
evaluation/metrics.py
=====================
Evaluation metrics for Project Apex (Bible §9.3).

Primary   (Table 32): Excess CAGR, Sortino Ratio, Max Drawdown
Secondary (Table 33): Sharpe, Turnover, Cost Drag, Effective N Positions
Tertiary  (Table 34): Skewness, Kurtosis, CVaR(5%), Hit Rate, Beta,
                       Information Ratio, Rank IC
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Primary metrics  (§9.3.1 / Table 32)
# ---------------------------------------------------------------------------

def compute_excess_cagr(
    excess_returns:  np.ndarray,
    weeks_per_year:  float = 52.0,
) -> float:
    """
    Annualised Excess CAGR.

      CAGR = (Π_t (1 + r_excess_t))^(52/T) − 1

    where r_excess_t = portfolio weekly return − QQQ weekly return.
    """
    excess_returns = np.asarray(excess_returns, dtype=float)
    T = len(excess_returns)
    if T == 0:
        return float("nan")
    cum = float(np.prod(1.0 + np.clip(excess_returns, -0.9999, None)))
    if cum <= 0:
        return float("-inf")
    return cum ** (weeks_per_year / T) - 1.0


def compute_sortino(
    excess_returns:  np.ndarray,
    weeks_per_year:  float = 52.0,
    mar:             float = 0.0,
) -> float:
    """
    Sortino Ratio (annualised).

      Sortino = mean(r_excess) / σ_down × √52

    where σ_down = std of negative excess returns relative to MAR (§9.3 Table 32).
    """
    excess_returns = np.asarray(excess_returns, dtype=float)
    if len(excess_returns) == 0:
        return float("nan")

    downside = excess_returns[excess_returns < mar]
    if len(downside) == 0:
        mu = float(excess_returns.mean())
        return float("inf") if mu > 0 else float("nan")

    sigma_down = float(np.std(downside, ddof=1))
    if sigma_down == 0.0:
        return float("inf")

    return float(excess_returns.mean() / sigma_down * math.sqrt(weeks_per_year))


def compute_max_drawdown(nav: np.ndarray) -> float:
    """
    Maximum peak-to-trough decline in cumulative portfolio NAV (§9.3 Table 32).
    Returns a negative value (or 0.0).
    """
    nav = np.asarray(nav, dtype=float)
    if len(nav) == 0:
        return float("nan")
    peak  = nav[0]
    max_dd = 0.0
    for v in nav:
        if v > peak:
            peak = v
        dd = (v - peak) / peak if peak != 0 else 0.0
        if dd < max_dd:
            max_dd = dd
    return float(max_dd)


# ---------------------------------------------------------------------------
# Secondary metrics  (§9.3.2 / Table 33)
# ---------------------------------------------------------------------------

def compute_sharpe(
    excess_returns: np.ndarray,
    weeks_per_year: float = 52.0,
) -> float:
    """Sharpe = mean(r_excess) / std(r_excess) × √52  (annualised)."""
    excess_returns = np.asarray(excess_returns, dtype=float)
    if len(excess_returns) < 2:
        return float("nan")
    std = float(np.std(excess_returns, ddof=1))
    if std == 0.0:
        return float("nan")
    return float(excess_returns.mean() / std * math.sqrt(weeks_per_year))


def compute_turnover_mean(turnover: np.ndarray) -> float:
    """Mean one-way turnover over OOS period."""
    t = np.asarray(turnover, dtype=float)
    return float(np.nanmean(t)) if len(t) > 0 else float("nan")


def compute_cost_drag(cost_bps: np.ndarray, gross_returns: np.ndarray) -> float:
    """
    Cost drag = total transaction costs / |compounded gross portfolio return| over OOS.
    cost_bps in basis points; gross_returns as fractions.

    Uses compounded return (not sum of absolutes) so that a single extreme-return
    week cannot artificially inflate the denominator and deflate cost_drag.
    """
    cost_bps      = np.asarray(cost_bps,      dtype=float)
    gross_returns = np.asarray(gross_returns, dtype=float)
    if len(cost_bps) == 0:
        return float("nan")
    total_cost_bps = float(np.nansum(cost_bps))
    compounded     = float(np.prod(1.0 + np.clip(gross_returns, -0.9999, None))) - 1.0
    gross_pnl_bps  = abs(compounded) * 10_000
    if gross_pnl_bps == 0.0:
        return float("nan")
    return total_cost_bps / gross_pnl_bps


def compute_effective_n_positions(w_exec: np.ndarray) -> float:
    """
    Effective N positions = mean(1 / Σ_i w_exec_i²) over OOS weeks.

    w_exec : [T, K] executed weight matrix.
    """
    w = np.asarray(w_exec, dtype=float)
    if w.ndim == 1:
        w = w[np.newaxis, :]
    herfindahl = np.sum(w ** 2, axis=1)          # [T]
    valid = herfindahl[herfindahl > 1e-12]
    if len(valid) == 0:
        return float("nan")
    return float(np.mean(1.0 / valid))


# ---------------------------------------------------------------------------
# Tertiary metrics  (§9.3.3 / Table 34)
# ---------------------------------------------------------------------------

def compute_skewness(excess_returns: np.ndarray) -> float:
    r = np.asarray(excess_returns, dtype=float)
    if len(r) < 3:
        return float("nan")
    return float(scipy_stats.skew(r, nan_policy="omit"))


def compute_kurtosis(excess_returns: np.ndarray) -> float:
    """Excess kurtosis (relative to normal)."""
    r = np.asarray(excess_returns, dtype=float)
    if len(r) < 4:
        return float("nan")
    return float(scipy_stats.kurtosis(r, nan_policy="omit"))


def compute_cvar(excess_returns: np.ndarray, alpha: float = 0.05) -> float:
    """CVaR(5%) = mean excess return in the worst alpha fraction of weeks."""
    r = np.asarray(excess_returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return float("nan")
    cutoff = np.percentile(r, alpha * 100)
    worst  = r[r <= cutoff]
    if len(worst) == 0:
        return float("nan")
    return float(worst.mean())


def compute_hit_rate(excess_returns: np.ndarray) -> float:
    """Fraction of OOS weeks where r_excess_t > 0."""
    r = np.asarray(excess_returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return float("nan")
    return float(np.mean(r > 0))


def compute_beta_to_qqq(
    portfolio_returns: np.ndarray,
    qqq_returns:       np.ndarray,
    winsor_pct:        float = 0.02,
) -> float:
    """
    Regression coefficient of portfolio return on QQQ return.

    Winsorises both series at the winsor_pct / (1-winsor_pct) percentiles
    before regression so that a handful of extreme weeks (e.g. from data
    quality spikes) cannot distort the beta estimate.
    """
    p = np.asarray(portfolio_returns, dtype=float)
    q = np.asarray(qqq_returns,       dtype=float)
    mask = ~(np.isnan(p) | np.isnan(q))
    if mask.sum() < 2:
        return float("nan")
    p_clean = p[mask]
    q_clean = q[mask]
    lo_p, hi_p = np.percentile(p_clean, [winsor_pct * 100, (1 - winsor_pct) * 100])
    lo_q, hi_q = np.percentile(q_clean, [winsor_pct * 100, (1 - winsor_pct) * 100])
    p_w = np.clip(p_clean, lo_p, hi_p)
    q_w = np.clip(q_clean, lo_q, hi_q)
    slope, _, _, _, _ = scipy_stats.linregress(q_w, p_w)
    return float(slope)


def compute_rank_ic(
    w_exec:        np.ndarray,   # [T, K] executed weights this week
    asset_returns: np.ndarray,   # [T, K] realized asset returns next week
) -> float:
    """
    Rank IC = mean Spearman correlation of w_exec weights with next-week asset
    returns, averaged over OOS weeks.
    """
    w = np.asarray(w_exec,        dtype=float)
    r = np.asarray(asset_returns, dtype=float)
    if w.ndim == 1:
        w = w[np.newaxis, :]
        r = r[np.newaxis, :]
    T = min(len(w), len(r))
    if T == 0:
        return float("nan")

    ics = []
    for t in range(T):
        wt = w[t]
        rt = r[t]
        valid = ~(np.isnan(wt) | np.isnan(rt))
        if valid.sum() < 2:
            continue
        corr, _ = scipy_stats.spearmanr(wt[valid], rt[valid])
        if not math.isnan(corr):
            ics.append(corr)

    return float(np.mean(ics)) if ics else float("nan")


# ---------------------------------------------------------------------------
# Combined: compute_all_metrics
# ---------------------------------------------------------------------------

def compute_all_metrics(
    nav:               np.ndarray,          # [T] portfolio cumulative NAV (OOS)
    excess_returns:    np.ndarray,          # [T] weekly excess returns vs QQQ
    qqq_returns:       np.ndarray,          # [T] QQQ weekly returns
    portfolio_returns: Optional[np.ndarray] = None,  # [T] gross portfolio weekly returns
    turnover:          Optional[np.ndarray] = None,  # [T] weekly one-way turnover
    cost_bps:          Optional[np.ndarray] = None,  # [T] weekly cost in bps
    w_exec:            Optional[np.ndarray] = None,  # [T, K] executed weights
    asset_returns:     Optional[np.ndarray] = None,  # [T, K] asset returns next week
    weeks_per_year:    float                = 52.0,
) -> Dict:
    """
    Compute all primary, secondary, and tertiary metrics.

    Returns
    -------
    dict with all metric keys.  NaN where input insufficient.
    """
    nav            = np.asarray(nav,            dtype=float)
    excess_returns = np.asarray(excess_returns, dtype=float)
    qqq_returns    = np.asarray(qqq_returns,    dtype=float)

    portfolio_returns = (
        np.asarray(portfolio_returns, dtype=float)
        if portfolio_returns is not None
        else excess_returns + qqq_returns
    )

    # ---- Primary (§9.3.1) ----
    primary = {
        "excess_cagr":  compute_excess_cagr(excess_returns, weeks_per_year),
        "sortino":      compute_sortino(portfolio_returns, weeks_per_year),
        "max_drawdown": compute_max_drawdown(nav),
    }

    # ---- Secondary (§9.3.2) ----
    secondary = {
        "sharpe":          compute_sharpe(portfolio_returns, weeks_per_year),
        "turnover_mean":   compute_turnover_mean(turnover) if turnover is not None else float("nan"),
        "cost_drag":       (
            compute_cost_drag(cost_bps, portfolio_returns)
            if cost_bps is not None else float("nan")
        ),
        "effective_n_positions": (
            compute_effective_n_positions(w_exec)
            if w_exec is not None else float("nan")
        ),
    }

    # ---- Tertiary (§9.3.3) ----
    beta = compute_beta_to_qqq(portfolio_returns, qqq_returns)
    tertiary = {
        "skewness":         compute_skewness(excess_returns),
        "kurtosis":         compute_kurtosis(excess_returns),
        "cvar_5pct":        compute_cvar(excess_returns, 0.05),
        "hit_rate":         compute_hit_rate(excess_returns),
        "beta_to_qqq":      beta,
        "information_ratio": compute_sharpe(excess_returns, weeks_per_year),  # IR = Sharpe of active returns; same formula when benchmark is QQQ (§9.3.3)
        "rank_ic":          (
            compute_rank_ic(w_exec, asset_returns)
            if (w_exec is not None and asset_returns is not None) else float("nan")
        ),
    }

    return {
        **primary,
        **secondary,
        **tertiary,
        "n_oos_weeks": len(excess_returns),
    }
