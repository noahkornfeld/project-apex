"""Show before/after metrics with Fix 1 (ret_asset clip) applied retroactively."""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import json
import os
from evaluation.metrics import (
    compute_excess_cagr, compute_sharpe, compute_sortino,
    compute_max_drawdown, compute_beta_to_qqq, compute_cost_drag,
)

PERIODS = {
    1: "2010-11", 2: "2012-13", 3: "2014-15", 4: "2016-17",
    5: "2018-19", 6: "2020-21", 7: "2022-23", 8: "2024-now",
}

WEEKS_PER_YEAR = 52.0
CLIP_LO, CLIP_HI = -0.99, 1.0   # Fix 1 bounds (per-asset; here applied at portfolio level as proxy)

def compute_nav_from_returns(rets):
    return np.cumprod(1.0 + np.clip(rets, -0.9999, None))

def metrics_from_arrays(pr, qr):
    nav    = compute_nav_from_returns(pr)
    excess = pr - qr
    n      = len(pr)
    cagr   = compute_excess_cagr(excess)
    sharpe = compute_sharpe(excess)
    sort_  = compute_sortino(excess)
    mdd    = compute_max_drawdown(nav)
    beta   = compute_beta_to_qqq(pr, qr)
    return dict(cagr=cagr, sharpe=sharpe, sortino=sort_, mdd=mdd, beta=beta,
                n=n, max_pr=pr.max(), min_pr=pr.min(), mean_pr=pr.mean(),
                n_extreme=int((np.abs(pr) > 0.20).sum()))

print("=" * 95)
print(f"{'Fold':<5}  {'Period':<9}  {'CAGR':>9}  {'Sharpe':>7}  {'Sortino':>8}  {'MaxDD':>8}  {'Beta':>6}  {'Extreme':>8}  Label")
print("-" * 95)

summaries = []
for fold in range(1, 9):
    csv  = f"results/fold_{fold}/oos_returns.csv"
    jpath = f"results/fold_{fold}/oos_metrics.json"
    if not os.path.exists(csv):
        continue
    df = pd.read_csv(csv)
    pr = df["portfolio_return"].values.astype(float)
    qr = df["qqq_return"].values.astype(float)

    # --- RAW (pre-fix) ---
    m_raw = metrics_from_arrays(pr, qr)

    # --- CORRECTED (Fix 1 clip applied) ---
    pr_fix = np.clip(pr, CLIP_LO, CLIP_HI)
    m_fix  = metrics_from_arrays(pr_fix, qr)

    period = PERIODS[fold]
    n_clipped = int((pr != pr_fix).sum())

    def fmt_sortino(v):
        if np.isinf(v) or abs(v) > 999: return "     inf"
        return f"{v:8.2f}"

    print(f"{fold:<5}  {period:<9}  {m_raw['cagr']:>8.1%}  {m_raw['sharpe']:>7.2f}  {fmt_sortino(m_raw['sortino'])}  "
          f"{m_raw['mdd']:>8.2%}  {m_raw['beta']:>6.2f}  {m_raw['n_extreme']:>5} wks   RAW")
    print(f"{'':5}  {'':9}  {m_fix['cagr']:>8.1%}  {m_fix['sharpe']:>7.2f}  {fmt_sortino(m_fix['sortino'])}  "
          f"{m_fix['mdd']:>8.2%}  {m_fix['beta']:>6.2f}  {n_clipped:>5} clpd   FIXED")
    print()
    summaries.append(m_fix)

# Cross-fold averages (fixed)
cagrs   = [m['cagr']   for m in summaries]
sharpes = [m['sharpe'] for m in summaries]
mdds    = [m['mdd']    for m in summaries]
betas   = [m['beta']   for m in summaries]
print("-" * 95)
print(f"{'Avg':<5}  {'(fixed)':9}  {np.mean(cagrs):>8.1%}  {np.mean(sharpes):>7.2f}  {'':8}  "
      f"{np.mean(mdds):>8.2%}  {np.mean(betas):>6.2f}  {'':>8}  FIXED AVG")
print("=" * 95)
print("\nNOTE: 'Extreme' = weeks where |portfolio_return| > 20%. These are the data-quality spikes.")
print("      After re-running on WSL with the fix, these weeks will be bounded at the asset level,")
print("      producing cleaner NAV paths and more reliable aggregate statistics.")
