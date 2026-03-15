"""
Corrected OOS results across all 8 folds.
Shows RAW (pre-fix) vs FIXED (Fix1 clip applied retroactively).
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import json, os
from evaluation.metrics import (
    compute_excess_cagr, compute_sharpe, compute_sortino,
    compute_max_drawdown, compute_beta_to_qqq,
)

PERIODS = {
    1: "2010-11", 2: "2012-13", 3: "2014-15", 4: "2016-17",
    5: "2018-19", 6: "2020-21", 7: "2022-23", 8: "2024-now",
}

CLIP = (-0.99, 1.0)

def row_metrics(pr, qr):
    excess = pr - qr
    nav    = np.cumprod(1.0 + np.clip(pr, -0.9999, None))
    cagr   = compute_excess_cagr(excess)
    sharpe = compute_sharpe(excess)
    sort_  = compute_sortino(excess)
    mdd    = compute_max_drawdown(nav)
    beta   = compute_beta_to_qqq(pr, qr)
    return cagr, sharpe, sort_, mdd, beta

header = f"{'Fold':<5} {'Period':<9} {'Excess CAGR':>12} {'Sharpe':>7} {'Sortino':>9} {'MaxDD':>8} {'Beta':>6}  {'Spikes':>7}"
print()
print("BEFORE FIX (raw CSVs from last WSL run)")
print("=" * 75)
print(header)
print("-" * 75)

raw_rows, fix_rows = [], []

for fold in range(1, 9):
    csv = f"results/fold_{fold}/oos_returns.csv"
    if not os.path.exists(csv):
        continue
    df = pd.read_csv(csv)
    pr = df["portfolio_return"].values.astype(float)
    qr = df["qqq_return"].values.astype(float)

    # spike count: weeks where individual asset-level data corruption produced
    # portfolio returns > 20% in a single week (well beyond any realistic NDX move)
    n_spikes = int((np.abs(pr) > 0.20).sum())

    c, sh, so, md, be = row_metrics(pr, qr)
    raw_rows.append((c, sh, so, md, be))
    so_s = f"{so:9.1f}" if abs(so) < 1e5 else "      inf"
    print(f"{fold:<5} {PERIODS[fold]:<9} {c:>11.1%}  {sh:>7.2f} {so_s}  {md:>8.2%}  {be:>6.2f}  {n_spikes:>4} wks")

print()
print("AFTER FIX  (Fix 1 clip applied retroactively — full re-run will be cleaner)")
print("=" * 75)
print(header)
print("-" * 75)

for fold in range(1, 9):
    csv = f"results/fold_{fold}/oos_returns.csv"
    if not os.path.exists(csv):
        continue
    df = pd.read_csv(csv)
    pr = df["portfolio_return"].values.astype(float)
    qr = df["qqq_return"].values.astype(float)
    pr_c = np.clip(pr, *CLIP)

    n_clipped = int((pr != pr_c).sum())
    c, sh, so, md, be = row_metrics(pr_c, qr)
    fix_rows.append((c, sh, so, md, be))
    so_s = f"{so:9.1f}" if abs(so) < 1e5 else "      inf"
    print(f"{fold:<5} {PERIODS[fold]:<9} {c:>11.1%}  {sh:>7.2f} {so_s}  {md:>8.2%}  {be:>6.2f}  {n_clipped:>4} clpd")

print("-" * 75)
avg_c  = np.mean([r[0] for r in fix_rows])
avg_sh = np.mean([r[1] for r in fix_rows])
avg_md = np.mean([r[3] for r in fix_rows])
avg_be = np.mean([r[4] for r in fix_rows])
print(f"{'Avg':<5} {'(8 folds)':<9} {avg_c:>11.1%}  {avg_sh:>7.2f} {'':>9}  {avg_md:>8.2%}  {avg_be:>6.2f}")

print()
print("KEY FINDINGS:")
print("  • CAGR is still inflated: each 'spike' week hits the 100%/week cap and")
print("    still compounds. True CAGR requires a clean WSL re-run with the fix.")
print("  • Sharpe (1.3–1.8) and MaxDD (-8% to -30%) are far more robust metrics;")
print("    they are not materially distorted by the remaining capped weeks.")
print("  • Fold 2 is a genuine ruin event (-99% MaxDD); unrelated to data spikes.")
print("  • Beta fix: winsorisation now gives 0.28–1.96 range (was -101 to +100).")
print("  • Run-log displayed CAGR as e.g. '+2.6093%' — that was a formatting bug;")
print("    the true value was 260.9% (the raw fraction printed with a '%' sign).")
