"""Re-compute OOS metrics with the three fixes applied to existing CSV data."""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')
from evaluation.metrics import compute_all_metrics

print(f"{'Fold':<6} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>9} {'MaxDD':>8} {'Beta':>8} {'CostDrag':>10}  Period")
print("-" * 80)

PERIODS = {
    1: "2010-11", 2: "2012-13", 3: "2014-15", 4: "2016-17",
    5: "2018-19", 6: "2020-21", 7: "2022-23", 8: "2024-now",
}

for fold in range(1, 9):
    df = pd.read_csv(f"results/fold_{fold}/oos_returns.csv")
    pr = df["portfolio_return"].values.astype(float)
    qr = df["qqq_return"].values.astype(float)

    # Apply Fix 1: clip individual weekly returns to [-0.99, 1.0]
    pr_clipped = np.clip(pr, -0.99, 1.0)

    excess = pr_clipped - qr
    nav    = np.cumprod(1.0 + pr_clipped)

    # Load cost_bps from metrics json (already recorded)
    import json, os
    mpath = f"results/fold_{fold}/oos_metrics.json"
    cost_bps_total = None
    if os.path.exists(mpath):
        m = json.load(open(mpath))
        # Rough estimate: re-derive cost_bps from cost_drag and old gross_pnl
        # (We don't have per-week cost_bps stored; use total from old metric)
        # Just show metrics without cost_drag for now
        pass

    m = compute_all_metrics(
        nav               = nav,
        excess_returns    = excess,
        qqq_returns       = qr,
        portfolio_returns = pr_clipped,
    )

    cagr    = m.get("excess_cagr", float("nan"))
    sharpe  = m.get("sharpe",      float("nan"))
    sortino = m.get("sortino",     float("nan"))
    maxdd   = m.get("max_drawdown",float("nan"))
    beta    = m.get("beta_to_qqq", float("nan"))

    sortino_str = f"{sortino:9.1f}" if abs(sortino) < 1e6 else "     inf"
    print(f"{fold:<6} {cagr:>8.2%} {sharpe:>8.2f} {sortino_str} {maxdd:>8.2%} {beta:>8.3f}  {PERIODS[fold]}")

print()
print("NOTE: cost_drag requires per-week cost_bps (not in CSV) — needs a re-run to see corrected values.")
print("      After Fix 1, NAV will no longer compound to extreme levels, so cost_t stays bounded.")
