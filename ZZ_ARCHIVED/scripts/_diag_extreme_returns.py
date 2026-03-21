"""Diagnostic: find root cause of extreme OOS portfolio returns."""
import numpy as np
import pandas as pd
from pathlib import Path
from environment.market_data import build_market_data
from config.config_loader import load_config

cfg = load_config("config/master_config.yaml")
data_dir = Path(cfg.data.data_dir)
shared_path = str(data_dir / "panels_v2" / "shared" / "feature_panel_shared.npz")

# ── Check each fold's extreme-return week ──────────────────────────────────
CASES = [
    ("fold_1", "2010-01-01", "2011-12-31", "2011-01-03"),   # +371%
    ("fold_2", "2012-01-01", "2013-12-31", "2012-12-31"),   # +1851%
    ("fold_5", "2018-01-01", "2019-12-31", "2018-12-24"),   # +9005%
]

for fold_name, test_start, test_end, bad_date in CASES:
    print(f"\n{'='*60}")
    print(f"{fold_name}  bad date: {bad_date}")
    print("="*60)

    md = build_market_data(
        bars_path         = str(data_dir / cfg.data.daily_bars_file),
        macro_path        = str(data_dir / cfg.data.macro_features_file),
        cal_path          = str(data_dir / cfg.data.trading_calendar_file),
        ndx_path          = str(data_dir / cfg.data.ndx_membership_file),
        shared_panel_path = shared_path,
        fold_train_start  = test_start,
        fold_train_end    = test_end,
    )

    adj_close  = md["adj_close"]   # [T_full, K_max]
    weekly_idx = md["weekly_idx"]  # OOS week indices into full panel
    dates_str  = md["dates_str"]   # full panel dates

    # Find the panel row for the bad date
    matches = np.where(dates_str == bad_date)[0]
    if len(matches) == 0:
        print(f"  Date {bad_date} not in panel — skipping")
        continue
    p_bad = int(matches[0])

    # Find the two consecutive weekly steps that cover bad_date
    found = False
    for i in range(len(weekly_idx) - 1):
        p_prev = int(weekly_idx[i])
        p_cur  = int(weekly_idx[i + 1])
        if p_prev <= p_bad <= p_cur:
            d_prev = dates_str[p_prev]
            d_cur  = dates_str[p_cur]
            print(f"  Covers step: weekly_idx[{i}]={p_prev} ({d_prev})  ->  weekly_idx[{i+1}]={p_cur} ({d_cur})")
            c_prev = adj_close[p_prev]
            c_cur  = adj_close[p_cur]
            ret = np.where(c_prev > 1e-8, c_cur / (c_prev + 1e-8) - 1.0, 0.0)
            print(f"  Max individual asset return: {ret.max():.4f}")
            print(f"  Min individual asset return: {ret.min():.4f}")
            big_slots = np.where(ret > 0.5)[0]
            print(f"  Slots with >50% ret: {len(big_slots)}")
            for s in big_slots[:10]:
                print(f"    slot {s}: c_prev={c_prev[s]:.6f}  c_cur={c_cur[s]:.6f}  ret={ret[s]:.2f}")
            found = True
            break

    if not found:
        # bad_date is in weekly_idx itself — check the step starting from it
        if p_bad in weekly_idx.tolist():
            pos = weekly_idx.tolist().index(p_bad)
            if pos + 1 < len(weekly_idx):
                p_prev = int(weekly_idx[pos])
                p_cur  = int(weekly_idx[pos + 1])
                d_prev = dates_str[p_prev]
                d_cur  = dates_str[p_cur]
                print(f"  Step: weekly_idx[{pos}]={p_prev} ({d_prev})  ->  weekly_idx[{pos+1}]={p_cur} ({d_cur})")
                c_prev = adj_close[p_prev]
                c_cur  = adj_close[p_cur]
                ret = np.where(c_prev > 1e-8, c_cur / (c_prev + 1e-8) - 1.0, 0.0)
                print(f"  Max individual asset return: {ret.max():.4f}")
                big_slots = np.where(ret > 0.5)[0]
                for s in big_slots[:10]:
                    print(f"    slot {s}: c_prev={c_prev[s]:.6f}  c_cur={c_cur[s]:.6f}  ret={ret[s]:.2f}")

# Also: check if weights from the OOS eval ever exceed 1.0 (leverage check)
print("\n\n=== Weight sum check (reading w_exec from OOS results via npz if available) ===")
print("(Checking oos_returns.csv to infer max theoretical weight needed)")
for fold in [1, 2, 3, 5]:
    df = pd.read_csv(f"results/fold_{fold}/oos_returns.csv")
    pr = df["portfolio_return"].values
    qr = df["qqq_return"].values
    # If port_ret > 1.0, it requires at least one stock with that return weighted at 100%
    # OR leverage. Check if QQQ or any stock in NDX had such returns.
    extreme = pr[pr > 0.5]
    print(f"  Fold {fold}: {len(extreme)} weeks with port_ret > 50%: max={pr.max():.2f}")
