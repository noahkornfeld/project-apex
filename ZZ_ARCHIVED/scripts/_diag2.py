"""Diagnose root cause of extreme portfolio returns."""
import numpy as np

shared = np.load('Ticker_Data/panels_v2/shared/feature_panel_shared.npz', allow_pickle=True)
dates  = shared['dates'].astype(str)

d5   = np.load('Ticker_Data/panels_v2/fold_5/feature_panel.npz', allow_pickle=True)
x    = d5['x_panel']    # [5577, 110, 25]
mask = d5['mask_panel'] # [5577, 110]

print("x_panel shape:", x.shape)
print("Any NaN:", np.isnan(x).any())
print("Any Inf:", np.isinf(x).any())
print("Max abs:", float(np.abs(x).max()))

# Extreme dates for fold 5
p_dec17 = int(np.where(dates == '2018-12-17')[0][0])
p_dec24 = int(np.where(dates == '2018-12-24')[0][0])

for p, label in [(p_dec17, 'Dec-17'), (p_dec24, 'Dec-24')]:
    active = mask[p] > 0.5
    xrow   = x[p][active]   # [n_active, 25]
    print(f"\n{label} (row {p}): {active.sum()} active slots")
    print(f"  feat max={xrow.max():.3f}  min={xrow.min():.3f}")
    big = np.abs(xrow) > 10
    if big.any():
        print(f"  {big.sum()} (slot,feat) entries with |val|>10:")
        for idx in np.argwhere(big)[:10]:
            print(f"    slot={idx[0]}  feat={idx[1]}  val={xrow[idx[0], idx[1]]:.4f}")

# Also check the raw adj_close-equivalent: ret_1w is feature index 4 (log_ret=3, ret_1w=4)
# Let's find what feature index corresponds to ret_1w
# Feature order from features/__init__.py: open,close,volume,log_ret,ret_1w,ret_4w,ret_13w,...
FEAT_NAMES = [
    "open","close","volume","log_ret","ret_1w","ret_4w","ret_13w",
    "vol_4w","vol_13w","vol_52w","vol_z","beta_qqq","rs_qqq",
    "rsi_14","bb_pos","bb_width","ret_z_1w","ret_z_4w","ret_z_13w",
    "ret_rank_1w","ret_rank_4w","cs_beta","cs_vol","vix_z","vix_chg"
]
print("\nFeature index map:", {n: i for i, n in enumerate(FEAT_NAMES)})

# Check ret_1w (index 4) around Christmas 2018
feat_idx = 4  # ret_1w
print(f"\nret_1w (feat {feat_idx}) values at Dec-17 2018 (active slots):")
active = mask[p_dec17] > 0.5
vals = x[p_dec17, active, feat_idx]
print(f"  mean={vals.mean():.3f}  std={vals.std():.3f}  max={vals.max():.3f}  min={vals.min():.3f}")
extreme = np.abs(vals) > 5
if extreme.any():
    print(f"  {extreme.sum()} slots with |ret_1w|>5 sigma:")
    for i in np.where(extreme)[0][:5]:
        slot_idx = np.where(active)[0][i]
        print(f"    active_slot={i} panel_slot={slot_idx} ret_1w={vals[i]:.4f}")
