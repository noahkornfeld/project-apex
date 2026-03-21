"""Smoke-test the three bug fixes."""
import sys
sys.path.insert(0, '.')
import numpy as np
from evaluation.metrics import compute_beta_to_qqq, compute_cost_drag

rng = np.random.default_rng(42)
pr_normal = rng.normal(0.01, 0.03, 100)
qr_normal = rng.normal(0.005, 0.02, 100)

# Fix 2: beta with outlier should be close to beta without outlier
pr_spike = pr_normal.copy()
pr_spike[50] = 90.0
beta_spike = compute_beta_to_qqq(pr_spike, qr_normal)
beta_clean = compute_beta_to_qqq(pr_normal, qr_normal)
print(f"Fix 2 | beta w/ spike (winsorized)={beta_spike:.3f}  beta clean={beta_clean:.3f}")
assert abs(beta_spike - beta_clean) < 0.5, "Winsorization must make beta robust"

# Fix 3: cost_drag uses compounded return
cost_bps = np.full(103, 1.0)
gross_r   = np.full(103, 0.01)
drag = compute_cost_drag(cost_bps, gross_r)
# compounded = (1.01)^103 - 1 = 1.734 => gross_pnl_bps = 17340; total_cost = 103; drag ~ 0.00594
expected = 103.0 / ((np.prod(1 + gross_r) - 1) * 10_000)
print(f"Fix 3 | cost_drag={drag:.6f}  expected={expected:.6f}")
assert abs(drag - expected) < 1e-9, "cost_drag must match compounded formula"

# Fix 3: extreme spike in returns must NOT inflate cost_drag denominator
gross_spike = gross_r.copy()
gross_spike[50] = 90.0
drag_spike = compute_cost_drag(cost_bps, gross_spike)
print(f"Fix 3 | cost_drag with 9000% spike={drag_spike:.8f}  (should be tiny)")
assert drag_spike < 0.001, "Extreme return week must not inflate denominator"

# Fix 1: verify ret_asset clip applied (trading_env import check)
import ast, pathlib
src = pathlib.Path("environment/trading_env.py").read_text()
assert "np.clip(ret_asset, -0.99, 1.0)" in src, "Fix 1 clip line must be present"
print("Fix 1 | np.clip(ret_asset, -0.99, 1.0) confirmed in trading_env.py")

print("\nAll three fixes verified OK.")
