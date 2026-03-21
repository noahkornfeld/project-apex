# Recommendations to Improve Sharpe/Sortino Ratios

**Current Performance:**
- Portfolio Sharpe: 0.95 vs QQQ: 1.01 (worse by -0.06)
- Portfolio generates +4.89% alpha but with higher volatility
- Goal: Increase Sharpe to >1.2 (beat QQQ by meaningful margin)

---

## 🎯 HIGH IMPACT (Implement First)

### 1. **Volatility Targeting** ⭐ HIGHEST PRIORITY
**Problem:** Portfolio takes on inconsistent risk across market regimes.

**Solution:** Scale positions based on recent realized volatility.

**Implementation:**
- Created `environment/volatility_scaler.py`
- Integrate into `trading_env.py` step() method
- Target 15% annualized vol, scale weights inversely with realized vol
- Reduce exposure during high-vol periods (2008, 2020), increase during calm periods

**Expected Impact:** +0.2-0.3 Sharpe by reducing drawdowns in crisis periods

**Code changes needed:**
```python
# In trading_env.py __init__:
from environment.volatility_scaler import VolatilityScaler
self._vol_scaler = VolatilityScaler(target_vol=0.15, lookback_weeks=20)

# In trading_env.py step(), after getting w_exec from projector:
w_exec_scaled = self._vol_scaler.scale_weights(w_exec)
# Update vol scaler with realized return
self._vol_scaler.update(realized_return)
```

---

### 2. **Increase Entropy Regularization** ⭐
**Problem:** Policy may be too deterministic, leading to concentrated bets.

**Current:** `entropy_scale_factor: 1.0`
**Recommended:** `entropy_scale_factor: 1.5`

**Rationale:** Higher entropy → more diversified portfolios → lower volatility

**Expected Impact:** +0.1-0.2 Sharpe by reducing concentration risk

**Code change:**
```yaml
# config/master_config.yaml
sac:
  entropy_scale_factor: 1.5  # Increase from 1.0
```

---

### 3. **Add Drawdown Penalty to Reward Function**
**Problem:** Current reward only penalizes turnover, not drawdowns.

**Solution:** Add drawdown penalty term to reward function.

**Implementation:**
```python
# In environment/reward_fn.py
def compute_reward_with_dd_penalty(
    r_t: float,
    turnover_t: float,
    nav_current: float,
    nav_peak: float,
    lambda_turn: float = 0.01,
    lambda_dd: float = 0.5,
) -> float:
    """
    Reward with drawdown penalty:
    R_t = r_t - λ_turn·turnover_t - λ_dd·max(0, (peak - current)/peak)
    """
    base_reward = r_t - lambda_turn * turnover_t
    
    # Drawdown penalty
    if nav_peak > 0:
        drawdown = max(0.0, (nav_peak - nav_current) / nav_peak)
        dd_penalty = lambda_dd * drawdown
    else:
        dd_penalty = 0.0
    
    return base_reward - dd_penalty
```

**Expected Impact:** +0.15-0.25 Sharpe by explicitly penalizing drawdowns

---

## 📊 MEDIUM IMPACT (Implement Second)

### 4. **Increase Training Intensity**
**Current:** `updates_per_step: 20`
**Recommended:** `updates_per_step: 40`

**Rationale:** More gradient updates → better policy convergence → higher returns

**Trade-off:** 2x longer training time (~36 hours vs 18 hours)

**Expected Impact:** +0.05-0.10 Sharpe from better-trained policy

---

### 5. **Add Maximum Drawdown Constraint**
**Problem:** No hard constraint on portfolio drawdown.

**Solution:** Add max drawdown constraint to projector.

**Implementation:**
```python
# In constraint_projector.py, add new constraint:
# If current drawdown > 20%, force equal-weight or reduce exposure
if current_dd > 0.20:
    w_exec = w_exec * 0.5  # Cut exposure in half during deep drawdowns
```

**Expected Impact:** +0.1-0.15 Sharpe by capping tail risk

---

### 6. **Reduce Critic Learning Rate Further**
**Current:** `critic_lr: 0.00005`
**Recommended:** `critic_lr: 0.00003`

**Rationale:** Slower critic updates → more stable Q-values → less policy oscillation

**Expected Impact:** +0.05-0.08 Sharpe from more stable training

---

## 🔧 LOW IMPACT (Nice to Have)

### 7. **Add Sector Diversification Constraint**
**Current:** `sector_cap: 0.50` (50% max per sector)
**Recommended:** `sector_cap: 0.35` (35% max per sector)

**Rationale:** Force more diversification across sectors

**Expected Impact:** +0.03-0.05 Sharpe from reduced sector concentration

---

### 8. **Increase Tau (Target Network Update Rate)**
**Current:** `tau: 0.01`
**Recommended:** `tau: 0.015`

**Rationale:** Faster target network updates → faster adaptation to new data

**Expected Impact:** +0.02-0.04 Sharpe

---

### 9. **Add Position Sizing Based on Confidence**
**Problem:** All positions sized equally (after constraints).

**Solution:** Scale positions by model confidence (entropy of action distribution).

**Expected Impact:** +0.05-0.08 Sharpe from better position sizing

---

## 📋 IMPLEMENTATION PRIORITY

**Phase 1 (Do Now):**
1. Volatility Targeting (biggest impact)
2. Increase entropy_scale_factor to 1.5
3. Add drawdown penalty to reward

**Phase 2 (After Phase 1 results):**
4. Increase updates_per_step to 40
5. Add max drawdown constraint
6. Reduce critic_lr to 0.00003

**Phase 3 (Polish):**
7. Tighten sector_cap to 0.35
8. Increase tau to 0.015
9. Add confidence-based position sizing

---

## 🎯 EXPECTED RESULTS

**Conservative Estimate:**
- Phase 1: Portfolio Sharpe 0.95 → 1.15 (+0.20)
- Phase 2: Portfolio Sharpe 1.15 → 1.30 (+0.15)
- Phase 3: Portfolio Sharpe 1.30 → 1.40 (+0.10)

**Target:** Portfolio Sharpe > 1.30, beating QQQ (1.01) by 30%

---

## 🚀 QUICK START

To implement Phase 1 immediately:

1. **Volatility Targeting:**
   ```bash
   # Already created: environment/volatility_scaler.py
   # Need to integrate into trading_env.py
   ```

2. **Config Changes:**
   ```yaml
   # config/master_config.yaml
   sac:
     entropy_scale_factor: 1.5  # Up from 1.0
   ```

3. **Reward Function:**
   ```bash
   # Modify environment/reward_fn.py to add drawdown penalty
   ```

4. **Run new training:**
   ```bash
   python run_full_apex.py --run-name phase1_improvements
   ```

Expected training time: ~18-20 hours for 8 folds
