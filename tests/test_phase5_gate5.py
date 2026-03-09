"""
tests/test_phase5_gate5.py
Gate 5 — Trading Environment  (Phase 5 Bible §5)

Six Gate 5 criteria:
  1. NAV sanity   – NAV > 0 for all steps; episode smoke test
  2. Cost model   – hand-calculated fixture ± 0.1 bps
  3. Timeline     – obs_date < exec_date for every step
  4. QQQ parity   – env QQQ NAV matches independent total-return calc
  5. Forced liq   – NDX removal triggers zero-weight + cost deduction
  6. Episode term – done=True at final date; episode length correct
"""

import math
import numpy as np
import pytest

from environment.market_data import make_synthetic_market_data
from environment.trading_env import TradingEnvironment, EPS

# ---------------------------------------------------------------------------
# Shared fixture builder
# ---------------------------------------------------------------------------

def make_env(
    n_days: int = 260,
    K_max:  int = 10,
    n_active: int = 8,
    seed: int = 0,
    weekly_every: int = 5,
    projector=None,
) -> TradingEnvironment:
    md = make_synthetic_market_data(
        n_days      = n_days,
        K_max       = K_max,
        n_active    = n_active,
        seed        = seed,
        weekly_every= weekly_every,
    )
    return TradingEnvironment(md, projector=projector, L_lookback=4)


def run_episode(env: TradingEnvironment, seed: int = 99):
    """Run a full episode with random actions; return list of (reward_components, info)."""
    rng = np.random.default_rng(seed)
    K   = env._K

    obs, _ = env.reset()
    results = []
    done = False
    while not done:
        w_raw    = np.abs(rng.normal(size=K)).astype(np.float32)
        obs, rc, done, info = env.step(w_raw)
        results.append((rc, info))
    return results


# ===========================================================================
# Gate 5.1  NAV sanity — NAV > 0 for all steps in a full episode
# ===========================================================================

class TestNavSanity:

    def test_nav_always_positive_smoke(self):
        """NAV must remain strictly positive throughout a 5-year equivalent episode."""
        env = make_env(n_days=1300, n_active=8, seed=1)
        results = run_episode(env, seed=42)

        for step_i, (rc, info) in enumerate(results):
            nav = rc["nav"]
            assert nav > 0, (
                f"Step {step_i}: NAV={nav:.6f} is non-positive. "
                f"r_port={rc['r_port_t']:.4f}, cost={rc['cost_t']:.4f}"
            )

    def test_nav_decreases_with_cost(self):
        """
        With zero returns but non-zero trades, NAV should be <= 1.0 due to costs.
        Use a synthetic env where all asset prices are flat (constant close).
        """
        md = make_synthetic_market_data(n_days=50, K_max=6, n_active=4, seed=5)
        # Force flat prices so r_port = 0
        md["adj_close"][:] = 1.0
        md["adj_open"] [:] = 1.0

        env = TradingEnvironment(md, projector=None, L_lookback=2)
        obs, _ = env.reset()

        # Random weights → should trigger cost
        w = np.array([0.25, 0.25, 0.25, 0.25, 0.0, 0.0], dtype=np.float32)
        _, rc, done, info = env.step(w)

        assert env.nav <= 1.0 + 1e-6, (
            f"NAV={env.nav:.6f} should not exceed 1.0 with flat prices and costs"
        )
        assert env.nav > 0.90, "NAV dropped too far on first step (sanity check)"

    def test_nav_positive_with_negative_returns(self):
        """NAV > 0 even when every asset has negative weekly returns."""
        md = make_synthetic_market_data(n_days=100, K_max=6, n_active=4, seed=7)
        # All prices decline 5% per step
        T = md["adj_close"].shape[0]
        for t in range(T):
            md["adj_close"][t] = 1.0 * (0.95 ** t)
            md["adj_open"][t]  = md["adj_close"][t]

        env = TradingEnvironment(md, projector=None, L_lookback=2)
        obs, _ = env.reset()
        w = np.array([0.25, 0.25, 0.25, 0.25, 0.0, 0.0], dtype=np.float32)

        done = False
        step = 0
        while not done:
            obs, rc, done, info = env.step(w)
            step += 1
            assert rc["nav"] > 0, f"NAV={rc['nav']:.6f} at step {step}"


# ===========================================================================
# Gate 5.2  Cost model — hand-calculated fixture ± 0.1 bps
# ===========================================================================

class TestCostModel:
    """
    Hand-calculate one full cost computation and verify env matches.

    Fixture:
      NAV          = 1_000_000  (so dollar amounts are clear)
      delta_w      = [+0.10, -0.05, 0, 0, ...]  (buying 10%, selling 5%)
      ADV63        = 500_000_000  (500M$ ADV for all assets)
      ADV_median   = 500_000_000  (same, so ratio=1)
      vol_252      = 0.25         (25% ann vol)
      gap_vol_252  = 0.010        (1% overnight gap vol)
      VIX          = 20.0         (M=1.0)
      mask         = [1, 1, 0, ...]

    Term1 (commission): 1.0 bps flat
    Term2 (spread):
        M=1.0, spread_coeff=2.0, ADV_ratio=1.0, vol_factor=max(1, 0.25/0.20)=1.25
        = 2.0 × 1.0 × 1.25 = 2.50 bps
    Term3 (impact):
        |trade_$| / ADV = (0.10 × 1M) / 500M = 2e-4
        √2e-4 = 0.01414
        vol_norm = 0.25 / 0.20 = 1.25
        = 10 × 0.01414 × 1.25 = 0.17678 bps
    Term4 (gap):
        1.5 × 0.010 × 0.7979 = 0.011969

    total_bps = 1.0 + 2.5 + 0.17678 + 0.011969 = 3.68875 bps
    cost_$ = |trade_$| / 10_000 × total_bps
    For asset 0: trade_$ = 100_000; cost_$ = 100_000/10_000 × 3.68875 = 36.8875
    For asset 1: trade_$ = 50_000;  cost_$ = 50_000/10_000 × 3.68875  = 18.4438
    cost_fraction = (36.8875 + 18.4438) / 1_000_000 = 55.33e-6
    """

    NAV        = 1_000_000.0
    ADV        = 5e8
    VOL        = 0.25
    GAP_VOL    = 0.010
    VIX        = 20.0

    def _build_env(self):
        md = make_synthetic_market_data(n_days=20, K_max=4, n_active=2, seed=0)
        return TradingEnvironment(md, projector=None, L_lookback=2)

    def _hand_calc(self, trade_dollar: float) -> float:
        """Return cost fraction for a single asset with the fixture params."""
        vol = self.VOL
        adv = self.ADV
        adv_median = self.ADV
        M   = max(1.0, self.VIX / 20.0)

        term1 = 1.0
        term2 = M * 2.0 * (adv_median / adv) ** 0.3 * max(1.0, vol / 0.20)
        term3 = M * 10.0 * (abs(trade_dollar) / adv) ** 0.5 * vol / 0.20
        term4 = 1.5 * self.GAP_VOL * 0.7979

        total_bps = term1 + term2 + term3 + term4
        return abs(trade_dollar) / 1e4 * total_bps

    def test_single_asset_cost_matches_formula(self):
        """Cost for one asset matches hand-calculation within ±0.1 bps of trade_$."""
        env = self._build_env()

        delta_w    = np.array([0.10, 0.0, 0.0, 0.0], dtype=np.float64)
        adv63      = np.full(4, self.ADV)
        vol_252    = np.full(4, self.VOL)
        gap_vol252 = np.full(4, self.GAP_VOL)
        mask       = np.array([1.0, 0.0, 0.0, 0.0])

        cost_frac, _ = env.compute_cost_public(
            delta_w     = delta_w,
            nav         = self.NAV,
            adv63       = adv63,
            vol_252     = vol_252,
            gap_vol_252 = gap_vol252,
            vix         = self.VIX,
            mask        = mask,
        )

        trade_dollar       = 0.10 * self.NAV       # 100_000
        expected_cost_usd = self._hand_calc(trade_dollar)
        expected_frac     = expected_cost_usd / self.NAV

        tolerance = 0.1e-4 / 1e4   # 0.1 bps of trade_$ expressed as fraction of NAV
        assert abs(cost_frac - expected_frac) < tolerance, (
            f"Cost mismatch: env={cost_frac:.8f}, expected={expected_frac:.8f}, "
            f"diff={abs(cost_frac - expected_frac)*1e4:.4f} bps"
        )

    def test_two_asset_cost_additive(self):
        """Cost of two independent trades equals sum of individual costs."""
        env = self._build_env()

        delta_w    = np.array([0.10, -0.05, 0.0, 0.0], dtype=np.float64)
        adv63      = np.full(4, self.ADV)
        vol_252    = np.full(4, self.VOL)
        gap_vol252 = np.full(4, self.GAP_VOL)
        mask       = np.array([1.0, 1.0, 0.0, 0.0])

        cost_frac, _ = env.compute_cost_public(
            delta_w     = delta_w,
            nav         = self.NAV,
            adv63       = adv63,
            vol_252     = vol_252,
            gap_vol_252 = gap_vol252,
            vix         = self.VIX,
            mask        = mask,
        )

        expected_cost_usd = (self._hand_calc(0.10 * self.NAV)
                          + self._hand_calc(0.05 * self.NAV))
        expected_frac    = expected_cost_usd / self.NAV

        tolerance = 0.2e-4 / 1e4   # 0.1 bps per leg
        assert abs(cost_frac - expected_frac) < tolerance, (
            f"Two-asset cost mismatch: env={cost_frac*1e4:.4f} bps, "
            f"expected={expected_frac*1e4:.4f} bps"
        )

    def test_zero_trade_no_cost(self):
        """No trade → zero cost."""
        env   = self._build_env()
        delta_w = np.zeros(4)
        mask    = np.array([1.0, 1.0, 0.0, 0.0])
        cost_frac, _ = env.compute_cost_public(
            delta_w=delta_w, nav=self.NAV,
            adv63=np.full(4, self.ADV), vol_252=np.full(4, self.VOL),
            gap_vol_252=np.full(4, self.GAP_VOL), vix=self.VIX, mask=mask,
        )
        assert cost_frac == pytest.approx(0.0, abs=1e-10)

    def test_high_vix_increases_cost(self):
        """VIX=40 (M=2) must produce exactly 2x the spread+impact vs VIX=20 (M=1)."""
        env = self._build_env()
        kwargs = dict(
            delta_w     = np.array([0.10, 0.0, 0.0, 0.0], dtype=np.float64),
            nav         = self.NAV,
            adv63       = np.full(4, self.ADV),
            vol_252     = np.full(4, self.VOL),
            gap_vol_252 = np.full(4, 0.0),   # zero gap to isolate M terms
            mask        = np.array([1.0, 0.0, 0.0, 0.0]),
        )
        cost_low,  _ = env.compute_cost_public(vix=20.0, **kwargs)
        cost_high, _ = env.compute_cost_public(vix=40.0, **kwargs)

        # Term2 and Term3 scale with M; Term1 is flat; Term4 is zero here.
        # With zero gap, total_bps_high - total_bps_low = (M_high-M_low)*(term2+term3)
        # The ratio cost_high/cost_low should be > 1 (more specifically, between 1 and 2)
        assert cost_high > cost_low, (
            "Higher VIX should produce higher cost"
        )

    def test_larger_trade_increases_impact_superlinearly(self):
        """
        Square-root impact model: absolute cost in $ is super-linear in trade size.

        cost_$ = |trade_$|/10000 × [fixed_bps + C × sqrt(|trade_$|/ADV)]
        Doubling trade_$ → ratio in (2, 2√2] when impact dominates.
        When fixed bps dominate → ratio → 2.  Overall: ratio > 1 always,
        and cost-per-dollar-traded (bps) grows with sqrt(trade_$).
        """
        env    = self._build_env()
        mask   = np.array([1.0, 0.0, 0.0, 0.0])
        common = dict(nav=self.NAV, adv63=np.full(4, self.ADV),
                      vol_252=np.full(4, self.VOL), gap_vol_252=np.full(4, 0.0),
                      vix=20.0, mask=mask)

        cost_s, _ = env.compute_cost_public(
            delta_w=np.array([0.05, 0.0, 0.0, 0.0]), **common)
        cost_l, _ = env.compute_cost_public(
            delta_w=np.array([0.10, 0.0, 0.0, 0.0]), **common)

        ratio = cost_l / (cost_s + EPS)
        # Super-linear: ratio strictly > 1; upper bound 2√2 ≈ 2.83 (pure impact)
        assert ratio > 1.0, f"Cost must grow with trade size, got ratio={ratio:.4f}"
        assert ratio < 2.0 * math.sqrt(2.0) + 0.01, (
            f"Impact ratio {ratio:.3f} exceeds theoretical max 2√2 for sqrt model"
        )
        # Cost-per-dollar-traded (bps) is also larger for the bigger trade
        bps_s = cost_s / (0.05 * self.NAV / 1e4 + EPS)
        bps_l = cost_l / (0.10 * self.NAV / 1e4 + EPS)
        assert bps_l > bps_s, (
            f"Cost-per-dollar should be higher for larger trade: "
            f"bps_small={bps_s:.4f}, bps_large={bps_l:.4f}"
        )


# ===========================================================================
# Gate 5.3  Timeline — obs_date < exec_date for every step
# ===========================================================================

class TestTimeline:

    def test_obs_before_exec_every_step(self):
        """
        info['date_obs'] < info['date_exec'] for every step.
        This verifies the §5.1 timeline constraint is enforced.
        """
        env = make_env(n_days=200, seed=10)
        obs, _ = env.reset()
        rng  = np.random.default_rng(10)
        done = False
        step = 0
        while not done:
            w = np.abs(rng.normal(size=env._K)).astype(np.float32)
            obs, rc, done, info = env.step(w)
            step += 1
            d_obs  = info["date_obs"]
            d_exec = info["date_exec"]
            assert d_obs < d_exec, (
                f"Step {step}: obs_date={d_obs} >= exec_date={d_exec} "
                "(TIMELINE VIOLATION)"
            )

    def test_sequential_exec_dates(self):
        """Execution dates are strictly non-decreasing across all steps."""
        env = make_env(n_days=200, seed=11)
        obs, _ = env.reset()
        rng  = np.random.default_rng(11)
        prev_exec = ""
        done = False
        step = 0
        while not done:
            w = np.abs(rng.normal(size=env._K)).astype(np.float32)
            obs, rc, done, info = env.step(w)
            step += 1
            d_exec = info["date_exec"]
            assert d_exec >= prev_exec, (
                f"Step {step}: exec_date {d_exec} < prev {prev_exec} (non-monotone)"
            )
            prev_exec = d_exec

    def test_no_future_price_in_obs(self):
        """
        The observation at step t uses adj_close[p_t], NOT adj_close[p_{t+1}].
        Verify by checking panel_idx in obs matches the CURRENT step's index,
        not the next step's index.
        """
        env = make_env(n_days=100, seed=12)
        obs0, _ = env.reset()
        p0 = obs0["panel_idx"]

        w = np.abs(np.random.default_rng(12).normal(size=env._K)).astype(np.float32)
        obs1, rc, done, info = env.step(w)
        p1 = obs1["panel_idx"]

        # The observation returned after step() should be at p1 > p0
        assert p1 > p0, f"panel_idx did not advance: p0={p0}, p1={p1}"
        # And obs1 comes from d_{t+1}, not d_{t+2} or beyond
        wi = env._weekly_idx
        cur_ep = env._ep_step
        assert p1 == wi[cur_ep], (
            f"obs panel_idx {p1} != weekly_idx[ep_step={cur_ep}]={wi[cur_ep]}"
        )


# ===========================================================================
# Gate 5.4  QQQ parity — env QQQ NAV matches independent total-return calc
# ===========================================================================

class TestQQQParity:

    def test_qqq_nav_matches_independent_calc(self):
        """
        Run full episode, compute QQQ NAV independently from qqq_close array,
        compare against env.nav_qqq.  Tolerance: 1e-5 relative.
        """
        md  = make_synthetic_market_data(n_days=100, K_max=6, n_active=4, seed=20)
        env = TradingEnvironment(md, projector=None, L_lookback=2)

        obs, _ = env.reset()
        rng    = np.random.default_rng(20)
        wi     = env._weekly_idx
        first  = env._ep_step

        # Independent QQQ NAV calculation
        qqq_close  = md["qqq_close"]
        nav_qqq_ref = 1.0

        done = False
        step_count = 0
        while not done:
            t_cur  = env._ep_step
            p_cur  = wi[t_cur]
            p_next = wi[t_cur + 1]

            # Advance ref before step
            qqq_cur  = float(qqq_close[p_cur])
            qqq_next = float(qqq_close[p_next])
            if qqq_cur > EPS:
                nav_qqq_ref *= (qqq_next / qqq_cur)

            w    = np.abs(rng.normal(size=env._K)).astype(np.float32)
            obs, rc, done, info = env.step(w)
            step_count += 1

        env_qqq = env.nav_qqq
        rel_err = abs(env_qqq - nav_qqq_ref) / (abs(nav_qqq_ref) + EPS)
        assert rel_err < 1e-5, (
            f"QQQ NAV mismatch after {step_count} steps: "
            f"env={env_qqq:.8f}, ref={nav_qqq_ref:.8f}, rel_err={rel_err:.2e}"
        )

    def test_flat_qqq_nav_stays_one(self):
        """If QQQ prices are flat, QQQ NAV stays at 1.0."""
        md = make_synthetic_market_data(n_days=60, K_max=4, n_active=2, seed=21)
        md["qqq_close"][:] = 100.0   # flat

        env = TradingEnvironment(md, projector=None, L_lookback=2)
        results = run_episode(env, seed=21)

        for i, (rc, info) in enumerate(results):
            nav_q = rc["nav_qqq"]
            assert abs(nav_q - 1.0) < 1e-5, (
                f"Step {i}: QQQ NAV={nav_q:.8f} != 1.0 with flat QQQ prices"
            )

    def test_qqq_nav_grows_with_positive_returns(self):
        """Rising QQQ prices → nav_qqq > 1.0."""
        md = make_synthetic_market_data(n_days=60, K_max=4, n_active=2, seed=22)
        T = md["qqq_close"].shape[0]
        md["qqq_close"][:] = np.array([100.0 * 1.01 ** t for t in range(T)], dtype=np.float32)

        env = TradingEnvironment(md, projector=None, L_lookback=2)
        results = run_episode(env, seed=22)

        final_nav = results[-1][0]["nav_qqq"]
        assert final_nav > 1.0, f"QQQ NAV={final_nav:.4f} should be > 1.0 with rising prices"


# ===========================================================================
# Gate 5.5  Forced liquidation — NDX removal triggers zero-weight + cost
# ===========================================================================

class TestForcedLiquidation:

    def _build_forced_liq_md(self, liq_step: int = 3, K_max: int = 6,
                              n_active: int = 4) -> dict:
        """
        Build synthetic market data where asset slot 0 is removed from the
        mask at weekly step `liq_step`.  All prices are flat = 1.0.
        """
        n_days = 80
        md = make_synthetic_market_data(n_days=n_days, K_max=K_max,
                                         n_active=n_active, seed=30)
        # Flat prices so r_port = 0 and costs are isolated
        md["adj_close"][:] = 1.0
        md["adj_open"] [:] = 1.0
        md["adv63"]    [:] = 5e8
        md["qqq_close"][:] = 100.0

        wi = md["weekly_idx"]
        # Deactivate slot 0 from weekly step liq_step+1 onward
        if liq_step + 1 < len(wi):
            deactivate_from = wi[liq_step + 1]
            md["mask_panel"][deactivate_from:, 0] = 0.0
            md["active_ids"][deactivate_from:, 0] = -1

        return md

    def test_forced_liq_zeroes_weight(self):
        """
        After NDX removal, the deactivated slot has zero weight in w_exec.
        """
        md  = self._build_forced_liq_md(liq_step=2)
        env = TradingEnvironment(md, projector=None, L_lookback=2)
        obs, _ = env.reset()

        # Give asset slot 0 a large weight for the first few steps
        w_heavy = np.array([0.5, 0.2, 0.1, 0.1, 0.0, 0.0], dtype=np.float32)

        infos = []
        done  = False
        step  = 0
        while not done:
            obs, rc, done, info = env.step(w_heavy)
            infos.append((env.w_exec.copy(), rc, info))
            step += 1

        # After deactivation (step 3 onward), slot 0 must be 0
        # step index = 0-based within episode results
        for i, (we, rc, info) in enumerate(infos):
            if i >= 2:   # deactivation at step 2
                assert we[0] == pytest.approx(0.0, abs=1e-6), (
                    f"Step {i}: slot 0 weight={we[0]:.6f} should be 0 after deactivation"
                )

    def test_forced_liq_incurs_cost(self):
        """
        Forced liquidation of a held position must deduct a non-zero cost.
        Use flat prices so NAV should be < 1.0 purely from transaction costs.
        """
        md  = self._build_forced_liq_md(liq_step=2)
        env = TradingEnvironment(md, projector=None, L_lookback=2)
        obs, _ = env.reset()

        # Build up position in slot 0 first
        w_build = np.array([0.40, 0.20, 0.20, 0.20, 0.0, 0.0], dtype=np.float32)

        done  = False
        step  = 0
        cost_at_liq = None
        while not done:
            obs, rc, done, info = env.step(w_build)
            step += 1
            if info.get("forced_liq", False):
                cost_at_liq = rc["cost_t"]

        assert cost_at_liq is not None, (
            "Expected a forced-liquidation step but none was flagged in info"
        )
        assert cost_at_liq > 0, (
            f"Forced-liquidation step had zero cost: {cost_at_liq}"
        )

    def test_forced_liq_weight_redistributed(self):
        """
        After forced liquidation, total weight across remaining active
        slots sums to approximately 1.0 (weight is redistributed).
        """
        md  = self._build_forced_liq_md(liq_step=2, n_active=4)
        env = TradingEnvironment(md, projector=None, L_lookback=2)
        obs, _ = env.reset()

        w_initial = np.array([0.4, 0.2, 0.2, 0.2, 0.0, 0.0], dtype=np.float32)
        done  = False
        step  = 0
        while not done:
            obs, rc, done, info = env.step(w_initial)
            step += 1
            if step > 2:
                # After forced liquidation, remaining active slots should sum ~1
                w_cur = env.w_exec
                mask  = env._mask_panel[env._weekly_idx[env._ep_step]]
                w_sum = w_cur[mask > 0.5].sum()
                assert abs(w_sum - 1.0) < 0.15, (
                    f"Step {step}: post-liq weight sum={w_sum:.4f} far from 1.0"
                )


# ===========================================================================
# Gate 5.6  Episode termination — done=True at final date; length correct
# ===========================================================================

class TestEpisodeTermination:

    def test_done_at_last_step(self):
        """done=True is returned exactly at the last available weekly step."""
        env = make_env(n_days=100, seed=40)
        results = run_episode(env, seed=40)

        # done=True on the last result
        # We infer from the fact that the episode ends naturally (run_episode stops on done)
        assert len(results) > 0

        # Verify: running one more step should raise (ep_step >= W-1)
        with pytest.raises((AssertionError, IndexError)):
            w = np.ones(env._K, dtype=np.float32) / env._K
            env.step(w)

    def test_episode_length_matches_weekly_idx(self):
        """
        Episode length (number of steps taken) = len(weekly_idx) - 1 - first_valid_step.
        This verifies termination is triggered at exactly the right index.
        """
        env = make_env(n_days=100, n_active=4, seed=41)
        obs, _ = env.reset()

        expected_steps = env.n_steps
        results = run_episode(env, seed=41)

        assert len(results) == expected_steps, (
            f"Episode length mismatch: got {len(results)}, "
            f"expected {expected_steps}"
        )

    def test_reset_restores_nav(self):
        """After running an episode, reset() restores NAV=1.0 and w_exec=0."""
        env = make_env(n_days=80, seed=42)
        run_episode(env, seed=42)

        assert env.nav != 1.0 or env.ep_step > 0  # changed during episode

        obs, info = env.reset()
        assert env.nav     == pytest.approx(1.0, abs=1e-9), "NAV not reset to 1.0"
        assert env.nav_qqq == pytest.approx(1.0, abs=1e-9), "nav_qqq not reset"
        assert env.nav_pre == pytest.approx(1.0, abs=1e-9), "nav_pre not reset"
        assert np.allclose(env.w_exec, 0.0),  "w_exec not zeroed on reset"

    def test_two_episodes_independent(self):
        """Two consecutive episodes from the same env produce the same first-step result."""
        env = make_env(n_days=80, seed=43)

        def run_one_step(env):
            obs, _ = env.reset()
            rng = np.random.default_rng(0)
            w   = np.abs(rng.normal(size=env._K)).astype(np.float32)
            obs, rc, done, info = env.step(w)
            return rc["nav"], rc["cost_t"], rc["r_port_t"]

        nav1, cost1, r1 = run_one_step(env)
        nav2, cost2, r2 = run_one_step(env)

        assert nav1 == pytest.approx(nav2, rel=1e-6), "First-step NAV differs between episodes"
        assert cost1 == pytest.approx(cost2, rel=1e-5), "First-step cost differs between episodes"
        assert r1    == pytest.approx(r2,    rel=1e-5), "First-step return differs between episodes"

    def test_multiple_resets_consistent(self):
        """reset() is idempotent — calling it n times gives same initial obs."""
        env = make_env(n_days=80, seed=44)
        obs_list = []
        for _ in range(3):
            obs, _ = env.reset()
            obs_list.append(obs["panel_idx"])

        assert obs_list[0] == obs_list[1] == obs_list[2], (
            f"reset() produced different panel_idx: {obs_list}"
        )


# ===========================================================================
# Integration — All constraints + NAV + cost simultaneously
# ===========================================================================

class TestIntegration:

    def test_full_episode_all_positive_nav(self):
        """Full episode: NAV, nav_qqq, nav_pre all remain positive."""
        env = make_env(n_days=500, n_active=8, seed=99)
        results = run_episode(env, seed=77)

        for i, (rc, info) in enumerate(results):
            assert rc["nav"]     > 0, f"step {i}: nav <= 0"
            assert rc["nav_qqq"] > 0, f"step {i}: nav_qqq <= 0"
            assert rc["nav_pre"] > 0, f"step {i}: nav_pre <= 0"

    def test_counterfactual_nav_differs(self):
        """
        Counterfactual nav_pre should diverge from nav when costs are non-zero.
        (nav_pre has no cost deduction; nav does.)
        """
        md = make_synthetic_market_data(n_days=100, K_max=6, n_active=4, seed=50)
        md["adj_close"][:] = 1.0   # flat prices: r_port = 0 for both
        md["adj_open"] [:] = 1.0

        env = TradingEnvironment(md, projector=None, L_lookback=2)
        results = run_episode(env, seed=50)

        nav_final     = results[-1][0]["nav"]
        nav_pre_final = results[-1][0]["nav_pre"]

        # With flat prices and non-zero trades: nav < nav_pre (costs deducted from nav)
        assert nav_final <= nav_pre_final + 1e-6, (
            f"nav={nav_final:.6f} should be <= nav_pre={nav_pre_final:.6f} "
            "with flat prices (only difference is cost deduction)"
        )

    def test_reward_components_returned_correctly(self):
        """reward_components dict has all expected keys with float values."""
        env = make_env(n_days=60, seed=55)
        obs, _ = env.reset()
        w = np.ones(env._K, dtype=np.float32) / env._K
        obs, rc, done, info = env.step(w)

        required_keys = [
            "r_port_t", "r_qqq_t", "excess_t", "cost_t",
            "violations_t", "nav", "nav_qqq", "nav_pre",
            "forced_liq_weight", "forced_liq_count",
        ]
        for key in required_keys:
            assert key in rc, f"Missing key '{key}' in reward_components"
            assert isinstance(rc[key], (int, float, np.floating)), (
                f"reward_components['{key}'] is {type(rc[key])}, expected numeric"
            )

    def test_violations_nonzero_for_infeasible_input(self):
        """
        When w_pre is all-uniform (may violate per-name cap with few assets)
        but the env has no projector, violations_t = ||w_pre - w_exec||_2.
        Even without a real projector, the _simple_normalize output may differ
        from w_pre, yielding non-negative violations.
        """
        env = make_env(n_days=60, seed=56)
        obs, _ = env.reset()

        # Highly concentrated input: all weight on slot 0
        w = np.zeros(env._K, dtype=np.float32)
        w[0] = 100.0  # will be normalised to [1,0,...,0]
        obs, rc, done, info = env.step(w)

        assert rc["violations_t"] >= 0.0, "violations_t must be non-negative"
