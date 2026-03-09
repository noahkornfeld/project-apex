"""
tests/test_phase6_gate6.py
Gate 6 — Reward Function  (Phase 6, Bible §6)

Six mandatory Gate 6 criteria:
  1. Sign check   – positive excess → Term 1 > 0; negative excess → Term 3 activates
  2. Scale check  – reward is O(1) for typical weekly returns (±2% excess, 15% vol)
  3. Cold start   – reward at t=0 is finite, well-scaled (not NaN, not ±inf, not zero)
  4. Double-cost  – costs appear in Term 1 (via net return) AND Term 4 (explicit)
  5. Clipping     – no reward outside [−5, +5] after all terms summed
  6. Isolation    – each λ can be zeroed independently; remaining terms unchanged
"""

import math
import numpy as np
import pytest

from environment.reward_fn import RewardFunction, from_config

EPS = 1e-9

# ---------------------------------------------------------------------------
# Helper: build a default RewardFunction (matches master_config.yaml values)
# ---------------------------------------------------------------------------

DEFAULT_KWARGS = dict(
    lambda_slow       = 0.75,
    lambda_tail       = 0.40,
    lambda_cost       = 1.0,
    lambda_cv         = 1.0,
    sigma_mkt_window  = 13,
    sigma_port_window = 52,
    sigma_mkt_prior   = 0.020,
    sigma_port_prior  = 0.025,
)


def make_rf(**overrides) -> RewardFunction:
    kw = {**DEFAULT_KWARGS, **overrides}
    return RewardFunction(**kw)


def single_step(rf: RewardFunction, r_port_gross=0.02, r_qqq=0.01,
                cost_t=0.0003, violations_t=0.0):
    """Run one compute() call and return the result dict."""
    return rf.compute(r_port_gross=r_port_gross, r_qqq=r_qqq,
                      cost_t=cost_t, violations_t=violations_t)


# ===========================================================================
# Gate 6.1  Sign check
# ===========================================================================

class TestSignCheck:

    def test_positive_excess_gives_positive_term1(self):
        """
        When e_t > 0 (portfolio beats QQQ on a net basis), Term 1 must be > 0.
        Uses large enough excess to dominate vol estimation noise.
        """
        rf = make_rf()
        rc = single_step(rf, r_port_gross=0.05, r_qqq=0.01, cost_t=0.0001)
        # Net excess = (1.05 * 0.9999 - 1) - 0.01 = 0.04980... > 0
        assert rc["term1"] > 0.0, (
            f"term1={rc['term1']:.6f} should be positive for positive excess return"
        )
        assert rc["e_t"] > 0.0, "e_t should be positive"
        assert rc["norm_excess"] > 0.0, "norm_excess should be positive"

    def test_negative_excess_gives_negative_term1(self):
        """When portfolio underperforms QQQ, Term 1 is negative."""
        rf = make_rf()
        rc = single_step(rf, r_port_gross=-0.01, r_qqq=0.02, cost_t=0.0001)
        assert rc["term1"] < 0.0, (
            f"term1={rc['term1']:.6f} should be negative for negative excess"
        )

    def test_negative_excess_activates_term3(self):
        """
        Term 3 = λ_tail × max(0, -e_t/σ)² must be > 0 when norm_excess < 0.
        When norm_excess > 0, Term 3 must be exactly 0.
        """
        rf_neg = make_rf()
        rc_neg = single_step(rf_neg, r_port_gross=-0.02, r_qqq=0.02, cost_t=0.0)
        assert rc_neg["term3"] > 0.0, (
            f"term3={rc_neg['term3']:.8f} should be positive for negative excess"
        )

        rf_pos = make_rf()
        rc_pos = single_step(rf_pos, r_port_gross=0.04, r_qqq=0.01, cost_t=0.0)
        assert rc_pos["term3"] == pytest.approx(0.0, abs=1e-12), (
            f"term3={rc_pos['term3']:.8e} should be exactly 0 for positive excess"
        )

    def test_term3_grows_with_underperformance(self):
        """Larger underperformance → larger Term 3 (quadratic growth)."""
        rf1 = make_rf()
        rc1 = single_step(rf1, r_port_gross=-0.01, r_qqq=0.02, cost_t=0.0)
        rf2 = make_rf()
        rc2 = single_step(rf2, r_port_gross=-0.03, r_qqq=0.02, cost_t=0.0)
        assert rc2["term3"] > rc1["term3"], (
            f"term3 should grow with underperformance: "
            f"small={rc1['term3']:.6f}, large={rc2['term3']:.6f}"
        )

    def test_term3_quadratic_scaling(self):
        """
        Doubling the normalized underperformance quadruples Term 3.
        Use sigma_mkt = 1.0 (prior = 1.0) to make the normalization trivial.
        """
        rf = make_rf(sigma_mkt_prior=1.0, sigma_port_prior=0.025)
        # Step 1: e_t = -0.02 (after zero cost), σ_mkt ≈ prior = 1.0
        rc1 = rf.compute(r_port_gross=-0.02, r_qqq=0.0, cost_t=0.0, violations_t=0.0)
        rf.reset()
        # Step 2: e_t = -0.04 (double)
        rc2 = rf.compute(r_port_gross=-0.04, r_qqq=0.0, cost_t=0.0, violations_t=0.0)

        ratio = rc2["term3"] / (rc1["term3"] + EPS)
        assert abs(ratio - 4.0) < 0.05, (
            f"Doubling underperformance should 4× term3: ratio={ratio:.4f}"
        )


# ===========================================================================
# Gate 6.2  Scale check
# ===========================================================================

class TestScaleCheck:

    def test_reward_order_one_typical_returns(self):
        """
        Typical weekly params: ±2% excess, 15% annualized vol (≈ 2.08% weekly).
        After cold start stabilises, |reward| should be O(1), not O(0.01) or O(100).
        """
        rf = make_rf()
        # Run enough steps to warm σ_mkt past the 13-week window
        for _ in range(14):
            rf.compute(r_port_gross=0.003, r_qqq=0.001,
                       cost_t=0.0003, violations_t=0.0)

        # Typical positive excess step
        rc_pos = rf.compute(r_port_gross=0.02, r_qqq=0.00,
                            cost_t=0.0003, violations_t=0.0)
        # Typical negative excess step
        rf2 = make_rf()
        for _ in range(14):
            rf2.compute(r_port_gross=0.003, r_qqq=0.001,
                        cost_t=0.0003, violations_t=0.0)
        rc_neg = rf2.compute(r_port_gross=-0.02, r_qqq=0.00,
                             cost_t=0.0003, violations_t=0.0)

        for label, rc in [("positive", rc_pos), ("negative", rc_neg)]:
            r = rc["reward"]
            assert -20.0 < r < 20.0, (
                f"{label}: reward={r:.4f} is wildly out of range (expected O(1))"
            )
            # Strictly O(1): should be within ±10 for typical weekly returns
            assert abs(r) < 10.0, (
                f"{label}: |reward|={abs(r):.4f} exceeds 10 for typical returns"
            )

    def test_scale_at_2pct_excess_15pct_vol(self):
        """
        Exact scale test: e_t = 0.02, σ_mkt = 0.02 (≈15% ann vol).
        Term 1 = 0.02 / 0.02 = 1.0.
        Total reward ≈ 1.0 - λ_slow×σ_port - λ_cost×costs = O(1).
        """
        rf = make_rf(sigma_mkt_prior=0.02, sigma_port_prior=0.02)
        # Single step; prior is used for σ (cold-start)
        rc = rf.compute(r_port_gross=0.02, r_qqq=0.0,
                        cost_t=0.0, violations_t=0.0)

        # Term 1 should be ≈ 0.02/0.02 = 1.0 (prior used, no cost → e_t = 0.02)
        assert abs(rc["term1"] - 1.0) < 0.05, (
            f"term1={rc['term1']:.4f}, expected ≈ 1.0 for e_t=σ_mkt=0.02"
        )
        # Full reward should be O(1), not O(0.01) or O(100)
        assert 0.0 < rc["reward"] < 5.0, (
            f"reward={rc['reward']:.4f} out of O(1) range for typical returns"
        )

    def test_range_assertion_one_year_episode(self):
        """
        Over a 52-step episode with typical ±2% excess, 15% vol:
        all per-step rewards stay within [-5, +5] (clipped).
        """
        rng = np.random.default_rng(42)
        rf  = make_rf()

        rewards = []
        for _ in range(52):
            r_port = float(rng.normal(0.002, 0.020))   # ~±2% weekly
            r_qqq  = float(rng.normal(0.001, 0.015))   # ~±1.5% weekly QQQ
            cost   = float(abs(rng.normal(0.0003, 0.0001)))
            viol   = 0.0
            rc = rf.compute(r_port, r_qqq, cost, viol)
            rewards.append(rc["reward"])

        rewards = np.array(rewards)
        assert rewards.min() >= -5.0, f"Min reward {rewards.min():.4f} < -5 (should be clipped)"
        assert rewards.max() <=  5.0, f"Max reward {rewards.max():.4f} > +5 (should be clipped)"
        # Most rewards in ±3 for typical returns (not using extreme values)
        assert np.abs(rewards).mean() < 5.0, "Mean |reward| > 5 seems too large"


# ===========================================================================
# Gate 6.3  Cold start
# ===========================================================================

class TestColdStart:

    def test_reward_finite_at_step_zero(self):
        """
        First call after reset() must return a finite reward.
        σ_mkt and σ_t are both initialised to their priors at step 0.
        """
        rf = make_rf()
        rc = rf.compute(r_port_gross=0.01, r_qqq=0.005,
                        cost_t=0.0003, violations_t=0.0)

        assert math.isfinite(rc["reward"]), (
            f"reward={rc['reward']} is not finite at cold start"
        )
        assert not math.isnan(rc["reward"]), "reward is NaN at cold start"
        assert rc["reward"] != 0.0, (
            "reward is exactly 0 at cold start — likely a computation bug"
        )

    def test_sigma_mkt_uses_prior_at_step_zero(self):
        """Before any history, σ_mkt should equal the cold-start prior."""
        prior = 0.030
        rf = make_rf(sigma_mkt_prior=prior)
        rc = rf.compute(r_port_gross=0.01, r_qqq=0.005,
                        cost_t=0.0, violations_t=0.0)
        assert abs(rc["sigma_mkt"] - prior) < 1e-6, (
            f"sigma_mkt={rc['sigma_mkt']:.6f} should equal prior={prior} at step 0"
        )

    def test_sigma_port_uses_prior_at_step_zero(self):
        """Before any history, σ_port should equal the cold-start prior."""
        prior = 0.035
        rf = make_rf(sigma_port_prior=prior)
        rc = rf.compute(r_port_gross=0.01, r_qqq=0.005,
                        cost_t=0.0, violations_t=0.0)
        assert abs(rc["sigma_port"] - prior) < 1e-6, (
            f"sigma_port={rc['sigma_port']:.6f} should equal prior={prior} at step 0"
        )

    def test_sigma_transitions_from_prior_to_in_episode(self):
        """
        σ_mkt blends from prior toward in-episode std as the window fills.
        After window_size steps, it should equal pure in-episode std.
        """
        window = 5
        prior  = 0.100   # Deliberately different from data
        rf = make_rf(sigma_mkt_window=window, sigma_mkt_prior=prior,
                     sigma_port_prior=0.025)

        # Feed constant QQQ returns so we know the in-episode std
        qqq_ret = 0.010   # constant → in-episode std = 0 (but ddof=1 with n<2 = prior)
        # Use varying returns so std > 0
        qqq_rets = [0.010, -0.005, 0.015, -0.010, 0.008]   # 5 varied returns
        for r in qqq_rets:
            rc = rf.compute(r_port_gross=r + 0.005, r_qqq=r,
                            cost_t=0.0, violations_t=0.0)

        # After `window` steps: sigma_mkt should be pure in-episode std
        expected_ep_std = float(np.std(qqq_rets, ddof=1))
        assert abs(rc["sigma_mkt"] - expected_ep_std) < 1e-5, (
            f"After {window} steps sigma_mkt={rc['sigma_mkt']:.6f} "
            f"should equal in-episode std={expected_ep_std:.6f}"
        )

    def test_reset_clears_buffers(self):
        """After reset(), σ reverts to prior (buffers cleared)."""
        prior = 0.040
        rf = make_rf(sigma_mkt_prior=prior)
        # Warm up with some history
        for _ in range(20):
            rf.compute(0.002, 0.001, 0.0, 0.0)
        # σ_mkt should now be in-episode (not prior)
        rc_before = rf.compute(0.002, 0.001, 0.0, 0.0)
        assert abs(rc_before["sigma_mkt"] - prior) > 1e-4, (
            "sigma_mkt should have diverged from prior after warmup"
        )
        # Reset and check
        rf.reset()
        rc_after = rf.compute(0.002, 0.001, 0.0, 0.0)
        assert abs(rc_after["sigma_mkt"] - prior) < 1e-6, (
            f"sigma_mkt={rc_after['sigma_mkt']:.6f} should equal prior={prior} "
            "after reset()"
        )


# ===========================================================================
# Gate 6.4  Double-cost
# ===========================================================================

class TestDoubleCost:

    def test_higher_cost_reduces_term1(self):
        """
        Increasing cost_t reduces Term 1 via the net return channel.
        Same gross return; higher cost → lower r_port_net → lower e_t → lower term1.
        """
        r_port_gross = 0.02
        r_qqq = 0.00

        rf_low  = make_rf()
        rf_high = make_rf()

        rc_low  = single_step(rf_low,  r_port_gross=r_port_gross, r_qqq=r_qqq, cost_t=0.001)
        rc_high = single_step(rf_high, r_port_gross=r_port_gross, r_qqq=r_qqq, cost_t=0.010)

        assert rc_high["term1"] < rc_low["term1"], (
            f"Higher cost should reduce term1: "
            f"low={rc_low['term1']:.6f}, high={rc_high['term1']:.6f}"
        )

    def test_higher_cost_increases_term4(self):
        """Increasing cost_t increases Term 4 (explicit penalty channel)."""
        r_port_gross = 0.02
        r_qqq = 0.00

        rf_low  = make_rf()
        rf_high = make_rf()

        rc_low  = single_step(rf_low,  r_port_gross=r_port_gross, r_qqq=r_qqq, cost_t=0.001)
        rc_high = single_step(rf_high, r_port_gross=r_port_gross, r_qqq=r_qqq, cost_t=0.010)

        assert rc_high["term4"] > rc_low["term4"], (
            f"Higher cost should increase term4: "
            f"low={rc_low['term4']:.6f}, high={rc_high['term4']:.6f}"
        )

    def test_both_channels_active_simultaneously(self):
        """
        With lambda_cost=1.0 and non-zero costs:
          - term4 > 0  (explicit channel active)
          - r_port_net < r_port_gross  (implicit channel via net return)
        Both channels must be active in the same step.
        """
        rf = make_rf(lambda_cost=1.0)
        rc = single_step(rf, r_port_gross=0.03, r_qqq=0.01, cost_t=0.005)

        # Explicit channel
        assert rc["term4"] > 0.0, "term4 must be > 0 when cost_t > 0"

        # Implicit channel: net return < gross return
        assert rc["r_port_net"] < 0.03, (
            f"r_port_net={rc['r_port_net']:.6f} should be < gross=0.03 due to cost"
        )

    def test_cost_still_penalises_when_lambda_cost_zero(self):
        """
        Setting lambda_cost=0 zeroes Term 4, but costs still penalise via Term 1
        (net return channel).  Higher cost must still produce lower reward even
        with lambda_cost=0.
        """
        rf_low  = make_rf(lambda_cost=0.0)
        rf_high = make_rf(lambda_cost=0.0)

        rc_low  = single_step(rf_low,  r_port_gross=0.02, r_qqq=0.00, cost_t=0.001)
        rc_high = single_step(rf_high, r_port_gross=0.02, r_qqq=0.00, cost_t=0.015)

        # Term 4 should be 0 in both cases
        assert rc_low["term4"]  == pytest.approx(0.0, abs=1e-12)
        assert rc_high["term4"] == pytest.approx(0.0, abs=1e-12)

        # But reward is lower for higher cost via Term 1
        assert rc_high["reward"] < rc_low["reward"], (
            "Cost should still penalise via Term 1 even when lambda_cost=0"
        )

    def test_decomposition_gross_vs_net(self):
        """
        Verify e_t = r_port_net - r_qqq  and  r_port_net = (1+gross)*(1-cost) - 1.
        """
        rf = make_rf()
        gross, qqq, cost = 0.025, 0.010, 0.004
        rc = rf.compute(r_port_gross=gross, r_qqq=qqq,
                        cost_t=cost, violations_t=0.0)

        expected_net = (1.0 + gross) * (1.0 - cost) - 1.0
        expected_e_t = expected_net - qqq

        assert abs(rc["r_port_net"] - expected_net) < 1e-9, (
            f"r_port_net={rc['r_port_net']:.9f} != expected={expected_net:.9f}"
        )
        assert abs(rc["e_t"] - expected_e_t) < 1e-9, (
            f"e_t={rc['e_t']:.9f} != expected={expected_e_t:.9f}"
        )


# ===========================================================================
# Gate 6.5  Clipping
# ===========================================================================

class TestClipping:

    def test_extremely_positive_return_clips_to_plus5(self):
        """Extreme positive excess return → reward clipped to +5.0."""
        rf = make_rf(sigma_mkt_prior=0.01)  # tiny σ → huge normalized excess
        rc = rf.compute(r_port_gross=0.50, r_qqq=0.0,
                        cost_t=0.0, violations_t=0.0)
        assert rc["reward"] == pytest.approx(5.0, abs=1e-6), (
            f"reward={rc['reward']:.6f} should be clipped to +5.0"
        )
        assert rc["reward_unclipped"] > 5.0, "Pre-clip reward should exceed +5"
        assert rc["was_clipped"] is True

    def test_extremely_negative_return_clips_to_minus5(self):
        """Extreme negative excess return → reward clipped to -5.0."""
        rf = make_rf(sigma_mkt_prior=0.01)
        rc = rf.compute(r_port_gross=-0.50, r_qqq=0.0,
                        cost_t=0.0, violations_t=0.0)
        assert rc["reward"] == pytest.approx(-5.0, abs=1e-6), (
            f"reward={rc['reward']:.6f} should be clipped to -5.0"
        )
        assert rc["reward_unclipped"] < -5.0, "Pre-clip reward should be below -5"
        assert rc["was_clipped"] is True

    def test_max_min_over_full_episode(self):
        """
        Over a 100-step episode with random extreme inputs,
        no clipped reward escapes the [-5, +5] band.
        """
        rng = np.random.default_rng(777)
        rf  = make_rf()
        rewards = []
        for _ in range(100):
            r_port = float(rng.uniform(-0.20, 0.20))
            r_qqq  = float(rng.uniform(-0.10, 0.10))
            cost   = float(rng.uniform(0.0, 0.02))
            viol   = float(rng.uniform(0.0, 0.5))
            rc = rf.compute(r_port, r_qqq, cost, viol)
            rewards.append(rc["reward"])

        rewards = np.array(rewards)
        assert rewards.max() <= 5.0 + 1e-9, (
            f"Max clipped reward {rewards.max():.6f} exceeds +5"
        )
        assert rewards.min() >= -5.0 - 1e-9, (
            f"Min clipped reward {rewards.min():.6f} below -5"
        )

    def test_no_clipping_for_typical_inputs(self):
        """
        With typical weekly returns (±2%), reward should NOT be clipped.
        Warm-up uses varied QQQ returns so sigma_mkt stays realistically non-zero.
        """
        rng = np.random.default_rng(9)
        rf = make_rf()
        for _ in range(20):
            r_qqq = float(rng.normal(0.001, 0.015))   # realistic QQQ weekly
            rf.compute(r_qqq + 0.002, r_qqq, 0.0003, 0.0)

        rc = rf.compute(r_port_gross=0.02, r_qqq=0.00,
                        cost_t=0.0003, violations_t=0.0)
        assert not rc["was_clipped"], (
            f"Typical-return step should NOT be clipped; "
            f"reward_unclipped={rc['reward_unclipped']:.4f}, "
            f"sigma_mkt={rc['sigma_mkt']:.6f}"
        )

    def test_q_value_no_clip(self):
        """
        Reward_unclipped may exceed ±5 — this confirms Q-value clipping
        is NOT applied at the reward-function level (§6.5).
        """
        rf = make_rf(sigma_mkt_prior=0.005)
        rc = rf.compute(r_port_gross=0.30, r_qqq=0.0,
                        cost_t=0.0, violations_t=0.0)
        # reward is clipped, but reward_unclipped should exceed 5
        assert rc["reward_unclipped"] > 5.0, (
            "reward_unclipped should exceed +5 for extreme inputs "
            "(confirms no Q-value clipping at this layer)"
        )


# ===========================================================================
# Gate 6.6  Isolation — lambda-zero sweep
# ===========================================================================

class TestIsolation:
    """
    Each term can be independently zeroed by setting its λ = 0.
    All other terms must remain unchanged.
    """

    def _base_result(self):
        """Reference result with all lambdas at default values."""
        rf = make_rf()
        return rf.compute(r_port_gross=0.015, r_qqq=0.008,
                          cost_t=0.0005, violations_t=0.05)

    def test_lambda_slow_zero_kills_term2_only(self):
        """λ_slow = 0 → term2 = 0; term1, term3, term4, term5 unchanged."""
        rf_base = make_rf()
        rc_base = rf_base.compute(0.015, 0.008, 0.0005, 0.05)

        rf_zero = make_rf(lambda_slow=0.0)
        rc_zero = rf_zero.compute(0.015, 0.008, 0.0005, 0.05)

        assert rc_zero["term2"] == pytest.approx(0.0, abs=1e-12)
        assert abs(rc_zero["term1"] - rc_base["term1"]) < 1e-9
        assert abs(rc_zero["term3"] - rc_base["term3"]) < 1e-9
        assert abs(rc_zero["term4"] - rc_base["term4"]) < 1e-9
        assert abs(rc_zero["term5"] - rc_base["term5"]) < 1e-9

    def test_lambda_tail_zero_kills_term3_only(self):
        """λ_tail = 0 → term3 = 0; term1, term2, term4, term5 unchanged."""
        rf_base = make_rf()
        # Use negative excess to ensure term3 would be active
        rc_base = rf_base.compute(-0.01, 0.02, 0.0005, 0.05)

        rf_zero = make_rf(lambda_tail=0.0)
        rc_zero = rf_zero.compute(-0.01, 0.02, 0.0005, 0.05)

        assert rc_zero["term3"] == pytest.approx(0.0, abs=1e-12)
        assert abs(rc_zero["term1"] - rc_base["term1"]) < 1e-9
        assert abs(rc_zero["term2"] - rc_base["term2"]) < 1e-9
        assert abs(rc_zero["term4"] - rc_base["term4"]) < 1e-9
        assert abs(rc_zero["term5"] - rc_base["term5"]) < 1e-9

    def test_lambda_cost_zero_kills_term4_only(self):
        """λ_cost = 0 → term4 = 0; term1, term2, term3, term5 unchanged."""
        rc_base = self._base_result()

        rf_zero = make_rf(lambda_cost=0.0)
        rc_zero = rf_zero.compute(0.015, 0.008, 0.0005, 0.05)

        assert rc_zero["term4"] == pytest.approx(0.0, abs=1e-12)
        assert abs(rc_zero["term1"] - rc_base["term1"]) < 1e-9
        assert abs(rc_zero["term2"] - rc_base["term2"]) < 1e-9
        assert abs(rc_zero["term3"] - rc_base["term3"]) < 1e-9
        assert abs(rc_zero["term5"] - rc_base["term5"]) < 1e-9

    def test_lambda_cv_zero_kills_term5_only(self):
        """λ_cv = 0 → term5 = 0; term1, term2, term3, term4 unchanged."""
        rc_base = self._base_result()

        rf_zero = make_rf(lambda_cv=0.0)
        rc_zero = rf_zero.compute(0.015, 0.008, 0.0005, 0.05)

        assert rc_zero["term5"] == pytest.approx(0.0, abs=1e-12)
        assert abs(rc_zero["term1"] - rc_base["term1"]) < 1e-9
        assert abs(rc_zero["term2"] - rc_base["term2"]) < 1e-9
        assert abs(rc_zero["term3"] - rc_base["term3"]) < 1e-9
        assert abs(rc_zero["term4"] - rc_base["term4"]) < 1e-9

    def test_all_lambdas_zero_leaves_only_term1(self):
        """All penalty lambdas = 0 → reward = term1 only."""
        rf = make_rf(lambda_slow=0.0, lambda_tail=0.0,
                     lambda_cost=0.0, lambda_cv=0.0)
        rc = rf.compute(0.015, 0.008, 0.0005, 0.05)

        assert rc["term2"] == pytest.approx(0.0, abs=1e-12)
        assert rc["term3"] == pytest.approx(0.0, abs=1e-12)
        assert rc["term4"] == pytest.approx(0.0, abs=1e-12)
        assert rc["term5"] == pytest.approx(0.0, abs=1e-12)

        # With all penalties zeroed, reward (unclipped) = term1
        assert abs(rc["reward_unclipped"] - rc["term1"]) < 1e-9, (
            f"reward_unclipped={rc['reward_unclipped']:.9f} "
            f"should equal term1={rc['term1']:.9f}"
        )

    def test_lambda_sweep_remaining_terms_unchanged(self):
        """
        Full sweep: zero each lambda one at a time.
        For each zeroed lambda, verify all OTHER term values are identical
        to the default (no cross-contamination).
        """
        base_inputs = dict(r_port_gross=0.012, r_qqq=-0.005,
                           cost_t=0.0008, violations_t=0.03)

        rf_ref = make_rf()
        rc_ref = rf_ref.compute(**base_inputs)

        sweep = [
            ("lambda_slow", "term2"),
            ("lambda_tail", "term3"),
            ("lambda_cost", "term4"),
            ("lambda_cv",   "term5"),
        ]

        all_terms = {"term1", "term2", "term3", "term4", "term5"}
        for lam_name, zeroed_term in sweep:
            rf_z = make_rf(**{lam_name: 0.0})
            rc_z = rf_z.compute(**base_inputs)

            assert rc_z[zeroed_term] == pytest.approx(0.0, abs=1e-12), (
                f"{lam_name}=0 should zero {zeroed_term}"
            )
            for other in all_terms - {zeroed_term}:
                assert abs(rc_z[other] - rc_ref[other]) < 1e-9, (
                    f"Setting {lam_name}=0 unexpectedly changed {other}: "
                    f"ref={rc_ref[other]:.9f}, got={rc_z[other]:.9f}"
                )


# ===========================================================================
# Additional correctness tests
# ===========================================================================

class TestFormula:

    def test_full_formula_manual_calculation(self):
        """
        Hand-calculate all 5 terms and compare against the implementation.

        Inputs (chosen to avoid cold-start noise):
          r_port_gross = 0.020, r_qqq = 0.005, cost_t = 0.002, violations = 0.10
          sigma_mkt_prior = 0.020, sigma_port_prior = 0.025

        At step 0 (cold start):
          r_port_net = (1.020)(0.998) - 1 = 0.017960
          e_t        = 0.017960 - 0.005 = 0.012960
          sigma_mkt  = prior = 0.020
          sigma_port = prior = 0.025
          norm_excess = 0.012960 / 0.020 = 0.648
          term1 =  0.648
          term2 =  0.75 × 0.025  = 0.018750
          term3 =  0 (norm_excess > 0)
          term4 =  1.0 × 0.002   = 0.002
          term5 =  1.0 × 0.10    = 0.10
          reward_raw = 0.648 - 0.01875 - 0 - 0.002 - 0.10 = 0.52725
        """
        rf = make_rf(sigma_mkt_prior=0.020, sigma_port_prior=0.025)
        rc = rf.compute(r_port_gross=0.020, r_qqq=0.005,
                        cost_t=0.002, violations_t=0.10)

        r_net     = (1.020) * (1.0 - 0.002) - 1.0
        e_t       = r_net - 0.005
        norm_ex   = e_t / 0.020
        expected  = {
            "r_port_net":  r_net,
            "e_t":         e_t,
            "term1":       norm_ex,
            "term2":       0.75 * 0.025,
            "term3":       0.0,
            "term4":       1.0 * 0.002,
            "term5":       1.0 * 0.10,
        }
        expected["reward_unclipped"] = (
            expected["term1"] - expected["term2"] - expected["term3"]
            - expected["term4"] - expected["term5"]
        )

        for key, exp_val in expected.items():
            assert abs(rc[key] - exp_val) < 1e-7, (
                f"{key}: got={rc[key]:.9f}, expected={exp_val:.9f}"
            )

    def test_reward_components_sum_to_reward_unclipped(self):
        """reward_unclipped = term1 - term2 - term3 - term4 - term5 exactly."""
        rf = make_rf()
        for _ in range(5):
            rf.compute(0.003, 0.001, 0.0003, 0.01)

        rc = rf.compute(r_port_gross=-0.01, r_qqq=0.02,
                        cost_t=0.001, violations_t=0.05)

        expected_raw = (rc["term1"] - rc["term2"] - rc["term3"]
                        - rc["term4"] - rc["term5"])
        assert abs(rc["reward_unclipped"] - expected_raw) < 1e-9, (
            f"reward_unclipped={rc['reward_unclipped']:.9f} != "
            f"sum_of_terms={expected_raw:.9f}"
        )

    def test_violations_term_proportional_to_lambda_cv(self):
        """term5 = lambda_cv × violations_t exactly."""
        for lam in [0.0, 0.5, 1.0, 2.0]:
            rf = make_rf(lambda_cv=lam)
            rc = rf.compute(0.01, 0.005, 0.0, 0.12345)
            expected = lam * 0.12345
            assert abs(rc["term5"] - expected) < 1e-10, (
                f"lambda_cv={lam}: term5={rc['term5']:.10f} != {expected:.10f}"
            )

    def test_cost_penalty_proportional_to_lambda_cost(self):
        """term4 = lambda_cost × cost_t exactly."""
        for lam in [0.0, 0.5, 1.0, 3.0]:
            rf = make_rf(lambda_cost=lam)
            rc = rf.compute(0.01, 0.005, 0.00777, 0.0)
            expected = lam * 0.00777
            assert abs(rc["term4"] - expected) < 1e-10, (
                f"lambda_cost={lam}: term4={rc['term4']:.10f} != {expected:.10f}"
            )


class TestFromConfig:

    def test_from_config_dict(self):
        """from_config() builds a RewardFunction from a plain dict."""
        cfg = dict(
            lambda_slow=0.5, lambda_tail=0.3, lambda_cost=2.0, lambda_cv=0.5,
            sigma_mkt_window_weeks=10, sigma_port_window_weeks=30,
        )
        rf = from_config(cfg)
        assert rf._lam_slow  == 0.5
        assert rf._lam_tail  == 0.3
        assert rf._lam_cost  == 2.0
        assert rf._lam_cv    == 0.5
        assert rf._w_mkt     == 10
        assert rf._w_port    == 30

    def test_from_env_wrapper(self):
        """compute_from_env() extracts components correctly from env dict."""
        rf = make_rf()
        env_rc = dict(
            r_port_t     = 0.015,
            r_qqq_t      = 0.005,
            cost_t       = 0.0003,
            violations_t = 0.0,
        )
        rc_wrapper = rf.compute_from_env(env_rc)
        rf2 = make_rf()
        rc_direct  = rf2.compute(
            r_port_gross=0.015, r_qqq=0.005, cost_t=0.0003, violations_t=0.0
        )

        assert abs(rc_wrapper["reward"] - rc_direct["reward"]) < 1e-9
        assert abs(rc_wrapper["term1"]  - rc_direct["term1"])  < 1e-9

    def test_integration_with_synthetic_env(self):
        """
        Run a full episode using TradingEnvironment + RewardFunction together.
        Verify reward is finite at every step and within [-5, +5].
        """
        from environment.market_data import make_synthetic_market_data
        from environment.trading_env import TradingEnvironment

        md  = make_synthetic_market_data(n_days=120, K_max=6, n_active=4, seed=123)
        env = TradingEnvironment(md, projector=None, L_lookback=4)
        rf  = make_rf()

        obs, _ = env.reset()
        rf.reset()

        rng  = np.random.default_rng(456)
        done = False
        step = 0
        while not done:
            w = np.abs(rng.normal(size=env._K)).astype(np.float32)
            obs, env_rc, done, info = env.step(w)
            reward_rc = rf.compute_from_env(env_rc)
            step += 1

            r = reward_rc["reward"]
            assert math.isfinite(r), f"Step {step}: reward is not finite"
            assert -5.0 <= r <= 5.0, (
                f"Step {step}: reward={r:.4f} outside [-5, +5]"
            )
