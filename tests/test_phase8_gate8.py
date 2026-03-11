"""
tests/test_phase8_gate8.py
Gate 8 — Replay Buffer and Collection  (Phase 8, Bible §8.2-8.6)

Six mandatory Gate 8 criteria:
  1. n-step    — No n-step return spans an episode boundary (done_n check)
  2. Warmup drop — Transitions at warmup indices 49, 50, 51 are not in buffer
  3. Recency   — Sampling 10K transitions: recent transitions drawn more often
  4. Augmentation — Critic batch has noise; actor batch is clean (same sample)
  5. Capacity  — Buffer correctly wraps at capacity=800
  6. Stats     — Running reward_std matches np.std of all stored rewards
"""

from __future__ import annotations

import numpy as np
import pytest

from training.replay_buffer import ReplayBuffer

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

K = 10      # small K_max for test speed
GAMMA = 0.975
N_STEP = 4
WARMUP_STEPS = 52


def make_buf(**kw) -> ReplayBuffer:
    defaults = dict(
        capacity=800, K_max=K, n_step=N_STEP, gamma=GAMMA,
        half_life_weeks=156.0, warmup_steps=WARMUP_STEPS,
        warmup_exclusion_threshold=128,
        aug_reward_noise_factor=0.01, aug_obs_noise_factor=0.015,
        trading_days_per_week=5,
    )
    defaults.update(kw)
    return ReplayBuffer(**defaults)


def _make_mask(n_active: int = 5) -> np.ndarray:
    m = np.zeros(K, dtype=np.float32)
    m[:n_active] = 1.0
    return m


def _make_w(n_active: int = 5) -> np.ndarray:
    w = np.zeros(K, dtype=np.float32)
    w[:n_active] = 1.0 / n_active
    return w


def _make_episode(
    T: int,
    base_t_idx: int = 0,
    base_reward: float = 1.0,
    done_at: int = None,        # force done=True at this step (0-indexed)
) -> list:
    """Return a list of T transition dicts with deterministic rewards."""
    transitions = []
    for t in range(T):
        transitions.append({
            "t_idx":      base_t_idx + t,
            "t_idx_next": base_t_idx + t + 1,
            "mask_t":     _make_mask(),
            "w_pre":      _make_w(),
            "w_exec":     _make_w(),
            "reward":     float(base_reward * (t + 1)),
            "done":       (t == done_at) if done_at is not None else (t == T - 1),
        })
    return transitions


# ===========================================================================
# Gate 8.1  n-step boundary test
# ===========================================================================

class TestNStep:
    """No n-step return spans an episode boundary."""

    def test_n_step_within_episode(self):
        """
        Episode of length 10, done at last step.  For step 0 the n-step
        return must equal  Σ_{k=0}^{3} γ^k (k+1).
        """
        buf = make_buf()
        ep  = _make_episode(T=10, base_reward=1.0, done_at=9)
        buf.add_episode(ep, is_warmup=False)

        # Manually compute expected R_n for step 0
        expected = sum(GAMMA**k * float(k+1) for k in range(N_STEP))
        stored = buf.stored_rewards()

        # First stored transition corresponds to ep step 0
        assert abs(stored[0] - expected) < 1e-4, (
            f"R_n[0]={stored[0]:.6f}, expected={expected:.6f}"
        )

    def test_done_truncates_return(self):
        """
        If done occurs at step 2 (0-indexed), steps 0,1,2 are stored.
        Step 0 should have R_n = r0 + γ r1 + γ² r2, done_n=True.
        """
        # Use T=3, done_at=2 so episode genuinely ends at step 2.
        # All 3 stored transitions must reach the done step within n_step=4.
        buf = make_buf()
        ep  = _make_episode(T=3, base_reward=1.0, done_at=2)
        buf.add_episode(ep, is_warmup=False)

        # Expected R_n for step 0: r0=1, r1=2, r2=3
        expected_r0 = 1.0 + GAMMA * 2.0 + GAMMA**2 * 3.0
        stored = buf.stored_rewards()
        assert abs(stored[0] - expected_r0) < 1e-4, (
            f"R_n[0]={stored[0]:.6f}, expected={expected_r0:.6f}"
        )

        # All 3 stored transitions must have done_n=True
        flags = buf._done_n[:buf.size]
        assert flags.all(), (
            f"All transitions in a truncated episode should have done_n=True; "
            f"got {flags}"
        )

    def test_done_n_false_before_episode_end(self):
        """
        Long episode (12 steps, done only at end).
        Steps 0..7 don't reach the done step within n_step=4 → done_n=False.
        Steps 8,9,10,11 do reach it → done_n=True.
        """
        buf = make_buf()
        T   = 12
        ep  = _make_episode(T=T, base_reward=1.0, done_at=T - 1)
        buf.add_episode(ep, is_warmup=False)

        done_flags = buf._done_n[:buf.size]

        # Steps 0..7: done_at=11, so step 0 looks ahead to steps 0..3 (no done)
        for t in range(T - N_STEP):
            assert not done_flags[t], (
                f"Step {t}: done_n should be False (done is {N_STEP}+ steps away)"
            )
        # Steps 8,9,10,11: n-step window reaches step 11 (done)
        for t in range(T - N_STEP, T):
            assert done_flags[t], (
                f"Step {t}: done_n should be True (done within n_step window)"
            )

    def test_no_cross_episode_contamination(self):
        """
        Two episodes added separately.  The last transitions of episode 1
        must NOT have rewards from episode 2.
        """
        buf = make_buf()
        # Episode 1: rewards 1..10, done at last step
        ep1 = _make_episode(T=10, base_t_idx=0,   base_reward=1.0, done_at=9)
        # Episode 2: all rewards are very large (1000+) to detect contamination
        ep2 = _make_episode(T=10, base_t_idx=100, base_reward=1000.0, done_at=9)

        buf.add_episode(ep1, is_warmup=False)
        buf.add_episode(ep2, is_warmup=False)

        # All episode-1 transitions should have R_n < 100 (rewards are 1..10)
        ep1_rewards = buf.stored_rewards()[:10]
        assert ep1_rewards.max() < 100, (
            f"Episode-1 transitions appear to contain episode-2 rewards: "
            f"max R_n = {ep1_rewards.max():.1f}"
        )

    def test_n_step_uses_gamma_n_step(self):
        """
        gamma_n stored must always equal γ^{n_step}, even when done_n=True.
        (§8.2: γ^n = γ^{n_step} stored per transition)
        """
        buf = make_buf()
        ep  = _make_episode(T=6, done_at=2)
        buf.add_episode(ep, is_warmup=False)

        expected_gn = GAMMA ** N_STEP
        gn_stored   = buf._gn[:buf.size]
        np.testing.assert_allclose(
            gn_stored, expected_gn, rtol=1e-5,
            err_msg="gamma_n must equal γ^n_step for all transitions"
        )


# ===========================================================================
# Gate 8.2  Warmup drop test
# ===========================================================================

class TestWarmupDrop:
    """Transitions at warmup indices 49, 50, 51 must NOT be stored."""

    def test_last_three_warmup_steps_not_stored(self):
        """
        A 52-step warmup episode: steps 49, 50, 51 must be absent from buffer.
        Only steps 0..48 (49 transitions) should be stored.
        """
        buf = make_buf()
        ep  = _make_episode(T=WARMUP_STEPS, base_t_idx=0, done_at=WARMUP_STEPS - 1)
        stored = buf.add_episode(ep, is_warmup=True)

        # Exactly 49 stored (52 - 3 = 49)
        assert stored == WARMUP_STEPS - (N_STEP - 1), (
            f"Expected {WARMUP_STEPS - (N_STEP - 1)} stored, got {stored}"
        )
        assert buf.size == WARMUP_STEPS - (N_STEP - 1)

    def test_dropped_steps_t_idx_absent(self):
        """
        t_idx values 49, 50, 51 must NOT appear in the buffer.
        """
        buf = make_buf()
        ep  = _make_episode(T=WARMUP_STEPS, base_t_idx=0, done_at=WARMUP_STEPS - 1)
        buf.add_episode(ep, is_warmup=True)

        stored_t = set(buf.stored_t_idx().tolist())
        for drop_idx in [49, 50, 51]:
            assert drop_idx not in stored_t, (
                f"Warmup step {drop_idx} should have been dropped but t_idx "
                f"{drop_idx} found in buffer"
            )

    def test_warmup_steps_before_49_are_stored(self):
        """Steps 0..48 must all be stored."""
        buf = make_buf()
        ep  = _make_episode(T=WARMUP_STEPS, base_t_idx=0, done_at=WARMUP_STEPS - 1)
        buf.add_episode(ep, is_warmup=True)

        stored_t = set(buf.stored_t_idx().tolist())
        for keep_idx in range(49):
            assert keep_idx in stored_t, (
                f"Warmup step {keep_idx} should be stored but t_idx not found"
            )

    def test_warmup_flag_set_correctly(self):
        """All stored warmup transitions must have warmup_flag=True."""
        buf = make_buf()
        ep  = _make_episode(T=WARMUP_STEPS, done_at=WARMUP_STEPS - 1)
        buf.add_episode(ep, is_warmup=True)

        assert buf._warmup_flag[:buf.size].all(), (
            "Stored warmup transitions must all have warmup_flag=True"
        )

    def test_policy_transitions_not_flagged(self):
        """Policy-generated transitions have warmup_flag=False."""
        buf = make_buf()
        ep  = _make_episode(T=10, done_at=9)
        buf.add_episode(ep, is_warmup=False)

        assert not buf._warmup_flag[:buf.size].any(), (
            "Policy transitions must have warmup_flag=False"
        )


# ===========================================================================
# Gate 8.3  Recency weighting test
# ===========================================================================

class TestRecencyWeighting:
    """Recent transitions are drawn more often than old ones under recency sampling."""

    def _fill_old_and_new(self, buf: ReplayBuffer, n_old: int = 100, n_new: int = 100):
        """
        Fill buffer with 'old' transitions (t_idx=0..n_old-1) and
        'new' transitions (t_idx=2000..2000+n_new-1), all policy-generated.

        Gap of 2000 trading days ≈ 400 weeks >> half_life=156 weeks, so
        recency weights for old entries are ~exp(-ln2*400/156) ≈ 0.17× new.
        Expected P(new) ≈ 100/(100 + 17) ≈ 85%, well above 70% threshold.
        """
        # Old transitions (low t_idx → far past)
        old_ep = []
        for i in range(n_old):
            old_ep.append({
                "t_idx": i, "t_idx_next": i + 1,
                "mask_t": _make_mask(), "w_pre": _make_w(), "w_exec": _make_w(),
                "reward": float(i), "done": (i == n_old - 1),
            })
        buf.add_episode(old_ep, is_warmup=False)

        # New transitions (high t_idx → recent, 2000 days ≈ 400 weeks gap)
        new_ep = []
        for i in range(n_new):
            t = 2000 + i
            new_ep.append({
                "t_idx": t, "t_idx_next": t + 1,
                "mask_t": _make_mask(), "w_pre": _make_w(), "w_exec": _make_w(),
                "reward": float(i), "done": (i == n_new - 1),
            })
        buf.add_episode(new_ep, is_warmup=False)

    def test_recent_drawn_more_often(self):
        """
        Sample 10K transitions: new-group (t_idx≥500) should be drawn
        significantly more often than old-group (t_idx<100).
        """
        buf = make_buf()
        self._fill_old_and_new(buf, n_old=100, n_new=100)

        rng = np.random.default_rng(0)
        batch = buf.sample(10_000, critic=False, rng=rng)

        t_idx_sampled  = batch["t_idx"]
        n_new_sampled  = (t_idx_sampled >= 2000).sum()
        n_old_sampled  = (t_idx_sampled  < 100).sum()
        pct_new        = n_new_sampled / len(t_idx_sampled)

        # Under uniform sampling: pct_new ≈ 0.50
        # Under recency weighting (old ~500 weeks older): pct_new >> 0.50
        assert pct_new > 0.70, (
            f"Expected >70% new transitions under recency weighting, "
            f"got {pct_new:.2%} (old: {n_old_sampled}, new: {n_new_sampled})"
        )

    def test_recency_weights_monotone_with_age(self):
        """
        Older transitions should have strictly lower weight than newer ones.
        Verify the weight formula: w = exp(-ln2/156 * age_weeks).
        """
        buf = make_buf()
        self._fill_old_and_new(buf, n_old=50, n_new=50)

        w = buf._recency_weights()
        t = buf.stored_t_idx()

        # Sort by t_idx and check weights are non-decreasing
        order = np.argsort(t)
        w_sorted = w[order]
        # Each weight should be ≤ the next (older = smaller weight)
        assert (w_sorted[:-1] <= w_sorted[1:] + 1e-9).all(), (
            "Recency weights must be non-decreasing with t_idx (newer = higher weight)"
        )

    def test_recency_chi_square(self):
        """
        Chi-square style test: observed frequency ratios between
        recent and old bins should significantly exceed 1:1.
        """
        buf = make_buf()
        self._fill_old_and_new(buf, n_old=100, n_new=100)

        rng = np.random.default_rng(1)
        batch = buf.sample(10_000, rng=rng)
        t_idx = batch["t_idx"]

        old_count = (t_idx < 100).sum()
        new_count = (t_idx >= 2000).sum()

        # Expected under uniform: ~50/50; under recency: strongly skewed toward new
        # Simple ratio check
        ratio = new_count / max(old_count, 1)
        assert ratio > 2.0, (
            f"Recency ratio new/old = {ratio:.2f}, expected > 2.0"
        )


# ===========================================================================
# Gate 8.4  Augmentation test
# ===========================================================================

class TestAugmentation:
    """Critic batch has noise; actor batch is clean (same underlying sample)."""

    def _fill(self, buf: ReplayBuffer, T: int = 50):
        ep = _make_episode(T=T, base_reward=1.0, done_at=T - 1)
        buf.add_episode(ep, is_warmup=False)

    def test_critic_r_n_differs_from_actor(self):
        """
        Same seed → same indices; critic R_n must differ from actor R_n.
        """
        buf = make_buf()
        self._fill(buf)

        rng_actor  = np.random.default_rng(42)
        rng_critic = np.random.default_rng(42)

        actor_batch  = buf.sample(32, critic=False, rng=rng_actor)
        critic_batch = buf.sample(32, critic=True,  rng=rng_critic)

        # Same indices sampled (same seed, same buffer state)
        np.testing.assert_array_equal(actor_batch["idx"], critic_batch["idx"])

        # R_n should differ for critic (noise added)
        r_diff = np.abs(actor_batch["R_n"] - critic_batch["R_n"]).max()
        assert r_diff > 1e-6, (
            f"Critic R_n should have noise added, but max diff = {r_diff:.2e}"
        )

    def test_actor_r_n_clean(self):
        """
        Actor batch R_n must be identical to R_n_clean (no noise).
        """
        buf = make_buf()
        self._fill(buf)

        rng   = np.random.default_rng(7)
        batch = buf.sample(32, critic=False, rng=rng)
        np.testing.assert_array_equal(
            batch["R_n"], batch["R_n_clean"],
            err_msg="Actor batch R_n must equal R_n_clean (no augmentation)"
        )

    def test_critic_aug_obs_factor_nonzero(self):
        """Critic batch must include nonzero aug_obs_std_factor."""
        buf = make_buf()
        self._fill(buf)

        critic_batch = buf.sample(32, critic=True)
        actor_batch  = buf.sample(32, critic=False)

        assert critic_batch["aug_obs_std_factor"] > 0.0, (
            "Critic batch aug_obs_std_factor must be > 0"
        )
        assert actor_batch["aug_obs_std_factor"] == 0.0, (
            "Actor batch aug_obs_std_factor must be 0"
        )

    def test_augmentation_resampled_every_draw(self):
        """
        Two independent critic draws on the same indices should produce
        different R_n values (noise resampled each time).
        """
        buf = make_buf()
        self._fill(buf)

        rng1 = np.random.default_rng(10)
        rng2 = np.random.default_rng(99)  # different seed

        b1 = buf.sample(32, critic=True, rng=rng1)
        b2 = buf.sample(32, critic=True, rng=rng2)

        # Different RNG seeds → different noise realizations
        diff = np.abs(b1["R_n"] - b2["R_n"]).max()
        assert diff > 1e-8, (
            "Different noise seeds should produce different augmented R_n"
        )

    def test_critic_noise_scale(self):
        """
        Noise std should be approximately aug_reward_noise_factor × buffer_reward_std.
        """
        buf = make_buf()
        ep  = _make_episode(T=200, base_reward=1.0, done_at=199)
        buf.add_episode(ep)

        r_std     = buf.buffer_reward_std
        expected_noise_std = 0.01 * r_std

        rng = np.random.default_rng(0)
        diffs = []
        for _ in range(100):
            b_actor  = buf.sample(64, critic=False, rng=rng)
            b_critic = buf.sample(64, critic=True,  rng=rng)
            # R_n indices differ here since rng advances; just collect raw noise
            diffs.append(b_critic["R_n"] - b_critic["R_n_clean"])

        noise_std_obs = np.std(np.concatenate(diffs))
        # Should be within 50% of expected
        assert abs(noise_std_obs - expected_noise_std) / max(expected_noise_std, EPS) < 0.5, (
            f"Observed noise std {noise_std_obs:.6f} too far from "
            f"expected {expected_noise_std:.6f}"
        )


EPS = 1e-8


# ===========================================================================
# Gate 8.5  Capacity / wrap test
# ===========================================================================

class TestCapacity:
    """Buffer wraps correctly at capacity=800."""

    def test_size_caps_at_capacity(self):
        """After adding more than capacity transitions, size == capacity."""
        cap = 100  # use small cap for speed
        buf = make_buf(capacity=cap)

        total = 0
        while total < cap + 50:
            ep = _make_episode(T=10, base_t_idx=total, done_at=9)
            buf.add_episode(ep)
            total += 10

        assert buf.size == cap, f"Buffer size {buf.size} should be capped at {cap}"

    def test_circular_overwrite(self):
        """After wrapping, new entries overwrite old slots correctly."""
        cap = 20
        buf = make_buf(capacity=cap)

        # Fill exactly to capacity with t_idx = 0..cap-1
        ep_exact = _make_episode(T=cap, base_t_idx=0, done_at=cap - 1)
        buf.add_episode(ep_exact)
        assert buf.size == cap

        # Add 5 more transitions — these should overwrite the oldest 5 slots
        ep_new = _make_episode(T=5, base_t_idx=1000, done_at=4)
        buf.add_episode(ep_new)
        assert buf.size == cap, "Size should remain at capacity after overflow"

        # The 5 new entries must exist somewhere in the buffer
        stored_t = buf.stored_t_idx()
        new_present = [(1000 + i) in stored_t for i in range(5)]
        assert all(new_present), (
            f"New entries (t_idx 1000-1004) should overwrite oldest; "
            f"found: {new_present}"
        )

    def test_sample_still_works_after_overflow(self):
        """Sampling must succeed without errors after buffer overflow."""
        cap = 50
        buf = make_buf(capacity=cap)

        # Add 80 transitions in 8 batches
        for batch_i in range(8):
            ep = _make_episode(T=10, base_t_idx=batch_i * 10, done_at=9)
            buf.add_episode(ep)

        rng = np.random.default_rng(0)
        batch = buf.sample(32, rng=rng)
        assert len(batch["t_idx"]) == 32

    def test_head_wraps_correctly(self):
        """Internal _head pointer wraps modulo capacity."""
        cap = 20
        buf = make_buf(capacity=cap)

        # Fill exactly to cap
        ep = _make_episode(T=cap, base_t_idx=0, done_at=cap - 1)
        buf.add_episode(ep)
        assert buf._head == 0, (
            f"After filling exactly to capacity, _head should be 0, got {buf._head}"
        )

        # Add one more — head moves to 1
        ep_one = _make_episode(T=1, base_t_idx=999, done_at=0)
        buf.add_episode(ep_one)
        assert buf._head == 1


# ===========================================================================
# Gate 8.6  Running statistics test
# ===========================================================================

class TestRunningStats:
    """buffer_reward_std matches np.std of all stored rewards."""

    def test_reward_std_matches_stored(self):
        """
        After adding transitions, buffer_reward_std must equal np.std
        of the stored R_n values.
        """
        buf = make_buf()
        ep  = _make_episode(T=50, base_reward=2.5, done_at=49)
        buf.add_episode(ep)

        stored   = buf.stored_rewards()
        expected = float(np.std(stored, ddof=1))
        actual   = buf.buffer_reward_std

        assert abs(actual - expected) < 1e-5, (
            f"buffer_reward_std={actual:.6f}, np.std={expected:.6f}"
        )

    def test_reward_std_after_multiple_episodes(self):
        """Std is consistent across multiple inserted episodes."""
        buf = make_buf()
        for i in range(5):
            ep = _make_episode(T=20, base_t_idx=i * 100,
                               base_reward=float(i + 1), done_at=19)
            buf.add_episode(ep)

        stored   = buf.stored_rewards()
        expected = float(np.std(stored, ddof=1))
        actual   = buf.buffer_reward_std

        assert abs(actual - expected) < 1e-5, (
            f"buffer_reward_std={actual:.6f}, np.std={expected:.6f}"
        )

    def test_reward_std_updates_after_overflow(self):
        """Std reflects only the current contents after circular wrap."""
        cap = 50
        buf = make_buf(capacity=cap)

        # Fill with two batches (second overwrites first)
        ep1 = _make_episode(T=cap, base_reward=1.0, done_at=cap - 1)
        buf.add_episode(ep1)

        ep2 = _make_episode(T=cap, base_t_idx=1000, base_reward=100.0, done_at=cap - 1)
        buf.add_episode(ep2)

        stored   = buf.stored_rewards()
        expected = float(np.std(stored, ddof=1))
        actual   = buf.buffer_reward_std

        assert abs(actual - expected) < 1e-5

    def test_reward_std_default_before_fill(self):
        """Empty buffer (size < 2) returns default std of 1.0."""
        buf = make_buf()
        assert buf.buffer_reward_std == 1.0

        ep = _make_episode(T=1, done_at=0)
        buf.add_episode(ep)
        assert buf.buffer_reward_std == 1.0  # size=1, still default


# ===========================================================================
# Additional: warmup exclusion test
# ===========================================================================

class TestWarmupExclusion:
    """Warmup transitions excluded from sampling once ≥128 policy transitions."""

    def test_warmup_excluded_after_threshold(self):
        """
        Add 128+ policy transitions; warmup transitions must not appear
        in the sampled batch.
        """
        buf = make_buf(warmup_exclusion_threshold=10)  # low threshold for speed

        # Add 5 warmup transitions (steps 0..4)
        wu_ep = _make_episode(T=5, base_t_idx=0, done_at=4)
        buf.add_episode(wu_ep, is_warmup=True)

        # Add 10+ policy transitions (exceeds threshold=10)
        pol_ep = _make_episode(T=15, base_t_idx=100, done_at=14)
        buf.add_episode(pol_ep, is_warmup=False)

        rng   = np.random.default_rng(0)
        batch = buf.sample(200, rng=rng)
        assert not batch["warmup_flag"].any(), (
            "Warmup transitions should not be sampled after exclusion threshold"
        )

    def test_warmup_included_before_threshold(self):
        """
        Below threshold, warmup transitions can be sampled.
        """
        buf = make_buf(warmup_exclusion_threshold=128)  # high threshold

        wu_ep  = _make_episode(T=5, base_t_idx=0, done_at=4)
        buf.add_episode(wu_ep, is_warmup=True)

        pol_ep = _make_episode(T=5, base_t_idx=100, done_at=4)
        buf.add_episode(pol_ep, is_warmup=False)

        rng   = np.random.default_rng(0)
        batch = buf.sample(500, rng=rng)
        assert batch["warmup_flag"].any(), (
            "Warmup transitions should be sampled before exclusion threshold"
        )

    def test_policy_count_correct(self):
        """policy_count returns number of non-warmup transitions."""
        buf = make_buf()

        wu_ep  = _make_episode(T=49, base_t_idx=0, done_at=48)  # 49 stored
        buf.add_episode(wu_ep, is_warmup=True)

        pol_ep = _make_episode(T=20, base_t_idx=100, done_at=19)
        buf.add_episode(pol_ep, is_warmup=False)

        assert buf.policy_count == 20, (
            f"Expected policy_count=20, got {buf.policy_count}"
        )


# ===========================================================================
# Additional: add_step / flush_episode (staging API)
# ===========================================================================

class TestAddStep:
    """Step-by-step staging API computes same n-step returns as add_episode."""

    def test_add_step_matches_add_episode(self):
        """
        add_step + flush_episode must produce same R_n values as add_episode
        for a simple 8-step episode.
        """
        T = 8
        base_reward = 2.0

        ep = _make_episode(T=T, base_reward=base_reward, done_at=T - 1)

        buf_ep   = make_buf()
        buf_step = make_buf()

        buf_ep.add_episode(ep, is_warmup=False)

        for i, step in enumerate(ep):
            buf_step.add_step(
                t_idx      = step["t_idx"],
                mask_t     = step["mask_t"],
                w_pre      = step["w_pre"],
                w_exec     = step["w_exec"],
                reward     = step["reward"],
                t_idx_next = step["t_idx_next"],
                done       = step["done"],
                is_warmup  = False,
                step_idx   = i,
            )
        buf_step.flush_episode()

        np.testing.assert_allclose(
            buf_ep.stored_rewards(),
            buf_step.stored_rewards(),
            rtol=1e-5,
            err_msg="add_step R_n must match add_episode R_n",
        )

    def test_warmup_drop_via_add_step(self):
        """
        add_step with step_idx ≥ 49 for is_warmup=True must NOT be stored.
        """
        buf = make_buf()

        for i in range(WARMUP_STEPS):
            t = i
            buf.add_step(
                t_idx=t, mask_t=_make_mask(), w_pre=_make_w(), w_exec=_make_w(),
                reward=float(i), t_idx_next=t + 1,
                done=(i == WARMUP_STEPS - 1),
                is_warmup=True, step_idx=i,
            )
        buf.flush_episode()

        assert buf.size == WARMUP_STEPS - (N_STEP - 1), (
            f"add_step warmup should store {WARMUP_STEPS - (N_STEP - 1)} transitions, "
            f"got {buf.size}"
        )
        stored_t = set(buf.stored_t_idx().tolist())
        for drop_idx in [49, 50, 51]:
            assert drop_idx not in stored_t
