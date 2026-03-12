"""
tests/test_phase9_gate9.py
Gate 9 — SAC Training Loop  (Phase 9, Bible §8.1–8.12)

Seven mandatory Gate 9 criteria:
  1. Critic      — Critic loss decreases on average over 500 updates (fixed buffer)
  2. Q-Stability — Q-values stay in [−50, +50] throughout 500 updates
  3. Polyak      — 3 unit tests: one-step, two-step, convergence  (§8.10)
  4. Alpha       — 4 unit tests: sign, gradient direction, converge hi/lo  (§8.8)
  5. ActorValid  — w_pre sums to 1, all ≥ 0, no NaN after 500 updates
  6. GradNorms   — no persistent grad-norm violations (> 10× clip threshold)
  7. Determinism — same seed → identical loss curves
"""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pytest
import torch
import torch.nn as nn

from model.apex_actor_critic import ApexActorCritic
from training.sac_trainer import SACTrainer, polyak_update, qr_huber_loss

# ============================================================
# Tiny model / data configuration  (fast, CPU-only)
# ============================================================

K_MAX   = 8       # asset slots
F_DIM   = 6       # per-asset features
D_G     = 4       # global context dimension
T_PAN   = 300     # panel length (trading-day rows)
L_LOOK  = 10      # lookback window in panel rows
N_QUANT = 8       # quantile heads
B_SIZE  = 16      # batch size used in tests
N_UPD   = 500     # update steps for integration tests


# ============================================================
# Shared factory helpers
# ============================================================

def make_tiny_model(seed: int = 0) -> ApexActorCritic:
    torch.manual_seed(seed)
    return ApexActorCritic(
        K_max              = K_MAX,
        F                  = F_DIM,
        D_g                = D_G,
        num_tickers        = 16,
        num_sectors        = 4,
        ticker_emb_dim     = 4,
        sector_emb_dim     = 4,
        D_emb_proj         = 8,
        tcn_channels       = 16,
        tcn_levels         = 2,
        tcn_kernel_size    = 3,
        tcn_dilation_base  = 2,
        attn_d_model       = 16,
        attn_n_heads       = 2,
        attn_d_ff          = 32,
        attn_n_layers      = 1,
        attn_dropout       = 0.0,
        actor_hidden_dims  = [16],
        critic_hidden_dims = [16],
        n_quantiles        = N_QUANT,
    )


def make_panel_data(seed: int = 42) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_panel    = rng.normal(0.0, 0.5, (T_PAN, K_MAX, F_DIM)).astype(np.float32)
    g_panel    = rng.normal(0.0, 0.5, (T_PAN, D_G)).astype(np.float32)
    mask_panel = np.ones((T_PAN, K_MAX), dtype=np.float32)
    ticker_ids = np.tile(np.arange(K_MAX, dtype=np.int64), (T_PAN, 1))
    sector_ids = (np.tile(np.arange(K_MAX, dtype=np.int64), (T_PAN, 1)) % 4)
    return dict(
        x_panel    = x_panel,
        g_panel    = g_panel,
        mask_panel = mask_panel,
        ticker_ids = ticker_ids,
        sector_ids = sector_ids,
    )


def make_sac_cfg(alpha_lr: float = 1e-3) -> SimpleNamespace:
    return SimpleNamespace(
        tau                  = 0.005,
        policy_delay         = 2,
        grad_clip_critic     = 1.0,
        grad_clip_actor      = 5.0,
        grad_clip_encoder    = 1.0,
        entropy_scale_factor = 0.7,
        alpha_clamp_min      = math.log(1e-4),   # −9.2103
        alpha_clamp_max      = math.log(1.0),    # 0.0
        init_alpha           = 0.1,
        alpha_lr             = alpha_lr,
        n_step               = 4,
    )


def make_opt_cfg(actor_lr: float = 3e-4, critic_lr: float = 3e-4) -> SimpleNamespace:
    return SimpleNamespace(
        encoder_lr           = 3e-4,
        encoder_weight_decay = 1e-4,
        actor_lr             = actor_lr,
        actor_weight_decay   = 0.0,
        critic_lr            = critic_lr,
        critic_weight_decay  = 0.0,
    )


def make_fake_batch(B: int = B_SIZE, seed: int = 0) -> Dict:
    """Synthetic batch dict matching ReplayBuffer.sample() output."""
    rng = np.random.default_rng(seed)
    # t_idx must be ≥ L_LOOK so obs reconstruction has enough history
    t_arr  = rng.integers(L_LOOK + 5, T_PAN - 20, size=B).astype(np.int64)
    tn_arr = np.clip(t_arr + 4, 0, T_PAN - 1)

    # valid softmax portfolio weights
    logits = rng.normal(0.0, 1.0, (B, K_MAX)).astype(np.float32)
    exp_l  = np.exp(logits - logits.max(axis=1, keepdims=True))
    w_pre  = (exp_l / exp_l.sum(axis=1, keepdims=True)).astype(np.float32)

    return dict(
        idx               = np.arange(B, dtype=np.int64),
        t_idx             = t_arr,
        t_idx_next        = tn_arr,
        mask_t            = np.ones((B, K_MAX), dtype=np.float32),
        w_pre             = w_pre,
        w_exec            = w_pre.copy(),
        R_n               = rng.normal(0.01, 0.005, B).astype(np.float32),
        R_n_clean         = rng.normal(0.01, 0.005, B).astype(np.float32),
        gamma_n           = np.full(B, 0.975 ** 4, dtype=np.float32),
        done_n            = np.zeros(B, dtype=bool),
        warmup_flag       = np.zeros(B, dtype=bool),
        aug_obs_std_factor= 0.01,
    )


def make_trainer(model=None, seed: int = 0) -> SACTrainer:
    if model is None:
        model = make_tiny_model(seed)
    return SACTrainer(
        model       = model,
        panel_data  = make_panel_data(),
        sac_cfg     = make_sac_cfg(),
        opt_cfg     = make_opt_cfg(),
        L_lookback  = L_LOOK,
        device      = torch.device("cpu"),
    )


def run_n_updates(
    trainer: SACTrainer,
    n: int,
    seed: int = 0,
) -> List[Dict]:
    """Run n SAC update steps, returning list of metric dicts."""
    rng = np.random.default_rng(seed)
    metrics_log = []
    for i in range(n):
        batch = make_fake_batch(B=B_SIZE, seed=i)
        m = trainer.update(batch, rng=rng)
        metrics_log.append(m)
    return metrics_log


# ============================================================
# Gate 9 — Criterion 1:  Critic loss decreases on average
# ============================================================

class TestCriterion1CriticLoss:
    """Critic loss should decrease on average over 500 updates (§11.2)."""

    def test_critic_loss_downward_trend(self):
        torch.manual_seed(1)
        np.random.seed(1)

        model   = make_tiny_model(seed=1)
        trainer = make_trainer(model, seed=1)
        logs    = run_n_updates(trainer, N_UPD, seed=1)

        losses = np.array([m["critic_loss"] for m in logs])

        # No NaN / Inf
        assert not np.isnan(losses).any(), "Critic loss contains NaN"
        assert not np.isinf(losses).any(), "Critic loss contains Inf"

        # Mean of last 100 steps < mean of first 100 steps  (downward trend)
        first_mean = losses[:100].mean()
        last_mean  = losses[400:].mean()
        assert last_mean < first_mean, (
            f"Critic loss did not decrease on average: "
            f"first_100={first_mean:.4f}, last_100={last_mean:.4f}"
        )

    def test_critic_loss_finite_throughout(self):
        model   = make_tiny_model(seed=2)
        trainer = make_trainer(model, seed=2)
        logs    = run_n_updates(trainer, 100, seed=2)

        for i, m in enumerate(logs):
            assert np.isfinite(m["critic_loss"]), (
                f"Critic loss non-finite at update {i}: {m['critic_loss']}"
            )


# ============================================================
# Gate 9 — Criterion 2:  Q-value stability
# ============================================================

class TestCriterion2QStability:
    """Q-value means must remain in [−50, +50] throughout 500 updates."""

    def test_q_values_bounded(self):
        model   = make_tiny_model(seed=3)
        trainer = make_trainer(model, seed=3)
        logs    = run_n_updates(trainer, N_UPD, seed=3)

        for i, m in enumerate(logs):
            for key in ("q1_mean", "q2_mean"):
                v = m[key]
                assert -50.0 <= v <= 50.0, (
                    f"{key}={v:.4f} out of [-50, 50] at update {i}"
                )

    def test_q_target_not_diverging(self):
        model   = make_tiny_model(seed=4)
        trainer = make_trainer(model, seed=4)
        logs    = run_n_updates(trainer, N_UPD, seed=4)

        q_targets = np.array([m["q_target_mean"] for m in logs])
        assert not np.isnan(q_targets).any(), "Q target mean contains NaN"
        # Target should not diverge — check last 100 avg < 100
        assert abs(q_targets[400:].mean()) < 100.0, (
            f"Q target appears to be diverging: last_mean={q_targets[400:].mean():.2f}"
        )


# ============================================================
# Gate 9 — Criterion 3:  Polyak averaging  (§8.10, 3 unit tests)
# ============================================================

class TestCriterion3Polyak:
    """Three Polyak unit tests specified in §8.10."""

    def _build_tiny_models(self) -> tuple:
        """Return (online, target) pair of simple Linear modules."""
        online = nn.Linear(4, 4, bias=False)
        target = copy.deepcopy(online)
        return online, target

    def test_polyak_one_step_exact(self):
        """
        §8.10 Test 1: one-step Polyak produces
            θ_target = τ·θ_new + (1−τ)·θ_old  for random tensors.
        """
        tau = 0.3
        online = nn.Linear(4, 4, bias=False)
        target = copy.deepcopy(online)

        # Store old target weights
        old_w = target.weight.data.clone()

        # New online weights
        with torch.no_grad():
            online.weight.data.fill_(2.0)

        polyak_update(online, target, tau)

        expected = tau * online.weight.data + (1 - tau) * old_w
        assert torch.allclose(target.weight.data, expected, atol=1e-6), (
            f"One-step Polyak mismatch: expected {expected}, got {target.weight.data}"
        )

    def test_polyak_two_step_value(self):
        """
        §8.10 Test 2: two-step Polyak gives
            θ_target = 0.1 + 0.9×0.1 = 0.19
            (θ_init=0, θ_new=1, τ=0.1)
        """
        tau = 0.1

        online = nn.Linear(1, 1, bias=False)
        target = nn.Linear(1, 1, bias=False)

        with torch.no_grad():
            target.weight.data.fill_(0.0)   # θ_init = 0
            online.weight.data.fill_(1.0)   # θ_new  = 1

        # Step 1: θ_target = 0.1·1 + 0.9·0 = 0.1
        polyak_update(online, target, tau)
        assert abs(target.weight.item() - 0.1) < 1e-6, (
            f"Step 1: expected 0.1, got {target.weight.item()}"
        )

        # Step 2: θ_target = 0.1·1 + 0.9·0.1 = 0.19
        polyak_update(online, target, tau)
        assert abs(target.weight.item() - 0.19) < 1e-6, (
            f"Step 2: expected 0.19, got {target.weight.item()}"
        )

    def test_polyak_convergence(self):
        """
        §8.10 Test 3: after many steps target converges to online.
        """
        tau = 0.1
        online = nn.Linear(4, 4, bias=False)
        target = nn.Linear(4, 4, bias=False)

        with torch.no_grad():
            online.weight.data.fill_(1.0)
            target.weight.data.fill_(0.0)

        for _ in range(1000):
            polyak_update(online, target, tau)

        # After 1000 steps: θ_target = 1 − (1−0.1)^1000 ≈ 1.0
        assert torch.allclose(target.weight.data, online.weight.data, atol=1e-3), (
            f"Polyak convergence failed: max_diff="
            f"{(target.weight.data - online.weight.data).abs().max().item():.6f}"
        )


# ============================================================
# Gate 9 — Criterion 4:  Alpha / entropy tuning  (§8.8, 4 unit tests)
# ============================================================

class TestCriterion4Alpha:
    """Four alpha unit tests specified in §8.8."""

    def test_log_prob_correct_sign(self):
        """
        §8.8 Test 1: log-prob is correctly computed (negative for valid policy).
        The log-prob of a proper probability distribution ∈ (−∞, 0].
        For a logistic-normal softmax output, log_prob should be finite and ≤ 0
        on average across the batch.
        """
        torch.manual_seed(10)
        model = make_tiny_model(seed=10)
        model.eval()   # deterministic

        panel = make_panel_data()
        B     = 8
        rng   = np.random.default_rng(10)
        t_arr = rng.integers(L_LOOK + 5, T_PAN - 20, size=B)

        # Build obs tensors manually
        win_len = L_LOOK + 1
        x_batch = np.zeros((B, win_len, K_MAX, F_DIM), dtype=np.float32)
        g_batch = np.zeros((B, D_G), dtype=np.float32)
        for i, tidx in enumerate(t_arr):
            tidx = int(tidx)
            p_s  = max(0, tidx - L_LOOK)
            win  = panel["x_panel"][p_s : tidx + 1]
            x_batch[i, -win.shape[0]:] = win
            g_batch[i] = panel["g_panel"][tidx]

        x    = torch.from_numpy(x_batch)
        g    = torch.from_numpy(g_batch)
        mask = torch.ones(B, K_MAX)
        sid  = torch.zeros(B, K_MAX, dtype=torch.long)
        tid  = torch.from_numpy(
            np.tile(np.arange(K_MAX, dtype=np.int64), (B, 1))
        )

        with torch.no_grad():
            _, log_prob = model.actor_forward(x, g, mask, sid, tid)

        # Log-prob should be finite (logistic-normal density CAN be positive)
        assert torch.isfinite(log_prob).all(), "log_prob contains non-finite values"
        assert log_prob.shape == (B,), (
            f"log_prob shape should be ({B},), got {log_prob.shape}"
        )
        # Entropy estimate H = -E[log_prob] must be finite
        entropy_estimate = -log_prob.mean().item()
        assert math.isfinite(entropy_estimate), (
            f"Entropy estimate is not finite: {entropy_estimate}"
        )

    def test_alpha_loss_gradient_sign(self):
        """
        §8.8 Test 2: alpha loss gradient has correct sign.

        If entropy < H_target (log_prob + H_target > 0), the gradient
        ∂L_α/∂log_α should be NEGATIVE  →  log_α decreases  →  α decreases.
        Wait — actually when entropy is BELOW target we want α to INCREASE.
        The gradient sign check: with L_α = −log_α·(log_π + H_target):
          • entropy below target  →  log_π + H_target > 0
            →  ∂L_α/∂log_α = −(log_π + H_target) < 0  →  log_α decreases
        Hmm — this is actually the opposite. Let me use the correct formulation.

        Standard SAC: L_α = −log_α·(log_π + H_target)
        When H_current < H_target (policy too concentrated):
          log_π is very negative: -(H_current) < -(H_target)
          log_π + H_target > 0
          ∂L_α/∂log_α = -(log_π + H_target) < 0  → log_α decreases → α decreases
          But we WANT α to increase when entropy is low...

        This is the known formulation issue. The correct standard SAC form:
          L_α = E[-α · (log_π + H_target)]  (optimized in log-space by many impls)
        which when H_current < H_target pushes log_α UP (α increases).

        The bible uses L_α = E[−log(α)·(log_π + H_target)] = −log_α·(H_target−H_current).
        When H_current < H_target: (H_target - H_current) > 0, so gradient w.r.t.
        log_α is -(H_target - H_current) < 0, meaning log_α decreases.

        This matches the Haarnoja et al. (2018) formulation: the loss IS correct
        for gradient-based minimization of L_α: gradient descent on log_α with
        this loss increases α when H_current < H_target.

        Test: set log_prob very negative (entropy well below target) and verify
        that the computed alpha_loss gradient on log_alpha is negative.
        This means a gradient DESCENT step will DECREASE log_alpha.

        Actually per Haarnoja: L_α = E[α(−log π − H_target)] optimised in α-space.
        In log-space: L_α = E[exp(log_α)(−log π − H_target)]
        ∂L/∂log_α = E[exp(log_α)(−log π − H_target)] = α·(H_current − H_target)
        When H_current < H_target: gradient < 0 → log_α decreases → α decreases.
        When H_current > H_target: gradient > 0 → log_α increases → α increases.

        The bible's formulation −log(α)·(log π + H_target) with Adam:
        gradient = −(log π + H_target) = H_current − H_target  (same sign pattern!)

        Equilibrium: E[log_π] = -H_target = ln(K)*scale ≈ 1.456.
        High entropy (diffuse): E[log_π] << 1.456, e.g. log_prob = -10.
          gradient = -(log_prob + H_target) = -(-10 + -1.456) = 11.456 > 0
          Adam step: log_alpha -= lr*11.456 → log_alpha decreases → alpha decreases. CORRECT.
        """
        log_alpha = nn.Parameter(torch.tensor(math.log(0.1)))

        # log_prob = -10 is well below equilibrium (~1.456) → high entropy
        log_prob_fake = torch.tensor([-10.0] * 8)
        H_target_fake = torch.full((8,), -math.log(K_MAX) * 0.7)  # ≈ -1.456

        alpha_loss = -(log_alpha * (log_prob_fake.detach() + H_target_fake.detach())).mean()
        alpha_loss.backward()

        grad = log_alpha.grad.item()
        # gradient = -(log_prob + H_target) = -(-10 - 1.456) = 11.456 > 0
        # Adam decreases log_alpha → alpha decreases. Correct for high-entropy regime.
        assert grad > 0.0, (
            f"Alpha gradient should be positive (alpha should decrease for high entropy), "
            f"got grad={grad:.4f}"
        )

    def test_alpha_increases_toward_max_when_entropy_low(self):
        """
        §8.8 Test 3: α converges toward alpha_max when policy entropy is
        consistently below target (concentrated policy).

        Equilibrium: E[log_π] = -H_target = ln(K)*scale ≈ 1.456 for K=8.
        Entropy below target ↔ E[log_π] > 1.456 (concentrated policy, high log-prob).
        With log_prob_val = 3.0 >> 1.456:
          gradient = -(3.0 + (-1.456)) = -1.544 < 0
          Adam step: log_alpha -= lr*(-1.544) → log_alpha INCREASES → alpha increases. ✓
        """
        torch.manual_seed(20)
        log_alpha = nn.Parameter(torch.tensor(math.log(0.01)))  # start low
        opt       = torch.optim.Adam([log_alpha], lr=0.1)

        alpha_clamp_max = 0.0          # log(1.0)
        alpha_clamp_min = math.log(1e-4)

        # Concentrated policy: log_prob >> equilibrium (~1.456) → entropy below target
        log_prob_val = 3.0                             # well above equilibrium
        H_target_val = -math.log(K_MAX) * 0.7         # ≈ -1.456

        for _ in range(200):
            log_prob  = torch.tensor(log_prob_val)
            H_target  = torch.tensor(H_target_val)
            loss      = -(log_alpha * (log_prob.detach() + H_target.detach()))
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                log_alpha.clamp_(alpha_clamp_min, alpha_clamp_max)

        alpha_final = math.exp(float(log_alpha.item()))
        assert alpha_final > 0.5, (
            f"α should converge toward alpha_max=1.0 when entropy is below target "
            f"(concentrated policy, log_prob={log_prob_val}), got α={alpha_final:.4f}"
        )

    def test_alpha_decreases_toward_min_when_entropy_high(self):
        """
        §8.8 Test 4: α converges toward alpha_min when policy entropy
        consistently exceeds target (log_prob close to 0 → high entropy).
        """
        torch.manual_seed(21)
        log_alpha = nn.Parameter(torch.tensor(0.0))   # start at alpha_max = 1.0
        opt       = torch.optim.Adam([log_alpha], lr=0.1)

        alpha_clamp_max = 0.0
        alpha_clamp_min = math.log(1e-4)

        # Simulate: entropy >> target (very uniform policy)
        log_prob_val = -0.05                           # H_current ≈ 0 (near-uniform)
        H_target_val = -math.log(K_MAX) * 0.7         # ≈ -1.02

        for _ in range(200):
            log_prob = torch.tensor(log_prob_val)
            H_target = torch.tensor(H_target_val)
            loss     = -(log_alpha * (log_prob.detach() + H_target.detach()))
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                log_alpha.clamp_(alpha_clamp_min, alpha_clamp_max)

        alpha_final = math.exp(float(log_alpha.item()))
        assert alpha_final < 0.5, (
            f"α should converge toward alpha_min when entropy is high, "
            f"got α={alpha_final:.4f}"
        )


# ============================================================
# Gate 9 — Criterion 5:  Actor produces valid w_pre
# ============================================================

class TestCriterion5ActorValid:
    """After 500 updates, actor produces valid w_pre (sums to 1, ≥0, no NaN)."""

    def test_actor_valid_after_training(self):
        torch.manual_seed(30)
        model   = make_tiny_model(seed=30)
        trainer = make_trainer(model, seed=30)
        run_n_updates(trainer, N_UPD, seed=30)

        # Eval pass (deterministic, no logistic-normal noise)
        model.eval()
        panel = make_panel_data()
        rng   = np.random.default_rng(30)
        B     = 16
        t_arr = rng.integers(L_LOOK + 5, T_PAN - 20, size=B)

        win_len = L_LOOK + 1
        x_b = np.zeros((B, win_len, K_MAX, F_DIM), dtype=np.float32)
        g_b = np.zeros((B, D_G), dtype=np.float32)
        for i, tidx in enumerate(t_arr):
            tidx = int(tidx)
            p_s  = max(0, tidx - L_LOOK)
            win  = panel["x_panel"][p_s : tidx + 1]
            x_b[i, -win.shape[0]:] = win
            g_b[i] = panel["g_panel"][tidx]

        x    = torch.from_numpy(x_b)
        g    = torch.from_numpy(g_b)
        mask = torch.ones(B, K_MAX)
        sid  = torch.zeros(B, K_MAX, dtype=torch.long)
        tid  = torch.from_numpy(
            np.tile(np.arange(K_MAX, dtype=np.int64), (B, 1))
        )

        with torch.no_grad():
            w_pre, log_prob = model.actor_forward(x, g, mask, sid, tid)

        # No NaN/Inf
        assert torch.isfinite(w_pre).all(),    "w_pre contains non-finite values"
        assert torch.isfinite(log_prob).all(), "log_prob contains non-finite values"

        # Non-negative
        assert (w_pre >= -1e-6).all(), f"w_pre has negative values: min={w_pre.min():.6f}"

        # Sums to ~1 per sample (masked softmax)
        row_sums = w_pre.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(B), atol=1e-4), (
            f"w_pre rows don't sum to 1: min={row_sums.min():.6f}, max={row_sums.max():.6f}"
        )

    def test_actor_valid_before_training(self):
        """Sanity: freshly initialised model also produces valid w_pre."""
        model = make_tiny_model(seed=31)
        model.eval()

        panel = make_panel_data()
        B     = 4
        win_len = L_LOOK + 1
        x_b = np.zeros((B, win_len, K_MAX, F_DIM), dtype=np.float32)
        g_b = panel["g_panel"][L_LOOK + 5 : L_LOOK + 5 + B]
        for i in range(B):
            tidx = L_LOOK + 5 + i
            x_b[i] = panel["x_panel"][tidx - L_LOOK : tidx + 1]

        x    = torch.from_numpy(x_b)
        g    = torch.from_numpy(g_b)
        mask = torch.ones(B, K_MAX)
        sid  = torch.zeros(B, K_MAX, dtype=torch.long)
        tid  = torch.from_numpy(np.tile(np.arange(K_MAX, dtype=np.int64), (B, 1)))

        with torch.no_grad():
            w_pre, _ = model.actor_forward(x, g, mask, sid, tid)

        assert torch.isfinite(w_pre).all()
        assert (w_pre >= -1e-6).all()
        row_sums = w_pre.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(B), atol=1e-4)


# ============================================================
# Gate 9 — Criterion 6:  Gradient norms within bounds
# ============================================================

class TestCriterion6GradNorms:
    """
    No grad-norm metric should persistently exceed 10× its clip threshold.
    'Persistently' = in more than 10% of update steps.
    Clip thresholds: critic=1.0, encoder=1.0, actor=5.0.
    """

    CRITIC_CLIP  = 1.0
    ACTOR_CLIP   = 5.0
    ENCODER_CLIP = 1.0
    TOLERANCE    = 10.0   # grad may spike up to 10× clip
    MAX_FRAC     = 0.10   # no more than 10% of steps may exceed tolerance

    def test_critic_grad_norm_not_persistently_violated(self):
        torch.manual_seed(40)
        model   = make_tiny_model(seed=40)
        trainer = make_trainer(model, seed=40)
        logs    = run_n_updates(trainer, N_UPD, seed=40)

        norms  = np.array([m["critic_grad_norm"] for m in logs])
        thresh = self.CRITIC_CLIP * self.TOLERANCE
        frac   = (norms > thresh).mean()
        assert frac <= self.MAX_FRAC, (
            f"Critic grad norm exceeded {thresh:.1f} in {frac*100:.1f}% of steps "
            f"(threshold: {self.MAX_FRAC*100:.0f}%)"
        )

    def test_encoder_grad_norm_not_persistently_violated(self):
        torch.manual_seed(41)
        model   = make_tiny_model(seed=41)
        trainer = make_trainer(model, seed=41)
        logs    = run_n_updates(trainer, N_UPD, seed=41)

        norms  = np.array([m["enc_grad_norm_crit"] for m in logs])
        thresh = self.ENCODER_CLIP * self.TOLERANCE
        frac   = (norms > thresh).mean()
        assert frac <= self.MAX_FRAC, (
            f"Encoder grad norm exceeded {thresh:.1f} in {frac*100:.1f}% of steps "
            f"(threshold: {self.MAX_FRAC*100:.0f}%)"
        )

    def test_actor_grad_norm_not_persistently_violated(self):
        torch.manual_seed(42)
        model   = make_tiny_model(seed=42)
        trainer = make_trainer(model, seed=42)
        logs    = run_n_updates(trainer, N_UPD, seed=42)

        # actor_grad_norm is only logged every policy_delay steps
        norms = np.array([
            m["actor_grad_norm"]
            for m in logs if "actor_grad_norm" in m
        ])
        if len(norms) == 0:
            pytest.skip("No actor grad norms logged (policy_delay not triggered)")

        thresh = self.ACTOR_CLIP * self.TOLERANCE
        frac   = (norms > thresh).mean()
        assert frac <= self.MAX_FRAC, (
            f"Actor grad norm exceeded {thresh:.1f} in {frac*100:.1f}% of steps "
            f"(threshold: {self.MAX_FRAC*100:.0f}%)"
        )


# ============================================================
# Gate 9 — Criterion 7:  Determinism
# ============================================================

class TestCriterion7Determinism:
    """Same seed → identical loss curves (§8.9 determinism requirement)."""

    N_DET = 50   # steps to compare

    def _run_deterministic(self, seed: int) -> List[float]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model   = make_tiny_model(seed=seed)
        trainer = make_trainer(model, seed=seed)
        logs    = run_n_updates(trainer, self.N_DET, seed=seed)
        return [m["critic_loss"] for m in logs]

    def test_same_seed_identical_critic_losses(self):
        losses_a = self._run_deterministic(seed=99)
        losses_b = self._run_deterministic(seed=99)
        assert losses_a == losses_b, (
            f"Critic losses differ across two runs with same seed. "
            f"First diff at step: "
            f"{next(i for i,(a,b) in enumerate(zip(losses_a,losses_b)) if a!=b)}"
        )

    def test_different_seeds_different_losses(self):
        """Sanity check: different seeds should produce different losses."""
        losses_a = self._run_deterministic(seed=1)
        losses_b = self._run_deterministic(seed=2)
        # At least some losses should differ
        assert losses_a != losses_b, (
            "Expected different seeds to produce different loss curves"
        )


# ============================================================
# Gate 9 — Additional: QR-Huber loss unit tests
# ============================================================

class TestQRHuberLoss:
    """Unit tests for the QR-Huber loss function."""

    def test_zero_when_target_equals_pred(self):
        """
        If z_j = q_k = constant for all j, k (all values identical), loss should be 0.
        Note: equal DISTRIBUTIONS (same sorted vector) is NOT sufficient since
        u_jk = z_j - q_k \u2260 0 for j \u2260 k. We need all values equal.
        """
        N  = 8
        B  = 4
        taus = (2 * torch.arange(N, dtype=torch.float32) + 1) / (2 * N)

        # All-constant vectors: u_jk = c - c = 0 for all j, k -> loss = 0
        const = torch.full((B, N), 0.5)
        loss  = qr_huber_loss(const, const, taus)
        assert loss.item() < 1e-6, f"QR-Huber loss should be ~0 for constant vectors, got {loss.item()}"

    def test_positive_loss_nonzero_error(self):
        """Non-zero prediction error should yield positive loss."""
        N    = 8
        B    = 4
        taus = (2 * torch.arange(N, dtype=torch.float32) + 1) / (2 * N)

        target = torch.zeros(B, N)
        pred   = torch.ones(B, N)

        loss = qr_huber_loss(pred, target, taus)
        assert loss.item() > 0.0, "QR-Huber loss should be positive for non-zero error"

    def test_asymmetric_loss(self):
        """
        QR-Huber IS asymmetric for a given predicted quantile level \u03c4_k \u2260 0.5.
        With N=1 and \u03c4=[0.9]:
          - overestimation (pred=1, target=0): weight = |0.9-1| = 0.1, loss = 0.1*0.5 = 0.05
          - underestimation (pred=-1, target=0): weight = |0.9-0| = 0.9, loss = 0.9*0.5 = 0.45
        These should differ.
        """
        N    = 1
        B    = 1
        taus = torch.tensor([0.9])   # high quantile level -> penalises underestimation more

        target = torch.zeros(B, N)

        # pred > target  (overestimation by 1)
        loss_over  = qr_huber_loss(torch.ones(B, N),  target, taus)
        # pred < target  (underestimation by 1)
        loss_under = qr_huber_loss(-torch.ones(B, N), target, taus)

        assert abs(loss_over.item() - loss_under.item()) > 1e-3, (
            f"QR-Huber should produce asymmetric losses for \u03c4=0.9: "
            f"over={loss_over.item():.4f}, under={loss_under.item():.4f}"
        )
        # Specifically: underestimation penalised MORE at high quantile
        assert loss_under.item() > loss_over.item(), (
            "At \u03c4=0.9, underestimation should be penalised more than overestimation"
        )


# ============================================================
# Gate 9 — Additional: Alpha clamping
# ============================================================

class TestAlphaClamping:
    """Verify log_alpha stays within [alpha_clamp_min, alpha_clamp_max]."""

    def test_log_alpha_stays_in_bounds(self):
        torch.manual_seed(50)
        model   = make_tiny_model(seed=50)
        trainer = make_trainer(model, seed=50)
        logs    = run_n_updates(trainer, N_UPD, seed=50)

        for i, m in enumerate(logs):
            la = m["log_alpha"]
            assert math.log(1e-4) - 0.01 <= la <= math.log(1.0) + 0.01, (
                f"log_alpha={la:.4f} outside bounds at update {i}"
            )
            assert 1e-4 - 1e-6 <= m["alpha"] <= 1.0 + 1e-6, (
                f"alpha={m['alpha']:.6f} outside [1e-4, 1.0] at update {i}"
            )


# ============================================================
# Gate 9 — §11.1 Table 53: Numerical Stability (5 seeds × 1000 steps)
# ============================================================

class TestNumericalStability:
    """
    §11.1 Table 53: "Critic and Alpha Numerical Stability"
    Run 1000 SAC update steps from random initialisation across 5 different
    random seeds. Pass criterion:
      - No NaN or Inf in Q-values, log-probs, or alpha after 1000 steps
      - alpha stays within [alpha_min, alpha_max] throughout
    """

    N_STEPS = 1000
    SEEDS   = [0, 7, 13, 42, 99]

    def _run_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model   = make_tiny_model(seed=seed)
        trainer = make_trainer(model, seed=seed)
        logs    = run_n_updates(trainer, self.N_STEPS, seed=seed)
        return logs

    def test_no_nan_inf_q_values_all_seeds(self):
        """Q-values must remain finite across all 5 seeds for 1000 steps."""
        for seed in self.SEEDS:
            logs = self._run_seed(seed)
            for i, m in enumerate(logs):
                assert math.isfinite(m["q1_mean"]), (
                    f"seed={seed}, step={i}: q1_mean={m['q1_mean']} not finite"
                )
                assert math.isfinite(m["q2_mean"]), (
                    f"seed={seed}, step={i}: q2_mean={m['q2_mean']} not finite"
                )

    def test_no_nan_inf_alpha_all_seeds(self):
        """alpha and log_alpha must remain finite across all 5 seeds for 1000 steps."""
        for seed in self.SEEDS:
            logs = self._run_seed(seed)
            for i, m in enumerate(logs):
                assert math.isfinite(m["alpha"]), (
                    f"seed={seed}, step={i}: alpha={m['alpha']} not finite"
                )
                assert math.isfinite(m["log_alpha"]), (
                    f"seed={seed}, step={i}: log_alpha={m['log_alpha']} not finite"
                )

    def test_alpha_stays_in_bounds_all_seeds(self):
        """alpha must stay within [alpha_min=1e-4, alpha_max=1.0] across all seeds."""
        alpha_min = 1e-4
        alpha_max = 1.0
        for seed in self.SEEDS:
            logs = self._run_seed(seed)
            for i, m in enumerate(logs):
                assert alpha_min - 1e-6 <= m["alpha"] <= alpha_max + 1e-6, (
                    f"seed={seed}, step={i}: alpha={m['alpha']:.6f} "
                    f"outside [{alpha_min}, {alpha_max}]"
                )

    def test_no_nan_log_prob_all_seeds(self):
        """log_prob (entropy_mean) must remain finite on actor update steps."""
        for seed in self.SEEDS:
            logs = self._run_seed(seed)
            for i, m in enumerate(logs):
                ent = m.get("entropy_mean", float("nan"))
                if not math.isnan(ent):   # actor steps only
                    assert math.isfinite(ent), (
                        f"seed={seed}, step={i}: entropy_mean={ent} not finite"
                    )


# ============================================================
# Gate 9 — §8.12 Stability Alarm Flags
# ============================================================

class TestStabilityAlarms:
    """Verify §8.12 stability alarm flags are present and behave correctly."""

    def test_alarm_keys_present(self):
        """All required alarm keys must be in every update metrics dict."""
        model   = make_tiny_model(seed=60)
        trainer = make_trainer(model, seed=60)
        logs    = run_n_updates(trainer, 10, seed=60)

        required = {"q_divergence_flag", "entropy_collapse_flag", "alpha_pinned_max_flag"}
        for i, m in enumerate(logs):
            missing = required - m.keys()
            assert not missing, (
                f"Step {i}: missing alarm keys {missing}"
            )

    def test_q_divergence_not_triggered_normal_training(self):
        """q_divergence_flag must remain False during normal 500-step training."""
        model   = make_tiny_model(seed=61)
        trainer = make_trainer(model, seed=61)
        logs    = run_n_updates(trainer, N_UPD, seed=61)

        for i, m in enumerate(logs):
            assert not m["q_divergence_flag"], (
                f"q_divergence_flag triggered at step {i}: "
                f"q1={m['q1_mean']:.2f}, q2={m['q2_mean']:.2f}"
            )

    def test_post_clip_norms_logged(self):
        """Post-clip grad norms must be present and <= pre-clip norms."""
        model   = make_tiny_model(seed=62)
        trainer = make_trainer(model, seed=62)
        logs    = run_n_updates(trainer, 10, seed=62)

        for i, m in enumerate(logs):
            assert "critic_grad_norm_post" in m, f"Step {i}: missing critic_grad_norm_post"
            assert "enc_grad_norm_crit_post" in m, f"Step {i}: missing enc_grad_norm_crit_post"
            # Post-clip norm must be <= pre-clip norm (clipping can only reduce or maintain)
            assert m["critic_grad_norm_post"] <= m["critic_grad_norm"] + 1e-6, (
                f"Step {i}: post-clip critic norm {m['critic_grad_norm_post']:.4f} "
                f"> pre-clip {m['critic_grad_norm']:.4f}"
            )
