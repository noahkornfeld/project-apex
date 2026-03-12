"""
tests/test_phase7_gate7.py
Gate 7 — Model Architecture  (Phase 7, Bible §7)

Seven mandatory Gate 7 criteria:
  1. Shapes   — forward pass with B=4, L=12, K_max=120, F=dim → correct shapes
  2. Mask prop— gradient w.r.t. inactive-slot features is exactly zero
  3. Attention— attention weights for masked slots are exactly zero
  4. Noise    — actor produces different w_pre for same input with different seeds
  5. Quantiles— critic outputs N_quantiles values in plausible range
  6. TCN RF   — receptive field formula matches / exceeds L  (analytic)
  7. No NaN   — no NaN/Inf in any output tensor for 100 random inputs
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import pytest
import numpy as np

from model.tcn import CausalTCN
from model.attention import CrossAssetAttention
from model.apex_actor_critic import ApexActorCritic

# ---------------------------------------------------------------------------
# Shared small config used across all tests  (fast to instantiate/run)
# ---------------------------------------------------------------------------
B       = 4
L       = 12
K       = 20          # K_max for tests  (subset of real 110/120)
F       = 10          # reduced F for speed
D_G     = 8
N_Q     = 32

NUM_TICKERS = 64
NUM_SECTORS = 8


def make_model(**overrides) -> ApexActorCritic:
    kw = dict(
        K_max             = K,
        F                 = F,
        D_g               = D_G,
        num_tickers       = NUM_TICKERS,
        num_sectors       = NUM_SECTORS,
        ticker_emb_dim    = 16,
        sector_emb_dim    = 8,
        D_emb_proj        = 16,
        tcn_channels      = 32,
        tcn_levels        = 5,
        tcn_kernel_size   = 3,
        tcn_dilation_base = 2,
        attn_d_model      = 32,
        attn_n_heads      = 4,
        attn_d_ff         = 64,
        attn_n_layers     = 2,
        attn_dropout      = 0.0,        # deterministic for shape/gradient tests
        actor_hidden_dims = [32, 32],
        critic_hidden_dims= [64, 64],
        n_quantiles       = N_Q,
        log_sigma_init    = -1.5,
    )
    kw.update(overrides)
    return ApexActorCritic(**kw)


def make_batch(seed: int = 0, n_active: int = 12):
    """Return a deterministic batch of input tensors."""
    rng = torch.Generator()
    rng.manual_seed(seed)

    x          = torch.randn(B, L, K, F,        generator=rng)
    g          = torch.randn(B, D_G,             generator=rng)
    mask       = torch.zeros(B, K)
    mask[:, :n_active] = 1.0                         # first n_active slots active
    sector_ids = torch.randint(0, NUM_SECTORS, (B, K), generator=rng)
    ticker_ids = torch.randint(0, NUM_TICKERS, (B, K), generator=rng)
    ticker_ids[:, n_active:] = -1                    # inactive slots → −1

    return x, g, mask, sector_ids, ticker_ids


# ===========================================================================
# Gate 7.1  Shape test
# ===========================================================================

class TestShapes:
    """Forward pass with B=4, L=12, K_max=120, F=dim → correct output shapes."""

    def test_actor_output_shapes(self):
        """w_pre [B,K] and log_prob [B] have correct shapes."""
        model = make_model()
        model.eval()
        x, g, mask, sids, tids = make_batch()
        with torch.no_grad():
            w_pre, log_prob, q1, q2 = model(x, g, mask, sids, tids)

        assert w_pre.shape   == (B, K),  f"w_pre shape {w_pre.shape}"
        assert log_prob.shape == (B,),    f"log_prob shape {log_prob.shape}"
        assert q1 is None
        assert q2 is None

    def test_critic_output_shapes(self):
        """q1, q2 both [B, N_quantiles] when w_pre_in is provided."""
        model = make_model()
        model.eval()
        x, g, mask, sids, tids = make_batch()
        w_in = torch.rand(B, K)
        w_in = w_in / w_in.sum(dim=-1, keepdim=True)

        with torch.no_grad():
            w_pre, log_prob, q1, q2 = model(x, g, mask, sids, tids, w_pre_in=w_in)

        assert q1.shape == (B, N_Q), f"q1 shape {q1.shape}"
        assert q2.shape == (B, N_Q), f"q2 shape {q2.shape}"

    def test_w_pre_sums_to_one_over_active(self):
        """w_pre must sum to ≈1 for each sample in the batch."""
        model = make_model()
        model.eval()
        x, g, mask, sids, tids = make_batch()
        with torch.no_grad():
            w_pre, _, _, _ = model(x, g, mask, sids, tids)

        sums = w_pre.sum(dim=-1)
        for b in range(B):
            assert abs(sums[b].item() - 1.0) < 1e-4, (
                f"Sample {b}: w_pre sums to {sums[b].item():.6f}, expected 1.0"
            )

    def test_w_pre_zero_for_inactive_slots(self):
        """Inactive slots (mask==0) must receive w_pre == 0."""
        model = make_model()
        model.eval()
        x, g, mask, sids, tids = make_batch(n_active=5)
        with torch.no_grad():
            w_pre, _, _, _ = model(x, g, mask, sids, tids)

        inactive = mask < 0.5
        assert w_pre[inactive].abs().max().item() < 1e-5, (
            "Inactive slots must have w_pre ≈ 0"
        )

    def test_full_forward_with_config_dimensions(self):
        """
        Smoke test with K_max=120 matching the Gate spec (B=4, L=12, K_max=120).
        Verifies no shape/dimension errors with full-spec K_max.
        """
        model = ApexActorCritic(
            K_max=120, F=F, D_g=D_G,
            num_tickers=NUM_TICKERS, num_sectors=NUM_SECTORS,
            ticker_emb_dim=16, sector_emb_dim=8, D_emb_proj=16,
            tcn_channels=32, tcn_levels=5, tcn_kernel_size=3, tcn_dilation_base=2,
            attn_d_model=32, attn_n_heads=4, attn_d_ff=64, attn_n_layers=2,
            attn_dropout=0.0,
            actor_hidden_dims=[32, 32], critic_hidden_dims=[64, 64],
            n_quantiles=N_Q,
        )
        model.eval()
        B2, K2 = 4, 120
        x2    = torch.randn(B2, L, K2, F)
        g2    = torch.randn(B2, D_G)
        mask2 = torch.ones(B2, K2)
        mask2[:, 80:] = 0.0
        sids2 = torch.randint(0, NUM_SECTORS, (B2, K2))
        tids2 = torch.randint(0, NUM_TICKERS, (B2, K2))

        with torch.no_grad():
            w_pre, lp, _, _ = model(x2, g2, mask2, sids2, tids2)

        assert w_pre.shape == (B2, K2)
        assert lp.shape    == (B2,)


# ===========================================================================
# Gate 7.2  Mask propagation test
# ===========================================================================

class TestMaskPropagation:
    """Gradient w.r.t. inactive-slot features must be exactly zero."""

    def _get_grad(self, model, x, g, mask, sids, tids):
        """Returns dx gradient for a scalar loss on w_pre.

        We use a random projection of w_pre to avoid the constant-sum trap:
        d(sum(softmax))/dlogits = 0 always, so w_pre.sum() gives zero grad.
        A random linear combination has nonzero gradient for generic weights.
        """
        x = x.detach().requires_grad_(True)
        model.train()
        torch.manual_seed(42)  # fix projection seed for reproducibility
        w_pre, log_prob, _, _ = model(x, g, mask, sids, tids)
        proj = torch.randn_like(w_pre)  # random projection → non-zero grad
        loss = (w_pre * proj).sum()
        loss.backward()
        return x.grad

    def test_inactive_slot_gradient_is_zero(self):
        """
        For inactive slot k (mask[:,k]=0), d(loss)/d(x[:,:,k,:]) must be zero.
        Tested for several inactive slots.
        """
        torch.manual_seed(1)
        model = make_model()
        model.train()
        x, g, mask, sids, tids = make_batch(seed=10, n_active=8)

        grad = self._get_grad(model, x, g, mask, sids, tids)
        # grad: [B, L, K, F]

        # All inactive slots k = 8..K-1 must have zero gradient
        for k in range(8, K):
            g_k = grad[:, :, k, :]
            assert g_k.abs().max().item() == pytest.approx(0.0, abs=1e-8), (
                f"Inactive slot {k}: max |grad| = {g_k.abs().max().item():.2e}"
            )

    def test_active_slot_gradient_nonzero(self):
        """Active slots should generally have non-zero gradient."""
        torch.manual_seed(2)
        model = make_model()
        model.train()
        x, g, mask, sids, tids = make_batch(seed=20, n_active=8)

        grad = self._get_grad(model, x, g, mask, sids, tids)

        # At least one active slot must have non-zero gradient
        active_grad_norms = [
            grad[:, :, k, :].abs().max().item() for k in range(8)
        ]
        assert max(active_grad_norms) > 1e-8, (
            "All active slot gradients are zero — likely a bug in the encoder"
        )


# ===========================================================================
# Gate 7.3  Attention mask test
# ===========================================================================

class TestAttentionMask:
    """Attention weights for masked (inactive) key slots must be exactly zero."""

    def test_attention_weights_zero_for_inactive_keys(self):
        """
        After a forward pass, the attention weight from ANY query to an
        inactive key must be ≈ 0 (exp(−∞) = 0).
        """
        model = make_model()
        model.eval()
        n_active = 6
        x, g, mask, sids, tids = make_batch(n_active=n_active)

        with torch.no_grad():
            model(x, g, mask, sids, tids)

        # Provide w_pre_in so critic branches (q1/q2 attn) also execute
        w_in = torch.ones(B, K) * mask / mask.sum(-1, keepdim=True).clamp(min=1)
        with torch.no_grad():
            model(x, g, mask, sids, tids, w_pre_in=w_in)

        for name, stack in [
            ("actor", model.actor_attn),
            ("q1",    model.q1_attn),
            ("q2",    model.q2_attn),
        ]:
            w = stack._last_attn_weights   # [B, K, K]
            assert w is not None, f"{name}: _last_attn_weights not set"

            # Columns corresponding to inactive keys must be 0
            for k in range(n_active, K):
                col = w[:, :, k]   # [B, K] — attention TO inactive slot k
                assert col.abs().max().item() < 1e-5, (
                    f"{name}: attn weight to inactive slot {k} = "
                    f"{col.abs().max().item():.2e} (expected 0)"
                )

    def test_attention_weights_positive_for_active_keys(self):
        """
        At least some attention weight must be directed to active key slots.
        """
        model = make_model()
        model.eval()
        x, g, mask, sids, tids = make_batch(n_active=8)

        with torch.no_grad():
            model(x, g, mask, sids, tids)

        w = model.actor_attn._last_attn_weights  # [B, K, K]
        active_weights = w[:, :, :8]
        assert active_weights.sum().item() > 0.0, (
            "Attention weights to active slots sum to 0 — masking bug"
        )

    def test_attention_rows_sum_to_one_over_active(self):
        """
        Each active query row of the attention matrix should sum to ≈ 1.0
        (all weight allocated among active keys; inactive keys get 0).
        """
        model = make_model()
        model.eval()
        n_active = 8
        x, g, mask, sids, tids = make_batch(n_active=n_active)

        with torch.no_grad():
            model(x, g, mask, sids, tids)

        w = model.actor_attn._last_attn_weights  # [B, K, K]
        row_sums = w[:, :n_active, :].sum(dim=-1)   # [B, n_active]
        for b in range(B):
            for q_idx in range(n_active):
                s = row_sums[b, q_idx].item()
                assert abs(s - 1.0) < 1e-4, (
                    f"Row sum for (batch={b}, query={q_idx}) = {s:.6f}, expected 1.0"
                )


# ===========================================================================
# Gate 7.4  Noise / stochasticity test
# ===========================================================================

class TestNoise:
    """Actor must produce different w_pre for same input with different seeds."""

    def test_different_seeds_different_output(self):
        """
        Two forward passes with different manual seeds (training mode) must
        produce different w_pre tensors.
        """
        model = make_model()
        model.train()
        x, g, mask, sids, tids = make_batch()

        torch.manual_seed(42)
        with torch.no_grad():
            w1, _, _, _ = model(x, g, mask, sids, tids)

        torch.manual_seed(99)
        with torch.no_grad():
            w2, _, _, _ = model(x, g, mask, sids, tids)

        diff = (w1 - w2).abs().max().item()
        assert diff > 1e-6, (
            f"w_pre identical under different seeds (diff={diff:.2e}) — "
            "noise injection not working"
        )

    def test_same_seed_same_output(self):
        """
        Two forward passes with the SAME manual seed (training mode) must
        produce identical w_pre tensors (deterministic given seed).
        """
        model = make_model()
        model.train()
        x, g, mask, sids, tids = make_batch()

        torch.manual_seed(7)
        with torch.no_grad():
            w1, _, _, _ = model(x, g, mask, sids, tids)

        torch.manual_seed(7)
        with torch.no_grad():
            w2, _, _, _ = model(x, g, mask, sids, tids)

        assert torch.allclose(w1, w2, atol=1e-7), (
            "Same seed produced different w_pre — unexpected non-determinism"
        )

    def test_eval_mode_deterministic(self):
        """In eval mode (no noise), same input → same output regardless of seed."""
        model = make_model()
        model.eval()
        x, g, mask, sids, tids = make_batch()

        torch.manual_seed(42)
        with torch.no_grad():
            w1, _, _, _ = model(x, g, mask, sids, tids)

        torch.manual_seed(777)
        with torch.no_grad():
            w2, _, _, _ = model(x, g, mask, sids, tids)

        assert torch.allclose(w1, w2, atol=1e-7), (
            "Eval mode: different seeds produced different w_pre (should be deterministic)"
        )

    def test_noise_scale_with_log_sigma(self):
        """
        Larger log_sigma parameter → larger expected spread in w_pre across seeds.
        """
        x, g, mask, sids, tids = make_batch()

        def spread(log_sigma_init):
            m = make_model(log_sigma_init=log_sigma_init)
            m.train()
            results = []
            for seed in range(10):
                torch.manual_seed(seed)
                with torch.no_grad():
                    w, _, _, _ = m(x, g, mask, sids, tids)
                results.append(w)
            stack = torch.stack(results, dim=0)   # [10, B, K]
            return stack.std(dim=0).mean().item()

        sp_small = spread(-3.0)
        sp_large = spread( 0.0)
        assert sp_large > sp_small, (
            f"Larger log_sigma should produce larger spread: "
            f"small={sp_small:.6f}, large={sp_large:.6f}"
        )


# ===========================================================================
# Gate 7.5  Quantile test
# ===========================================================================

class TestQuantiles:
    """Critic outputs N_quantiles values in monotonically plausible range."""

    def test_output_shape_and_count(self):
        """q1 and q2 must have shape [B, N_quantiles]."""
        model = make_model()
        model.eval()
        x, g, mask, sids, tids = make_batch()
        w_in = torch.rand(B, K); w_in /= w_in.sum(-1, keepdim=True)

        with torch.no_grad():
            _, _, q1, q2 = model(x, g, mask, sids, tids, w_pre_in=w_in)

        assert q1.shape == (B, N_Q), f"q1 shape {q1.shape}"
        assert q2.shape == (B, N_Q), f"q2 shape {q2.shape}"

    def test_taus_registered_buffer(self):
        """Fixed quantile levels τ must be registered as a buffer."""
        model = make_model()
        assert hasattr(model, "taus"), "model.taus buffer missing"
        assert model.taus.shape == (N_Q,), f"taus shape {model.taus.shape}"

        expected_first = 1.0 / (2.0 * N_Q)
        expected_last  = (2.0 * N_Q - 1.0) / (2.0 * N_Q)
        assert abs(model.taus[0].item()  - expected_first) < 1e-6
        assert abs(model.taus[-1].item() - expected_last)  < 1e-6

    def test_quantiles_finite(self):
        """No NaN or Inf in quantile outputs."""
        model = make_model()
        model.eval()
        x, g, mask, sids, tids = make_batch()
        w_in = torch.rand(B, K); w_in /= w_in.sum(-1, keepdim=True)

        with torch.no_grad():
            _, _, q1, q2 = model(x, g, mask, sids, tids, w_pre_in=w_in)

        assert torch.isfinite(q1).all(), "q1 contains NaN/Inf"
        assert torch.isfinite(q2).all(), "q2 contains NaN/Inf"

    def test_twin_critics_independent(self):
        """Q1 and Q2 must produce different outputs (separate parameters)."""
        model = make_model()
        model.eval()
        x, g, mask, sids, tids = make_batch()
        w_in = torch.rand(B, K); w_in /= w_in.sum(-1, keepdim=True)

        with torch.no_grad():
            _, _, q1, q2 = model(x, g, mask, sids, tids, w_pre_in=w_in)

        diff = (q1 - q2).abs().max().item()
        assert diff > 1e-6, (
            f"Q1 and Q2 are identical (diff={diff:.2e}) — not independent"
        )


# ===========================================================================
# Gate 7.6  TCN receptive field test
# ===========================================================================

class TestTCNReceptiveField:
    """Receptive field formula matches or exceeds L."""

    def test_default_rf_exceeds_L(self):
        """
        With levels=5, kernel=3, base=2:
          RF = 1 + (3-1)×(2^5 - 1) = 1 + 2×31 = 63 ≥ 60 = L.
        """
        tcn = CausalTCN(
            in_channels=10, channels=32,
            levels=5, kernel_size=3, dilation_base=2,
        )
        L_default = 60
        rf = tcn.receptive_field
        assert rf >= L_default, (
            f"TCN receptive field {rf} < L={L_default}"
        )
        assert rf == 63, f"Expected RF=63, got {rf}"

    def test_rf_formula(self):
        """
        Verify RF matches analytic formula:
          RF = 1 + (kernel_size − 1) × sum(dilation_base^l  for l in range(levels))
        """
        for levels, k, base in [(3, 3, 2), (5, 3, 2), (4, 2, 3)]:
            tcn = CausalTCN(10, 32, levels=levels, kernel_size=k, dilation_base=base)
            expected = 1 + (k - 1) * sum(base ** l for l in range(levels))
            assert tcn.receptive_field == expected, (
                f"levels={levels},k={k},base={base}: "
                f"RF={tcn.receptive_field}, expected={expected}"
            )

    def test_causality_no_future_leakage(self):
        """
        Changing the LAST timestep of input should NOT affect the first
        (L/2) output timesteps.  (Strict causality: output[t] depends only on
        input[0..t].)
        """
        torch.manual_seed(0)
        L_test  = 12
        in_c, out_c = 8, 16
        tcn = CausalTCN(in_c, out_c, levels=2, kernel_size=3, dilation_base=2)
        tcn.eval()

        x     = torch.randn(1, in_c, L_test)
        x_mod = x.clone()
        x_mod[0, :, -1] = x_mod[0, :, -1] + 10.0   # perturb only last timestep

        with torch.no_grad():
            y     = tcn(x)
            y_mod = tcn(x_mod)

        # First L_test//2 outputs must be identical
        safe_t = L_test // 2
        diff   = (y[0, :, :safe_t] - y_mod[0, :, :safe_t]).abs().max().item()
        assert diff < 1e-6, (
            f"Perturbing last timestep changed early output by {diff:.2e} — "
            "future leakage detected!"
        )

    def test_model_tcn_rf_adequate(self):
        """The model's embedded TCN has receptive_field ≥ L (model L=60 default)."""
        model = make_model()
        rf = model.tcn.receptive_field
        assert rf >= 60, f"Model TCN RF={rf} < L=60"


# ===========================================================================
# Gate 7.7  No NaN sweep
# ===========================================================================

class TestNoNaN:
    """No NaN/Inf in any output tensor for 100 random inputs."""

    def _check_no_nan(self, tensors: dict, step: int):
        for name, t in tensors.items():
            if t is None:
                continue
            assert torch.isfinite(t).all(), (
                f"Step {step}: tensor '{name}' contains NaN or Inf"
            )

    def test_actor_no_nan_100_random(self):
        """Actor outputs (w_pre, log_prob) must be finite for 100 random inputs."""
        model = make_model()
        model.train()
        rng = torch.Generator(); rng.manual_seed(0)

        for i in range(100):
            x          = torch.randn(B, L, K, F,        generator=rng)
            g          = torch.randn(B, D_G,             generator=rng)
            mask       = (torch.rand(B, K, generator=rng) > 0.5).float()
            mask[:, 0] = 1.0                              # ensure ≥1 active slot
            sector_ids = torch.randint(0, NUM_SECTORS, (B, K), generator=rng)
            ticker_ids = torch.randint(0, NUM_TICKERS, (B, K), generator=rng)
            ticker_ids[mask < 0.5] = -1

            with torch.no_grad():
                w_pre, log_prob, _, _ = model(x, g, mask, sector_ids, ticker_ids)

            self._check_no_nan({"w_pre": w_pre, "log_prob": log_prob}, step=i)

    def test_critic_no_nan_100_random(self):
        """Critic outputs (q1, q2) must be finite for 100 random inputs."""
        model = make_model()
        model.eval()
        rng = torch.Generator(); rng.manual_seed(1)

        for i in range(100):
            x          = torch.randn(B, L, K, F,        generator=rng)
            g          = torch.randn(B, D_G,             generator=rng)
            mask       = (torch.rand(B, K, generator=rng) > 0.5).float()
            mask[:, 0] = 1.0
            sector_ids = torch.randint(0, NUM_SECTORS, (B, K), generator=rng)
            ticker_ids = torch.randint(0, NUM_TICKERS, (B, K), generator=rng)
            w_in       = torch.rand(B, K, generator=rng) * mask
            w_in_sum   = w_in.sum(-1, keepdim=True).clamp(min=1e-8)
            w_in       = w_in / w_in_sum

            with torch.no_grad():
                _, _, q1, q2 = model(x, g, mask, sector_ids, ticker_ids,
                                     w_pre_in=w_in)

            self._check_no_nan({"q1": q1, "q2": q2}, step=i)

    def test_no_nan_with_all_but_one_masked(self):
        """
        Edge case: only 1 active slot per sample.
        Pooling and attention should still produce finite outputs.
        """
        model = make_model()
        model.eval()
        x, g, _, sids, tids = make_batch(n_active=0)
        mask = torch.zeros(B, K)
        mask[:, 0] = 1.0    # only slot 0 active

        with torch.no_grad():
            w_pre, log_prob, _, _ = model(x, g, mask, sids, tids)

        assert torch.isfinite(w_pre).all(),    "w_pre not finite with 1 active slot"
        assert torch.isfinite(log_prob).all(), "log_prob not finite with 1 active slot"
        # Only slot 0 should have w_pre=1, rest=0
        assert (w_pre[:, 0] - 1.0).abs().max().item() < 1e-4

    def test_no_nan_with_many_inactive(self):
        """Majority inactive (only 2 out of K active): no NaN."""
        model = make_model()
        model.eval()
        x, g, _, sids, tids = make_batch(n_active=2)
        mask = torch.zeros(B, K)
        mask[:, :2] = 1.0

        with torch.no_grad():
            w_pre, log_prob, _, _ = model(x, g, mask, sids, tids)

        assert torch.isfinite(w_pre).all()
        assert torch.isfinite(log_prob).all()


# ===========================================================================
# Additional: embedding and parameter group tests
# ===========================================================================

class TestEmbeddingsAndParams:

    def test_embedding_parameters_identified(self):
        """embedding_parameters() returns the two embedding weight tensors."""
        model = make_model()
        emb_params = list(model.embedding_parameters())
        assert len(emb_params) == 2, (
            f"Expected 2 embedding param tensors, got {len(emb_params)}"
        )

    def test_non_embedding_parameters_excludes_embeddings(self):
        """non_embedding_parameters() must not contain embedding weights."""
        model = make_model()
        emb_ids  = {id(p) for p in model.embedding_parameters()}
        non_emb  = list(model.non_embedding_parameters())
        overlap  = [id(p) for p in non_emb if id(p) in emb_ids]
        assert len(overlap) == 0, (
            "non_embedding_parameters() contains embedding weights — "
            "weight-decay exclusion will be incorrect"
        )

    def test_three_independent_attention_stacks(self):
        """Actor, Q1, Q2 attention stacks must have separate parameters."""
        model = make_model()
        actor_ids = {id(p) for p in model.actor_attn.parameters()}
        q1_ids    = {id(p) for p in model.q1_attn.parameters()}
        q2_ids    = {id(p) for p in model.q2_attn.parameters()}

        assert len(actor_ids & q1_ids) == 0, "actor_attn shares params with q1_attn"
        assert len(actor_ids & q2_ids) == 0, "actor_attn shares params with q2_attn"
        assert len(q1_ids   & q2_ids) == 0, "q1_attn shares params with q2_attn"

    def test_tcn_shared_between_branches(self):
        """The single TCN is used by all branches (shared weights)."""
        model = make_model()
        # Only one TCN object should exist
        assert hasattr(model, "tcn"), "model.tcn not found"
        # No separate q1_tcn / q2_tcn
        assert not hasattr(model, "q1_tcn"),  "model has separate q1_tcn (should be shared)"
        assert not hasattr(model, "actor_tcn"), "model has separate actor_tcn (should be shared)"
