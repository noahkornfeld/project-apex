"""
Gate 4 Tests -- Constraint Projector (Bible §4.5)
==================================================
All 8 gate tests from the Phase 4 milestone specification.

RED FLAG (per spec): If gradcheck fails or simplex violation exceeds 1e-4 on
any test case, stop and redesign the projection algorithm before proceeding.
A broken projector will silently corrupt every downstream component.

Run with:
    pytest tests/test_phase4_gate4.py -v
"""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.constraint_projector import (
    ConstraintProjector,
    _project_simplex,
    _project_per_name,
    _project_sector,
)

# ---------------------------------------------------------------------------
# Constants matching master_config.yaml §4.5
# ---------------------------------------------------------------------------
PER_NAME_CAP = 0.20
SECTOR_CAP   = 0.50
K_MAX        = 110
N_SECTORS    = 11       # GICS has 11 sectors (coded 0..10)

torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_projector(max_iters: int = 200, tol: float = 1e-7) -> ConstraintProjector:
    return ConstraintProjector(
        per_name_cap=PER_NAME_CAP,
        sector_cap=SECTOR_CAP,
        max_iters=max_iters,
        tol=tol,
    )


def _make_random_input(
    B: int = 1,
    K: int = K_MAX,
    n_active: int = 90,
    seed: int = 0,
):
    """
    Build random (w_pre, mask, sector_ids) suitable for projection tests.

    Active slots: indices 0..n_active-1.
    w_pre: softmax of random logits over active slots (sums to 1 over active).
    sector_ids: cycle through 0..N_SECTORS-1 for active slots; -1 elsewhere.
    """
    torch.manual_seed(seed)

    mask = torch.zeros(B, K)
    mask[:, :n_active] = 1.0

    logits = torch.randn(B, K)
    logits_masked = logits * mask - (1.0 - mask) * 1e9
    w_pre = torch.softmax(logits_masked, dim=-1) * mask

    sector_ids = torch.full((B, K), -1, dtype=torch.long)
    for b in range(B):
        for i in range(n_active):
            sector_ids[b, i] = i % N_SECTORS

    return w_pre, mask, sector_ids


# ===========================================================================
# Gate 4.1  Simplex constraint: output sums to 1.0 +/- 1e-6 for 1000 inputs
# ===========================================================================

class TestSimplex:
    def test_sum_to_one_1000_random(self):
        proj = _make_projector()
        max_violation = 0.0

        for i in range(1000):
            n_active = torch.randint(7, K_MAX + 1, (1,)).item()
            w_pre, mask, sid = _make_random_input(n_active=n_active, seed=i)
            with torch.no_grad():
                w_exec = proj(w_pre, mask, sid)
            violation = abs(w_exec.sum().item() - 1.0)
            max_violation = max(max_violation, violation)

        assert max_violation <= 1e-6, (
            f"Simplex violation {max_violation:.3e} > 1e-6 over 1000 random inputs"
        )

    def test_sum_to_one_concentrated_input(self):
        """All weight on one asset must still give sum=1 after projection."""
        proj = _make_projector()
        w_pre = torch.zeros(K_MAX)
        w_pre[0] = 1.0
        mask = torch.zeros(K_MAX); mask[:30] = 1.0
        sid = torch.full((K_MAX,), -1, dtype=torch.long)
        for i in range(30): sid[i] = i % N_SECTORS
        with torch.no_grad():
            w_exec = proj(w_pre, mask, sid)
        assert abs(w_exec.sum().item() - 1.0) <= 1e-6


# ===========================================================================
# Gate 4.2  Long-only: all output elements >= 0
# ===========================================================================

class TestLongOnly:
    def test_non_negative_1000_random(self):
        proj = _make_projector()
        min_global = float("inf")

        for i in range(1000):
            n_active = torch.randint(7, K_MAX + 1, (1,)).item()
            w_pre, mask, sid = _make_random_input(n_active=n_active, seed=i + 10_000)
            with torch.no_grad():
                w_exec = proj(w_pre, mask, sid)
            min_global = min(min_global, w_exec.min().item())

        assert min_global >= -1e-7, (
            f"Long-only violation: min weight = {min_global:.4e} over 1000 inputs"
        )

    def test_negative_input_becomes_non_negative(self):
        """Negative w_pre inputs must never produce negative w_exec."""
        proj = _make_projector()
        w_pre = torch.full((K_MAX,), -5.0)
        mask = torch.zeros(K_MAX); mask[:20] = 1.0
        sid = torch.full((K_MAX,), -1, dtype=torch.long)
        for i in range(20): sid[i] = i % N_SECTORS
        with torch.no_grad():
            w_exec = proj(w_pre, mask, sid)
        assert w_exec.min().item() >= -1e-7


# ===========================================================================
# Gate 4.3  Per-name cap: no element exceeds per_name_cap + 1e-6
# ===========================================================================

class TestPerNameCap:
    def test_per_name_cap_1000_random(self):
        proj = _make_projector()
        max_excess = 0.0

        for i in range(1000):
            n_active = torch.randint(7, K_MAX + 1, (1,)).item()
            w_pre, mask, sid = _make_random_input(n_active=n_active, seed=i + 20_000)
            with torch.no_grad():
                w_exec = proj(w_pre, mask, sid)
            excess = (w_exec - PER_NAME_CAP).clamp(min=0).max().item()
            max_excess = max(max_excess, excess)

        assert max_excess <= 1e-6, (
            f"Per-name cap violation: max excess = {max_excess:.3e}"
        )

    def test_concentrated_input_is_capped(self):
        """All weight on one asset must not exceed per_name_cap after projection."""
        proj = _make_projector()
        w_pre = torch.zeros(K_MAX); w_pre[0] = 1.0
        mask = torch.zeros(K_MAX); mask[:20] = 1.0
        sid = torch.full((K_MAX,), -1, dtype=torch.long)
        for i in range(20): sid[i] = i % N_SECTORS
        with torch.no_grad():
            w_exec = proj(w_pre, mask, sid)
        assert w_exec.max().item() <= PER_NAME_CAP + 1e-6


# ===========================================================================
# Gate 4.4  Sector cap: sum of weights per GICS sector <= sector_cap + 1e-6
# ===========================================================================

class TestSectorCap:
    def test_sector_cap_1000_random(self):
        proj = _make_projector()
        max_excess = 0.0

        for i in range(1000):
            n_active = torch.randint(7, K_MAX + 1, (1,)).item()
            w_pre, mask, sid = _make_random_input(n_active=n_active, seed=i + 30_000)
            with torch.no_grad():
                w_exec = proj(w_pre.squeeze(0), mask.squeeze(0), sid.squeeze(0))
            for s in range(N_SECTORS):
                s_sum = w_exec[sid.squeeze(0) == s].sum().item()
                excess = max(0.0, s_sum - SECTOR_CAP)
                max_excess = max(max_excess, excess)

        assert max_excess <= 1e-6, (
            f"Sector cap violation: max excess = {max_excess:.3e}"
        )

    def test_single_sector_universe_is_capped(self):
        """All active assets in one sector must have sector sum <= sector_cap."""
        proj = _make_projector()
        n_active = 20
        w_pre = torch.full((K_MAX,), 1.0 / n_active)
        mask = torch.zeros(K_MAX); mask[:n_active] = 1.0
        sid = torch.full((K_MAX,), -1, dtype=torch.long)
        sid[:n_active] = 0                          # all in sector 0
        with torch.no_grad():
            w_exec = proj(w_pre, mask, sid)
        sector_sum = w_exec[:n_active].sum().item()
        assert sector_sum <= SECTOR_CAP + 1e-6, (
            f"Single-sector sum = {sector_sum:.4f} > sector_cap = {SECTOR_CAP}"
        )


# ===========================================================================
# Gate 4.5  Mask: w_exec[i] == 0 for all i where mask[i] == 0
# ===========================================================================

class TestMaskEnforcement:
    def test_inactive_slots_zero_1000_random(self):
        proj = _make_projector()

        for i in range(1000):
            n_active = torch.randint(1, K_MAX, (1,)).item()
            w_pre, mask, sid = _make_random_input(n_active=n_active, seed=i + 40_000)
            with torch.no_grad():
                w_exec = proj(w_pre, mask, sid)
            inactive_mass = w_exec[mask == 0].abs().max().item()
            assert inactive_mass < 1e-7, (
                f"Inactive slot mass {inactive_mass:.2e} != 0 at input {i}"
            )

    def test_random_logits_at_inactive_ignored(self):
        """Large random values at inactive slots must not leak into w_exec."""
        proj = _make_projector()
        n_active = 50
        w_pre = torch.randn(K_MAX) * 100.0        # extreme values everywhere
        mask = torch.zeros(K_MAX); mask[:n_active] = 1.0
        sid = torch.full((K_MAX,), -1, dtype=torch.long)
        for i in range(n_active): sid[i] = i % N_SECTORS
        with torch.no_grad():
            w_exec = proj(w_pre, mask, sid)
        assert w_exec[n_active:].abs().max().item() < 1e-7


# ===========================================================================
# Gate 4.6  Differentiability: gradcheck passes on 50 random inputs
# ===========================================================================

class TestDifferentiability:
    def test_gradcheck_50_random(self):
        """
        torch.autograd.gradcheck verifies d(w_exec)/d(w_pre) via finite
        differences. Uses K=15 and float64 for numerical accuracy.
        RED FLAG: if any input fails, redesign the projector.
        """
        proj = ConstraintProjector(
            per_name_cap=PER_NAME_CAP,
            sector_cap=SECTOR_CAP,
            max_iters=60,
            tol=1e-9,           # effectively runs all 60 iters for stable grad
        ).double()

        K = 15
        n_active = 10

        passed = 0
        for i in range(50):
            torch.manual_seed(i + 50_000)

            # Build double-precision inputs
            mask = torch.zeros(K, dtype=torch.float64)
            mask[:n_active] = 1.0
            sid = torch.full((K,), -1, dtype=torch.long)
            for j in range(n_active):
                sid[j] = j % 4             # 4 sectors for small K

            # Realistic w_pre: softmax over active
            logits = torch.randn(K, dtype=torch.float64)
            logits_m = logits * mask - (1.0 - mask) * 1e9
            w_base = torch.softmax(logits_m, dim=-1) * mask
            w_input = w_base.detach().requires_grad_(True)

            try:
                torch.autograd.gradcheck(
                    lambda w: proj(w, mask, sid),
                    (w_input,),
                    eps=1e-5,
                    atol=1e-4,
                    rtol=1e-3,
                )
                passed += 1
            except Exception:
                pass

        assert passed == 50, (
            f"RED FLAG: gradcheck passed {passed}/50 -- redesign projector"
        )


# ===========================================================================
# Gate 4.7  Idempotence: P(P(w)) ~= P(w)  (+/- 1e-5)
# ===========================================================================

class TestIdempotence:
    def test_idempotent_100_random(self):
        """Projecting an already-projected vector returns it unchanged."""
        proj = _make_projector()
        max_delta = 0.0

        for i in range(100):
            n_active = torch.randint(10, 80, (1,)).item()
            w_pre, mask, sid = _make_random_input(n_active=n_active, seed=i + 60_000)
            with torch.no_grad():
                w1 = proj(w_pre, mask, sid)
                w2 = proj(w1.clone(), mask, sid)
            delta = (w2 - w1).abs().max().item()
            max_delta = max(max_delta, delta)

        assert max_delta <= 1e-5, (
            f"Idempotence violation: max second-projection delta = {max_delta:.3e}"
        )

    def test_equal_weight_feasible_is_idempotent(self):
        """
        Equal-weight over 30 assets (30 * 0.15 = 4.5 > 1, sector sum = 3/11 ~ 0.27
        < 0.35) is feasible and must be returned unchanged.
        """
        proj = _make_projector()
        n_active = 30
        w = torch.zeros(K_MAX)
        w[:n_active] = 1.0 / n_active
        mask = torch.zeros(K_MAX); mask[:n_active] = 1.0
        sid = torch.full((K_MAX,), -1, dtype=torch.long)
        for i in range(n_active): sid[i] = i % N_SECTORS

        with torch.no_grad():
            w_exec = proj(w, mask, sid)

        delta = (w_exec - w).abs().max().item()
        assert delta <= 1e-5, (
            f"Equal-weight idempotence failed: delta = {delta:.3e}"
        )

    def test_known_feasible_vector_unchanged(self):
        """
        Manually constructed feasible point must be returned essentially
        unchanged (up to convergence tolerance).
        """
        proj = _make_projector()
        n_active = 44           # 44 * 0.15 = 6.6 > 1
        # Spread equally across 11 sectors: 4 per sector
        # Per-sector weight = 4/44 * 0.15... let's set weight carefully
        w = torch.zeros(K_MAX)
        # Give each asset per_name_cap/2 = 0.075, only the first n_active
        # Total weight = 44 * 0.075 = 3.3 > 1, so we normalise
        raw = torch.zeros(K_MAX)
        raw[:n_active] = 0.075
        raw[:n_active] /= raw.sum()     # now sums to 1; each = 1/44 ~ 0.023

        mask = torch.zeros(K_MAX); mask[:n_active] = 1.0
        sid = torch.full((K_MAX,), -1, dtype=torch.long)
        for i in range(n_active): sid[i] = i % N_SECTORS

        with torch.no_grad():
            w_exec = proj(raw, mask, sid)

        delta = (w_exec - raw).abs().max().item()
        assert delta <= 1e-5, (
            f"Known feasible vector changed by {delta:.3e} after projection"
        )


# ===========================================================================
# Gate 4.8  Fallback: empty feasibility set -> equal-weight, no crash
# ===========================================================================

class TestFallback:
    def test_zero_active_returns_all_zeros(self):
        """No active assets -> all-zero output without crash."""
        proj = _make_projector()
        w_pre = torch.randn(K_MAX)
        mask = torch.zeros(K_MAX)
        sid = torch.full((K_MAX,), -1, dtype=torch.long)
        with torch.no_grad():
            w_exec = proj(w_pre, mask, sid)
        assert w_exec.abs().sum().item() < 1e-7

    def test_infeasible_cap_gives_equal_weight(self):
        """
        n_active=4, per_name_cap=0.20 -> max feasible sum = 0.80 < 1.
        Projector must fall back to equal-weight (0.25 each) without crash.
        """
        proj = _make_projector()
        n_active = 4
        w_pre = torch.randn(K_MAX)
        mask = torch.zeros(K_MAX); mask[:n_active] = 1.0
        sid = torch.full((K_MAX,), -1, dtype=torch.long)
        for i in range(n_active): sid[i] = i % N_SECTORS

        with torch.no_grad():
            w_exec = proj(w_pre, mask, sid)

        expected_w = 1.0 / n_active
        active_w = w_exec[:n_active]
        assert (active_w - expected_w).abs().max().item() < 1e-6, (
            f"Fallback not equal-weight: got {active_w.tolist()}"
        )
        assert w_exec[n_active:].abs().max().item() < 1e-7

    def test_single_active_asset_weight_is_one(self):
        """
        1 active asset: n_active * per_name_cap = 0.20 < 1 -> fallback.
        That single asset must receive weight 1.0.
        """
        proj = _make_projector()
        w_pre = torch.randn(K_MAX)
        mask = torch.zeros(K_MAX); mask[42] = 1.0
        sid = torch.full((K_MAX,), -1, dtype=torch.long); sid[42] = 0
        with torch.no_grad():
            w_exec = proj(w_pre, mask, sid)
        assert abs(w_exec[42].item() - 1.0) < 1e-6

    def test_edge_cases_do_not_crash(self):
        """Various edge cases must not raise exceptions and produce finite output."""
        proj = _make_projector()
        configs = [
            (1, False),
            (2, True),
            (5, False),     # infeasible
            (6, False),     # 6*0.15=0.90 < 1 -> infeasible
            (7, True),      # 7*0.15=1.05 >= 1 -> feasible
            (K_MAX, False),
        ]
        for n_active, same_sector in configs:
            w_pre = torch.randn(K_MAX)
            mask = torch.zeros(K_MAX); mask[:n_active] = 1.0
            sid = torch.full((K_MAX,), -1, dtype=torch.long)
            for i in range(n_active):
                sid[i] = 0 if same_sector else (i % N_SECTORS)
            try:
                with torch.no_grad():
                    w_exec = proj(w_pre, mask, sid)
                assert torch.isfinite(w_exec).all(), (
                    f"Non-finite output for n_active={n_active}"
                )
            except Exception as e:
                pytest.fail(f"Edge case n_active={n_active} raised: {e}")


# ===========================================================================
# Bonus: Batch consistency
# ===========================================================================

class TestBatchConsistency:
    def test_batch_equals_individual(self):
        """Batched projection must equal stacked individual projections."""
        proj = _make_projector()
        B = 8
        n_active = 85

        w_list, m_list, s_list = [], [], []
        for i in range(B):
            wp, m, s = _make_random_input(n_active=n_active, seed=i + 70_000)
            w_list.append(wp.squeeze(0))
            m_list.append(m.squeeze(0))
            s_list.append(s.squeeze(0))

        w_batch = torch.stack(w_list)
        m_batch = torch.stack(m_list)
        s_batch = torch.stack(s_list)

        with torch.no_grad():
            w_exec_batch = proj(w_batch, m_batch, s_batch)

        for b in range(B):
            with torch.no_grad():
                w_single = proj(w_list[b], m_list[b], s_list[b])
            delta = (w_exec_batch[b] - w_single).abs().max().item()
            assert delta < 1e-5, (
                f"Batch[{b}] differs from individual by {delta:.2e}"
            )


# ===========================================================================
# Integration: all constraints satisfied simultaneously
# ===========================================================================

class TestAllConstraintsSimultaneous:
    def test_all_constraints_1000_random(self):
        """
        Single unified test: for 1000 random inputs, ALL of the following
        must hold simultaneously:
            - sum(w_exec) == 1.0 +/- 1e-6
            - w_exec[i] >= 0  for all i
            - w_exec[i] <= per_name_cap + 1e-6  for all i
            - sum(w_exec[sector==s]) <= sector_cap + 1e-6  for all s
            - w_exec[i] == 0  for all inactive i
        """
        proj = _make_projector()

        for i in range(1000):
            n_active = torch.randint(7, K_MAX + 1, (1,)).item()
            w_pre, mask, sid = _make_random_input(n_active=n_active, seed=i + 80_000)
            sid1 = sid.squeeze(0)
            m1 = mask.squeeze(0)
            wp1 = w_pre.squeeze(0)

            with torch.no_grad():
                w_exec = proj(wp1, m1, sid1)

            # simplex
            assert abs(w_exec.sum().item() - 1.0) <= 1e-6, (
                f"[{i}] simplex violation: sum={w_exec.sum().item():.8f}"
            )
            # long-only
            assert w_exec.min().item() >= -1e-7, (
                f"[{i}] long-only violation: min={w_exec.min().item():.4e}"
            )
            # per-name cap
            assert (w_exec - PER_NAME_CAP).clamp(min=0).max().item() <= 1e-6, (
                f"[{i}] per-name cap violation: max={w_exec.max().item():.4f}"
            )
            # sector cap
            for s in range(N_SECTORS):
                s_sum = w_exec[sid1 == s].sum().item()
                assert s_sum <= SECTOR_CAP + 1e-6, (
                    f"[{i}] sector {s} cap violation: sum={s_sum:.4f}"
                )
            # mask
            inactive = w_exec[m1 == 0]
            if inactive.numel() > 0:
                assert inactive.abs().max().item() < 1e-7, (
                    f"[{i}] mask violation"
                )
