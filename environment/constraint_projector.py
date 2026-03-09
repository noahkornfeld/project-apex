"""
Constraint Projector -- Bible §4.5
====================================
Deterministic, differentiable Euclidean projection of w_pre onto the
portfolio feasibility set C = C_simplex ∩ C_per_name ∩ C_sector:

    C_simplex  = {w : sum_i w_i = 1,  w_i >= 0}   (over active slots only)
    C_per_name = {w : w_i <= per_name_cap  for all i}
    C_sector   = {w : sum_{i in s} w_i <= sector_cap  for all GICS sectors s}

Algorithm: Dykstra's alternating projections (Boyle & Dykstra, 1986).
Converges to the minimum-Euclidean-distance point in C from w_pre.
Fully implemented in PyTorch; gradients flow through autograd automatically.

Mask enforcement: inactive slots (mask[i] = 0) are zeroed before projection
and cannot receive any weight.

Fallback: if n_active * per_name_cap < 1.0 (feasibility set provably empty),
the projector returns equal-weight among active assets without crashing.

Config values (§4.5, master_config.yaml):
    per_name_cap = 0.15
    sector_cap   = 0.35
"""

import torch
import torch.nn as nn

EPS = 1e-8


# ---------------------------------------------------------------------------
# Projection sub-routines (each differentiable via autograd)
# ---------------------------------------------------------------------------

def _project_simplex(v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Euclidean projection onto the masked probability simplex.

    Solves: argmin ||w - v||^2  s.t.  sum_{active} w_i = 1, w_i >= 0,
                                       w_i = 0 for inactive slots.

    Uses the sort-based O(K log K) algorithm (Duchi et al., 2008).
    Differentiable at non-degenerate inputs via torch.sort.

    Args:
        v    : [B, K] float tensor -- input weights.
        mask : [B, K] float tensor -- 1=active, 0=inactive.

    Returns:
        [B, K] projected tensor.
    """
    B, K = v.shape
    LARGE = 1e9

    # Inactive slots pushed to -inf so they sort to the bottom
    v_adj = v * mask + (-LARGE) * (1.0 - mask)

    # Sort descending
    u = torch.sort(v_adj, dim=-1, descending=True).values   # [B, K]

    # Cumulative sum
    cssv = torch.cumsum(u, dim=-1)                          # [B, K]

    # 1-indexed position vector
    j = torch.arange(1, K + 1, device=v.device, dtype=v.dtype).unsqueeze(0)  # [1, K]

    # Condition: u_j * j > cssv_j - 1  (equivalent to u_j > (cssv_j - 1) / j)
    # Only count positions that are genuine (not the -LARGE padding)
    active_pos = u > (-LARGE / 2)                           # [B, K]
    rho_cond = (u * j > cssv - 1.0) & active_pos           # [B, K]

    # rho = last j satisfying the condition (clamped >= 1 for safety)
    rho = (rho_cond * j).amax(dim=-1, keepdim=True).clamp(min=1.0)  # [B, 1]

    # theta = (cssv[rho] - 1) / rho  -- Lagrange multiplier
    rho_idx = (rho - 1).long().clamp(min=0, max=K - 1)     # [B, 1]
    cssv_at_rho = torch.gather(cssv, -1, rho_idx)           # [B, 1]
    theta = (cssv_at_rho - 1.0) / rho                       # [B, 1]

    # Project and zero inactive slots
    return torch.clamp(v - theta, min=0.0) * mask


def _project_per_name(w: torch.Tensor, per_name_cap: float) -> torch.Tensor:
    """
    Project w onto {w : 0 <= w_i <= per_name_cap}.
    Differentiable (clamp is piecewise linear).
    """
    return torch.clamp(w, min=0.0, max=per_name_cap)


def _project_sector(
    w: torch.Tensor,
    mask: torch.Tensor,
    sector_ids: torch.Tensor,
    sector_cap: float,
) -> torch.Tensor:
    """
    For each GICS sector s: if sum_{i in s, active} w_i > sector_cap,
    scale those weights down proportionally to sector_cap.

    Differentiable: uses only multiply / divide / clamp (no in-place ops).

    Args:
        w          : [B, K] float.
        mask       : [B, K] float (1=active, 0=inactive).
        sector_ids : [B, K] int64; GICS sector code per slot, -1=inactive.
        sector_cap : float.

    Returns:
        [B, K] with each sector sum <= sector_cap.
    """
    unique_sectors = sector_ids.unique()
    adjustment = torch.zeros_like(w)

    for s in unique_sectors:
        s_val = s.item()
        if s_val < 0:
            continue                                        # skip inactive sentinel

        s_mask = (sector_ids == s).float() * mask          # [B, K]
        sector_sum = (w * s_mask).sum(dim=-1, keepdim=True)  # [B, 1]

        # scale = min(1, sector_cap / sector_sum) applied only where over-cap
        scale = torch.clamp(sector_cap / (sector_sum + EPS), max=1.0)  # [B, 1]
        needs_scale = (sector_sum > sector_cap).float()                 # [B, 1]
        effective_scale = 1.0 - needs_scale + needs_scale * scale       # [B, 1]

        # Accumulate the signed delta for this sector
        adjustment = adjustment + s_mask * w * (effective_scale - 1.0)

    return w + adjustment


# ---------------------------------------------------------------------------
# ConstraintProjector module
# ---------------------------------------------------------------------------

class ConstraintProjector(nn.Module):
    """
    Deterministic, differentiable Euclidean projection onto the portfolio
    feasibility set (Bible §4.5).

    Runs Dykstra's alternating projections over three constraint sets:
        C1 -- Probability simplex (active slots only)
        C2 -- Per-name cap box:  w_i <= per_name_cap
        C3 -- Sector cap:        sum_{i in s} w_i <= sector_cap  for all s

    Mask enforcement: inactive slots (mask[i] = 0) are zeroed before
    projection and cannot receive weight after projection.

    Fallback: if n_active * per_name_cap < 1.0 (feasibility provably empty),
    returns equal-weight among active assets.

    Args:
        per_name_cap : Max weight per asset (§4.5 default 0.15).
        sector_cap   : Max total weight per GICS sector (§4.5 default 0.35).
        max_iters    : Dykstra iterations (default 200; use 50 for gradcheck).
        tol          : Convergence tolerance on max weight change (default 1e-7).
    """

    def __init__(
        self,
        per_name_cap: float = 0.15,
        sector_cap: float = 0.35,
        max_iters: int = 200,
        tol: float = 1e-7,
    ):
        super().__init__()
        self.per_name_cap = per_name_cap
        self.sector_cap = sector_cap
        self.max_iters = max_iters
        self.tol = tol

    def forward(
        self,
        w_pre: torch.Tensor,
        mask: torch.Tensor,
        sector_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project w_pre to a feasible w_exec.

        Args:
            w_pre      : [..., K_max] pre-projection weights (any scale/sign).
            mask       : [..., K_max] float; 1=active, 0=inactive.
            sector_ids : [..., K_max] int64; GICS sector code per slot, -1=inactive.

        Returns:
            w_exec : [..., K_max] feasible weights satisfying all constraints.
        """
        squeeze = w_pre.dim() == 1
        if squeeze:
            w_pre = w_pre.unsqueeze(0)
            mask = mask.unsqueeze(0)
            sector_ids = sector_ids.unsqueeze(0)

        mask = mask.float()
        B, K = w_pre.shape

        # ── Step 1: enforce mask ──────────────────────────────────────────
        w = w_pre * mask

        # ── Step 2: detect infeasibility, prepare fallback ────────────────
        n_active = mask.sum(dim=-1, keepdim=True)               # [B, 1]
        max_feasible = n_active * self.per_name_cap             # [B, 1]
        # Infeasible when no active slots, or per-name cap too tight
        infeasible = (n_active < 1) | (max_feasible < 1.0 - 1e-6)   # [B, 1]
        w_fallback = mask / (n_active + EPS)                    # equal-weight [B, K]

        # ── Step 3: Dykstra's alternating projections ─────────────────────
        # Increment (correction) vectors -- initialised to zero
        p1 = torch.zeros_like(w)    # for C_simplex
        p2 = torch.zeros_like(w)    # for C_per_name
        p3 = torch.zeros_like(w)    # for C_sector

        for _ in range(self.max_iters):
            w_prev = w.detach()     # reference for convergence check only

            # C1: simplex projection
            y1 = _project_simplex(w + p1, mask)
            p1 = p1 + w - y1
            w = y1

            # C2: per-name cap projection
            y2 = _project_per_name(w + p2, self.per_name_cap)
            p2 = p2 + w - y2
            w = y2

            # C3: sector cap projection
            y3 = _project_sector(w + p3, mask, sector_ids, self.sector_cap)
            p3 = p3 + w - y3
            w = y3

            # Safety: re-enforce mask (numerical guard)
            w = w * mask

            # Convergence check (outside computation graph)
            with torch.no_grad():
                if (w.detach() - w_prev).abs().max().item() < self.tol:
                    break

        # ── Step 4: apply fallback where infeasible ───────────────────────
        w_exec = torch.where(infeasible, w_fallback, w)

        if squeeze:
            w_exec = w_exec.squeeze(0)

        return w_exec
