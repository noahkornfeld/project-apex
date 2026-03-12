"""
model/apex_actor_critic.py
Full Actor-Critic model — Project Apex Phase 7 (Bible §7).

Component pipeline (§7.1):
  1. Ticker + Sector Embeddings (§7.3)  → linear projection
  2. Shared CausalTCN encoder (§7.4)    → per-asset temporal representation
  3. Global context injection (§7.1)    → additive broadcast into TCN output
  4. Three independent CrossAssetAttention stacks (§7.5)
     actor_attn,  q1_attn,  q2_attn   (same arch, separate params)
  5. Branch-specific pooling (§7.6)
     mean-pool + learned query-attention pool  →  state_summary [D_pool]
  6. asset_repr = concat(per_asset_attn_out, g_broadcast, state_summary_broadcast)
  7. Actor head (§7.7)
     per-asset MLP → logits → mask → logistic-normal noise → re-mask
     → softmax → w_pre;  log_prob from pre-projection normal
  8. Twin distributional critic heads (§7.8)
     input = concat(state_summary, w_pre)  →  N_quantiles quantile values

Shapes (default config):
  x:          [B, L, K, F]     (L=60, K=K_max, F=25)
  g:          [B, D_g]
  mask:       [B, K]           (float32  1=active 0=inactive)
  sector_ids: [B, K]           (int64)
  ticker_ids: [B, K]           (int64; −1 for inactive slots)
  w_pre out:  [B, K]
  log_prob:   [B]
  q{1,2}:     [B, N_quantiles]
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.tcn import CausalTCN
from model.attention import CrossAssetAttention

EPS = 1e-8


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _mlp(dims: List[int], activation: str = "silu") -> nn.Sequential:
    """Build a dense MLP: dims[0] → dims[1] → … → dims[-1], SiLU between layers."""
    act = nn.SiLU if activation == "silu" else nn.ReLU
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class _QueryPool(nn.Module):
    """
    Learned single-query attention pooling (§7.6).
    Aggregates [B, K, D] → [B, D] by computing attention weights over K
    using a learned query vector, then taking the weighted sum.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.scale = d_model ** -0.5

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x    : [B, K, D]
        mask : [B, K] float  1=active
        returns [B, D]
        """
        scores = (x @ self.query) * self.scale          # [B, K]
        scores = scores - (1.0 - mask) * 1e9            # mask inactive → −∞
        weights = F.softmax(scores, dim=-1)              # [B, K]
        return (weights.unsqueeze(-1) * x).sum(dim=1)   # [B, D]


class _BranchPool(nn.Module):
    """
    Dual pooling (§7.6): masked mean-pool + learned query-attention pool.
    Concatenates both → state_summary ∈ R^(2*D).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.q_pool = _QueryPool(d_model)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        x    : [B, K, D]
        mask : [B, K] float
        returns state_summary [B, 2*D]
        """
        n_active = mask.sum(dim=1, keepdim=True).clamp(min=1.0)     # [B, 1]
        mean_pool = (x * mask.unsqueeze(-1)).sum(dim=1) / n_active   # [B, D]
        q_pool    = self.q_pool(x, mask)                              # [B, D]
        return torch.cat([mean_pool, q_pool], dim=-1)                 # [B, 2*D]


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ApexActorCritic(nn.Module):
    """
    Full Actor-Critic for Project Apex (Bible §7).

    Parameters
    ----------
    K_max            : max padded asset slots                    (default 110)
    F                : per-asset feature dimension               (default 25)
    D_g              : global context vector dimension           (default 20)
    num_tickers      : vocabulary size for ticker embedding
    num_sectors      : vocabulary size for sector embedding
    ticker_emb_dim   : D_ticker                                  (default 32)
    sector_emb_dim   : D_sector                                  (default 8)
    D_emb_proj       : linear projection of concatenated embs    (default 32)
    tcn_channels     : D_tcn                                     (default 128)
    tcn_levels       : dilated-conv levels                       (default 5)
    tcn_kernel_size  : conv kernel                               (default 3)
    tcn_dilation_base                                            (default 2)
    attn_d_model     : D_attn = D_tcn (must match)              (default 128)
    attn_n_heads                                                 (default 4)
    attn_d_ff                                                    (default 256)
    attn_n_layers                                                (default 2)
    attn_dropout                                                 (default 0.05)
    actor_hidden_dims: hidden dims for per-asset actor MLP       (default [128,128])
    critic_hidden_dims: hidden dims for critic MLP               (default [256,256])
    n_quantiles      : N for distributional critic               (default 32)
    log_sigma_init   : initial log of action noise std           (default -1.5)
    log_sigma_min    : clamp floor on log_sigma                  (default -5.0)
    log_sigma_max    : clamp ceiling on log_sigma                (default  1.0)
    """

    def __init__(
        self,
        K_max:             int          = 110,
        F:                 int          = 25,
        D_g:               int          = 20,
        num_tickers:       int          = 512,
        num_sectors:       int          = 12,
        ticker_emb_dim:    int          = 32,
        sector_emb_dim:    int          = 8,
        D_emb_proj:        int          = 32,
        tcn_channels:      int          = 128,
        tcn_levels:        int          = 5,
        tcn_kernel_size:   int          = 3,
        tcn_dilation_base: int          = 2,
        attn_d_model:      int          = 128,
        attn_n_heads:      int          = 4,
        attn_d_ff:         int          = 256,
        attn_n_layers:     int          = 2,
        attn_dropout:      float        = 0.05,
        actor_hidden_dims: List[int]    = None,
        critic_hidden_dims: List[int]   = None,
        n_quantiles:       int          = 32,
        log_sigma_init:    float        = -1.5,
        log_sigma_min:     float        = -5.0,
        log_sigma_max:     float        =  1.0,
    ):
        super().__init__()

        if actor_hidden_dims  is None: actor_hidden_dims  = [128, 128]
        if critic_hidden_dims is None: critic_hidden_dims = [256, 256]

        self.K_max      = K_max
        self.n_quantiles = n_quantiles
        self._log_sigma_min = log_sigma_min
        self._log_sigma_max = log_sigma_max

        # ------------------------------------------------------------------
        # §7.3  Embeddings (weight decay excluded — managed by optimizer)
        # ------------------------------------------------------------------
        self.ticker_emb = nn.Embedding(num_tickers, ticker_emb_dim, padding_idx=None)
        self.sector_emb = nn.Embedding(num_sectors, sector_emb_dim, padding_idx=None)
        self.emb_proj   = nn.Linear(ticker_emb_dim + sector_emb_dim, D_emb_proj)

        nn.init.normal_(self.ticker_emb.weight, std=0.02)
        nn.init.normal_(self.sector_emb.weight, std=0.02)

        # ------------------------------------------------------------------
        # §7.4  Shared CausalTCN (shared across actor + critic branches)
        # ------------------------------------------------------------------
        self.tcn = CausalTCN(
            in_channels  = F + D_emb_proj,
            channels     = tcn_channels,
            levels       = tcn_levels,
            kernel_size  = tcn_kernel_size,
            dilation_base = tcn_dilation_base,
        )
        assert attn_d_model == tcn_channels, (
            f"attn_d_model ({attn_d_model}) must equal tcn_channels ({tcn_channels})"
        )
        self._D = attn_d_model   # working dimension throughout

        # ------------------------------------------------------------------
        # §7.1  Global context injection  W_g : R^D_g → R^D_tcn
        # ------------------------------------------------------------------
        self.global_proj = nn.Linear(D_g, tcn_channels)

        # ------------------------------------------------------------------
        # §7.5  Three independent CrossAssetAttention stacks
        # ------------------------------------------------------------------
        attn_kw = dict(
            d_model=attn_d_model, n_heads=attn_n_heads,
            d_ff=attn_d_ff, n_layers=attn_n_layers,
            dropout=attn_dropout, use_sector_bias=True,
        )
        self.actor_attn = CrossAssetAttention(**attn_kw)
        self.q1_attn    = CrossAssetAttention(**attn_kw)
        self.q2_attn    = CrossAssetAttention(**attn_kw)

        # ------------------------------------------------------------------
        # §7.6  Branch-specific pooling modules (mean + query-attn)
        # ------------------------------------------------------------------
        self.actor_pool = _BranchPool(attn_d_model)
        self.q1_pool    = _BranchPool(attn_d_model)
        self.q2_pool    = _BranchPool(attn_d_model)

        # ------------------------------------------------------------------
        # Derived dimensions
        # D_pool = 2 * D (mean_pool + query_pool)
        # D_repr = D + D_g + D_pool  (per-asset attn + g_t broadcast + summary)
        # ------------------------------------------------------------------
        D_pool = 2 * attn_d_model                      # 256
        D_repr = attn_d_model + D_g + D_pool            # 128 + D_g + 256

        # ------------------------------------------------------------------
        # §7.7  Actor head  — per-asset MLP (shared weights across K axis)
        # ------------------------------------------------------------------
        actor_dims  = [D_repr] + actor_hidden_dims + [1]
        self.actor_mu_head = _mlp(actor_dims)                  # → scalar logit per asset
        self.log_sigma     = nn.Parameter(
            torch.full((), log_sigma_init)
        )                                                       # shared scalar noise std

        # Fixed quantile levels τ ∈ {1/(2N), 3/(2N), …, (2N-1)/(2N)}  (§7.8)
        taus = (2 * torch.arange(1, n_quantiles + 1) - 1) / (2.0 * n_quantiles)
        self.register_buffer("taus", taus)                     # [N_quantiles]

        # ------------------------------------------------------------------
        # §7.8  Twin distributional critic heads
        # critic input = concat(state_summary [D_pool], w_pre [K_max])
        # ------------------------------------------------------------------
        critic_in   = D_pool + K_max
        q1_dims     = [critic_in] + critic_hidden_dims + [n_quantiles]
        q2_dims     = [critic_in] + critic_hidden_dims + [n_quantiles]
        self.q1_head = _mlp(q1_dims)
        self.q2_head = _mlp(q2_dims)

    # ======================================================================
    # Public forward methods
    # ======================================================================

    def forward(
        self,
        x:          torch.Tensor,                  # [B, L, K, F]
        g:          torch.Tensor,                  # [B, D_g]
        mask:       torch.Tensor,                  # [B, K] float32
        sector_ids: torch.Tensor,                  # [B, K] int64
        ticker_ids: torch.Tensor,                  # [B, K] int64  (−1 inactive)
        w_pre_in:   Optional[torch.Tensor] = None, # [B, K]  for critic path
    ) -> Tuple[torch.Tensor, torch.Tensor,
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Full forward pass.

        Returns
        -------
        w_pre        : [B, K]           pre-projection portfolio weights
        log_prob     : [B]              log π(w_pre | s)
        q1_quantiles : [B, N_quantiles] or None  (only if w_pre_in provided)
        q2_quantiles : [B, N_quantiles] or None
        """
        # Shared encoding
        tcn_out, g_emb = self._encode(x, g, mask, sector_ids, ticker_ids)
        # tcn_out: [B, K, D_tcn]  g_emb: [B, D_tcn]

        # Actor branch
        w_pre, log_prob, actor_summary = self._actor_branch(
            tcn_out, mask, sector_ids, g_emb, x, g
        )

        # Critic branches (only when w_pre_in is supplied)
        q1 = q2 = None
        if w_pre_in is not None:
            q1, q2 = self._critic_branches(
                tcn_out, mask, sector_ids, w_pre_in
            )

        return w_pre, log_prob, q1, q2

    def actor_forward(
        self,
        x:          torch.Tensor,
        g:          torch.Tensor,
        mask:       torch.Tensor,
        sector_ids: torch.Tensor,
        ticker_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Actor-only forward.  Returns (w_pre [B,K], log_prob [B])."""
        tcn_out, g_emb = self._encode(x, g, mask, sector_ids, ticker_ids)
        w_pre, log_prob, _ = self._actor_branch(
            tcn_out, mask, sector_ids, g_emb, x, g
        )
        return w_pre, log_prob

    def critic_forward(
        self,
        x:          torch.Tensor,
        g:          torch.Tensor,
        mask:       torch.Tensor,
        sector_ids: torch.Tensor,
        ticker_ids: torch.Tensor,
        w_pre_in:   torch.Tensor,          # [B, K]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Critic-only forward.  Returns (q1 [B,N_q], q2 [B,N_q])."""
        tcn_out, _ = self._encode(x, g, mask, sector_ids, ticker_ids)
        return self._critic_branches(tcn_out, mask, sector_ids, w_pre_in)

    # ======================================================================
    # Parameter groups for optimizer (embedding params → no weight decay)
    # ======================================================================

    def embedding_parameters(self):
        """Parameters that must NOT receive weight decay."""
        return list(self.ticker_emb.parameters()) + list(self.sector_emb.parameters())

    def non_embedding_parameters(self):
        """All parameters except embedding weights (can receive weight decay)."""
        emb_ids = {id(p) for p in self.embedding_parameters()}
        return [p for p in self.parameters() if id(p) not in emb_ids]

    # ======================================================================
    # Internal helpers
    # ======================================================================

    def _encode(
        self,
        x:          torch.Tensor,   # [B, L, K, F]
        g:          torch.Tensor,   # [B, D_g]
        mask:       torch.Tensor,   # [B, K]
        sector_ids: torch.Tensor,   # [B, K]
        ticker_ids: torch.Tensor,   # [B, K]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Shared feature extraction:
          embeddings → TCN → global context injection
        Returns:
          tcn_out [B, K, D_tcn]
          g_emb   [B, D_tcn]
        """
        B, L, K, F_dim = x.shape

        # §7.3 Embedding lookup (clamp −1 inactive ids to 0, then zero by mask)
        safe_ticker  = ticker_ids.clamp(min=0)                     # [B, K]
        safe_sector  = sector_ids.clamp(min=0)                     # [B, K]
        t_emb = self.ticker_emb(safe_ticker)                       # [B, K, D_ticker]
        s_emb = self.sector_emb(safe_sector)                       # [B, K, D_sector]
        emb   = self.emb_proj(torch.cat([t_emb, s_emb], dim=-1))  # [B, K, D_emb]
        emb   = emb * mask.unsqueeze(-1)                           # zero inactive slots

        # Broadcast embeddings across time: [B, K, D_emb] → [B, L, K, D_emb]
        emb_t = emb.unsqueeze(1).expand(-1, L, -1, -1)            # [B, L, K, D_emb]

        # Concatenate features + embeddings: [B, L, K, F+D_emb]
        inp = torch.cat([x, emb_t], dim=-1)                        # [B, L, K, F+D_emb]

        # Reshape for TCN: [B*K, F+D_emb, L]  (per-asset independent processing)
        inp = inp.permute(0, 2, 3, 1).contiguous()                 # [B, K, F+D_emb, L]
        inp = inp.view(B * K, inp.shape[2], L)                     # [B*K, C_in, L]

        # §7.4 Shared TCN → take last timestep
        tcn_all = self.tcn(inp)                                     # [B*K, D_tcn, L]
        tcn_last = tcn_all[..., -1]                                 # [B*K, D_tcn]
        tcn_out  = tcn_last.view(B, K, self._D)                    # [B, K, D_tcn]

        # §7.1 Global context injection: cond = W_g · g_t
        g_emb   = self.global_proj(g)                              # [B, D_tcn]
        tcn_out = tcn_out + g_emb.unsqueeze(1)                     # [B, K, D_tcn]
        tcn_out = tcn_out * mask.unsqueeze(-1)                     # zero inactive

        return tcn_out, g_emb

    def _build_asset_repr(
        self,
        attn_out:     torch.Tensor,   # [B, K, D]
        g:            torch.Tensor,   # [B, D_g]
        state_summary: torch.Tensor,  # [B, D_pool]
        mask:         torch.Tensor,   # [B, K]
    ) -> torch.Tensor:
        """
        Construct per-asset representation (§7.6):
          concat(attn_out, g_broadcast, state_summary_broadcast)  → [B, K, D_repr]
        """
        B, K, _ = attn_out.shape
        g_bc   = g.unsqueeze(1).expand(-1, K, -1)             # [B, K, D_g]
        sum_bc = state_summary.unsqueeze(1).expand(-1, K, -1) # [B, K, D_pool]
        return torch.cat([attn_out, g_bc, sum_bc], dim=-1)     # [B, K, D_repr]

    def _actor_branch(
        self,
        tcn_out:    torch.Tensor,   # [B, K, D_tcn]
        mask:       torch.Tensor,   # [B, K]
        sector_ids: torch.Tensor,   # [B, K]
        g_emb:      torch.Tensor,   # [B, D_tcn]  (unused in repr but kept for signature)
        x:          torch.Tensor,   # [B, L, K, F]  (unused, kept for compat)
        g:          torch.Tensor,   # [B, D_g]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (w_pre [B,K], log_prob [B], state_summary [B, D_pool]).
        """
        # §7.5 Actor attention
        attn_out = self.actor_attn(tcn_out, mask, sector_ids)   # [B, K, D]

        # §7.6 Pooling → state_summary
        summary = self.actor_pool(attn_out, mask)               # [B, 2*D]

        # §7.6 Per-asset representation
        asset_repr = self._build_asset_repr(attn_out, g, summary, mask)  # [B, K, D_repr]

        # §7.7 Actor MLP → logits → logistic-normal sampling
        logits = self.actor_mu_head(asset_repr).squeeze(-1)     # [B, K]
        logits = logits - (1.0 - mask) * 1e9                    # mask inactive → −∞

        # Logistic-normal noise (reparameterization)
        sigma = torch.exp(
            self.log_sigma.clamp(self._log_sigma_min, self._log_sigma_max)
        )
        if self.training:
            eps  = torch.randn_like(logits) * mask              # zero noise for inactive
            z    = logits + sigma * eps
        else:
            z    = logits
            eps  = torch.zeros_like(logits)

        # Re-mask and softmax
        z     = z - (1.0 - mask) * 1e9
        w_pre = F.softmax(z, dim=-1)                            # [B, K]

        # Log-prob from pre-projection logistic-normal (active slots only)
        n_active = mask.sum(dim=1).clamp(min=1.0)               # [B]
        log_2pi  = math.log(2.0 * math.pi)
        # per-slot log N(z_k | logit_k, sigma)  only for active slots
        lp_per  = -0.5 * (eps ** 2) - self.log_sigma.clamp(
            self._log_sigma_min, self._log_sigma_max
        ) - 0.5 * log_2pi
        log_prob = (lp_per * mask).sum(dim=1)                   # [B]

        return w_pre, log_prob, summary

    def _critic_branches(
        self,
        tcn_out:    torch.Tensor,   # [B, K, D_tcn]
        mask:       torch.Tensor,   # [B, K]
        sector_ids: torch.Tensor,   # [B, K]
        w_pre:      torch.Tensor,   # [B, K]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (q1 [B, N_q], q2 [B, N_q])."""
        # Q1
        a1  = self.q1_attn(tcn_out, mask, sector_ids)          # [B, K, D]
        s1  = self.q1_pool(a1, mask)                            # [B, D_pool]
        c1  = torch.cat([s1, w_pre], dim=-1)                    # [B, D_pool+K]
        q1  = self.q1_head(c1)                                  # [B, N_quantiles]

        # Q2
        a2  = self.q2_attn(tcn_out, mask, sector_ids)
        s2  = self.q2_pool(a2, mask)
        c2  = torch.cat([s2, w_pre], dim=-1)
        q2  = self.q2_head(c2)                                  # [B, N_quantiles]

        return q1, q2


# ---------------------------------------------------------------------------
# Config-driven constructor
# ---------------------------------------------------------------------------

def from_config(arch_cfg: dict, D_g: int = 20,
                num_tickers: int = 512, num_sectors: int = 12) -> ApexActorCritic:
    """Construct model from the 'architecture' section of master_config.yaml."""
    return ApexActorCritic(
        K_max             = arch_cfg.get("K_max",            110),
        F                 = arch_cfg.get("F",                 25),
        D_g               = D_g,
        num_tickers       = num_tickers,
        num_sectors       = num_sectors,
        ticker_emb_dim    = arch_cfg.get("ticker_emb_dim",    32),
        sector_emb_dim    = arch_cfg.get("sector_emb_dim",     8),
        tcn_channels      = arch_cfg.get("tcn_channels",     128),
        tcn_levels        = arch_cfg.get("tcn_levels",          5),
        tcn_kernel_size   = arch_cfg.get("tcn_kernel_size",     3),
        tcn_dilation_base = arch_cfg.get("tcn_dilation_base",   2),
        attn_d_model      = arch_cfg.get("attn_d_model",      128),
        attn_n_heads      = arch_cfg.get("attn_num_heads",       4),
        attn_d_ff         = arch_cfg.get("attn_d_ff",         256),
        attn_n_layers     = arch_cfg.get("attn_num_layers",      2),
        attn_dropout      = arch_cfg.get("attn_dropout",      0.05),
        actor_hidden_dims = arch_cfg.get("actor_hidden_dims", [128, 128]),
        critic_hidden_dims= arch_cfg.get("critic_hidden_dims",[256, 256]),
        n_quantiles       = arch_cfg.get("n_quantiles",        32),
    )
