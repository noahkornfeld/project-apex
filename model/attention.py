"""
model/attention.py
Cross-Asset Multi-Head Self-Attention — Bible §7.5.

Architecture (per stack):
  N Pre-LN transformer blocks:
    LayerNorm → MHA (sector adjacency bias + key padding mask) → residual
    LayerNorm → Feed-Forward (SiLU) → residual

Three independent stacks are instantiated in ApexActorCritic:
  actor_attn, q1_attn, q2_attn  (same architecture, separate parameters).

Masking:
  - key_padding_mask: inactive slots (mask==0) receive −∞ in attention logits
    so their softmax weight is exactly 0.
  - Slot outputs for inactive slots are explicitly zeroed after each block.

Sector adjacency bias:
  A learned scalar parameter added to attention logits where query and key
  share the same GICS sector code.  Shape [B, K, K] → broadcast over heads.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _MHA(nn.Module):
    """
    Custom multi-head self-attention supporting additive attention bias
    (for sector adjacency) and explicit key padding mask.

    Returns (output, attn_weights) where attn_weights is averaged over heads.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.05):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = self.head_dim ** -0.5

        self.q_proj  = nn.Linear(d_model, d_model)
        self.k_proj  = nn.Linear(d_model, d_model)
        self.v_proj  = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x:         torch.Tensor,            # [B, K, D]
        attn_bias: torch.Tensor | None,     # [B, K, K] additive
        key_mask:  torch.Tensor | None,     # [B, K] bool  True = mask out
    ):
        B, K, D = x.shape
        H, Hd   = self.n_heads, self.head_dim

        Q = self.q_proj(x).view(B, K, H, Hd).transpose(1, 2)   # [B, H, K, Hd]
        Kk = self.k_proj(x).view(B, K, H, Hd).transpose(1, 2)
        V  = self.v_proj(x).view(B, K, H, Hd).transpose(1, 2)

        logits = torch.matmul(Q, Kk.transpose(-2, -1)) * self.scale  # [B, H, K, K]

        if attn_bias is not None:
            logits = logits + attn_bias.unsqueeze(1)          # [B, 1, K, K]

        if key_mask is not None:
            # mask inactive KEY positions  (query dim unchanged)
            logits = logits.masked_fill(
                key_mask.unsqueeze(1).unsqueeze(2),           # [B, 1, 1, K]
                float("-inf"),
            )

        weights = F.softmax(logits, dim=-1)                    # [B, H, K, K]
        # Replace any NaN arising from all-masked rows (shouldn't occur in practice)
        weights = torch.nan_to_num(weights, nan=0.0)
        weights = self.attn_drop(weights)

        out = torch.matmul(weights, V)                         # [B, H, K, Hd]
        out = out.transpose(1, 2).contiguous().view(B, K, D)
        out = self.out_proj(out)

        attn_avg = weights.mean(dim=1)                         # [B, K, K] head-avg
        return out, attn_avg


class _PreLNBlock(nn.Module):
    """Pre-LayerNorm transformer block (§7.5)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = _MHA(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, d_model),
        )
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, attn_bias, key_mask):
        h, attn_w = self.attn(self.norm1(x), attn_bias, key_mask)
        x = x + self.drop(h)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x, attn_w


class CrossAssetAttention(nn.Module):
    """
    Cross-asset multi-head self-attention stack (Bible §7.5).

    Parameters
    ----------
    d_model        : total model dimension  (default 128)
    n_heads        : number of attention heads  (default 4)
    d_ff           : feed-forward inner dimension  (default 256)
    n_layers       : number of stacked Pre-LN blocks  (default 2)
    dropout        : attention + FF dropout  (default 0.05)
    use_sector_bias: add learned sector-adjacency additive bias  (default True)
    """

    def __init__(
        self,
        d_model:         int   = 128,
        n_heads:         int   = 4,
        d_ff:            int   = 256,
        n_layers:        int   = 2,
        dropout:         float = 0.05,
        use_sector_bias: bool  = True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            _PreLNBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.use_sector_bias = use_sector_bias
        if use_sector_bias:
            self.sector_adj_weight = nn.Parameter(torch.zeros(1))

        # Storage for inspection / gate tests
        self._last_attn_weights: torch.Tensor | None = None

    # ------------------------------------------------------------------
    def forward(
        self,
        x:          torch.Tensor,   # [B, K, D]
        mask:       torch.Tensor,   # [B, K] float  1=active 0=inactive
        sector_ids: torch.Tensor,   # [B, K] int64
    ) -> torch.Tensor:
        """Returns [B, K, D] with inactive slots zeroed."""
        B, K, _ = x.shape

        # Key padding mask: True where inactive → attention weight = 0
        key_mask = mask < 0.5                                          # [B, K] bool

        # Sector adjacency bias  [B, K, K]
        attn_bias: torch.Tensor | None = None
        if self.use_sector_bias:
            same = (sector_ids.unsqueeze(2) == sector_ids.unsqueeze(1)).float()
            attn_bias = same * self.sector_adj_weight                  # [B, K, K]

        last_w = None
        for block in self.blocks:
            x, last_w = block(x, attn_bias, key_mask)
            x = x * mask.unsqueeze(-1)                                 # zero inactive

        self._last_attn_weights = last_w   # [B, K, K] head-averaged, last block
        return x
