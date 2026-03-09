"""
model/tcn.py
Causal Temporal Convolutional Network encoder — Bible §7.4.

Architecture:
  Input projection  : Conv1d(in_channels, channels, 1)
  N residual blocks : each block has two causal dilated Conv1d layers, Pre-LN,
                      SiLU activation, and a 1×1 skip connection.
  Receptive field   : 1 + (kernel_size − 1) × (dilation_base^levels − 1)
                      With levels=5, kernel=3, base=2 → RF = 63 ≥ 60 = L.

Causality: strictly left-only padding ((kernel−1)×dilation zeros prepended).
No future-timestep information can leak into the current output.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CausalConv1d(nn.Module):
    """Single causal dilated conv (left-only padding, no future leakage)."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        self._pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self._pad, 0))
        return self.conv(x)


class _TCNResBlock(nn.Module):
    """
    Pre-LN residual block (§7.4):
      LayerNorm → CausalConv → SiLU → CausalConv → + skip
    Skip connection uses 1×1 conv if in_channels ≠ out_channels.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int):
        super().__init__()
        self.norm  = nn.LayerNorm(in_channels)
        self.conv1 = _CausalConv1d(in_channels,  out_channels, kernel_size, dilation)
        self.conv2 = _CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.skip  = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]  (channel-first)
        h = self.norm(x.transpose(1, 2)).transpose(1, 2)   # Pre-LN on channel dim
        h = F.silu(self.conv1(h))
        h = self.conv2(h)
        return F.silu(h + self.skip(x))


class CausalTCN(nn.Module):
    """
    Causal dilated-conv TCN encoder (Bible §7.4).

    Processes each asset's [L, F+D_emb] history independently.
    Call signature (flat-batch):  x [B*K, in_channels, L] → [B*K, channels, L]

    Parameters
    ----------
    in_channels  : C_in  (= F + D_emb after embedding projection)
    channels     : D_tcn output channels per layer  (default 128)
    levels       : number of residual blocks  (default 5)
    kernel_size  : conv kernel k  (default 3)
    dilation_base: d_l = dilation_base^l  (default 2)
    """

    def __init__(
        self,
        in_channels:   int,
        channels:      int = 128,
        levels:        int = 5,
        kernel_size:   int = 3,
        dilation_base: int = 2,
    ):
        super().__init__()
        self._channels     = channels
        self._levels       = levels
        self._kernel_size  = kernel_size
        self._dilation_base = dilation_base

        self.input_proj = nn.Conv1d(in_channels, channels, 1)

        self.blocks = nn.ModuleList([
            _TCNResBlock(
                channels, channels,
                kernel_size, dilation_base ** l,
            )
            for l in range(levels)
        ])

        # Analytic receptive field (§7.4)
        total_dilation = sum(dilation_base ** l for l in range(levels))
        self._rf = 1 + (kernel_size - 1) * total_dilation

    # ------------------------------------------------------------------
    @property
    def receptive_field(self) -> int:
        """Analytic receptive field (must be ≥ L)."""
        return self._rf

    @property
    def out_channels(self) -> int:
        return self._channels

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [N, in_channels, L]  (N = B*K in the main pipeline)
        returns [N, channels, L]
        """
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return x
