"""
integration/ablation_stubs.py
==============================
Ablation switches for Project Apex (Bible §11.4 / Table 54).

Implements ablation configuration and model patching stubs.
Do NOT run full ablations here — stubs only (Phase 12 spec).

Ablations defined in Table 54:
  1. Ticker Embedding Ablation  — replace ticker embeddings with zeros
  2. Sector Embedding Ablation  — replace sector embeddings with zeros
  3. Cross-Asset Attention Ablation — bypass CrossAssetAttention (pass-through)
  4. Sector Adjacency Bias Ablation — disable sector bias in attention
  5. n-step Return Ablation     — use n_step=1 instead of n_step=4
  6. Downside Deviation Ablation — set λ_dd=0 (pure excess return reward)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Ablation configuration
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    """
    §11.4 ablation switches.  All default to False (no ablation = full model).

    Parameters
    ----------
    zero_ticker_embeddings        : replace ticker_emb weights with zeros
    zero_sector_embeddings        : replace sector_emb weights with zeros
    disable_cross_asset_attention : bypass all CrossAssetAttention modules
    disable_sector_adjacency_bias : zero out sector bias in attention
    n_step_one                    : use n_step=1 (1-step TD) instead of n_step=4
    zero_downside_penalty         : set λ_dd=0 (pure excess return reward)
    """

    zero_ticker_embeddings:        bool = False
    zero_sector_embeddings:        bool = False
    disable_cross_asset_attention: bool = False
    disable_sector_adjacency_bias: bool = False
    n_step_one:                    bool = False
    zero_downside_penalty:         bool = False

    def any_active(self) -> bool:
        """Return True if any ablation switch is active."""
        return any([
            self.zero_ticker_embeddings,
            self.zero_sector_embeddings,
            self.disable_cross_asset_attention,
            self.disable_sector_adjacency_bias,
            self.n_step_one,
            self.zero_downside_penalty,
        ])

    def active_ablations(self) -> Dict[str, bool]:
        """Return dict of only the active (True) ablation switches."""
        return {
            k: v for k, v in {
                "zero_ticker_embeddings":        self.zero_ticker_embeddings,
                "zero_sector_embeddings":        self.zero_sector_embeddings,
                "disable_cross_asset_attention": self.disable_cross_asset_attention,
                "disable_sector_adjacency_bias": self.disable_sector_adjacency_bias,
                "n_step_one":                    self.n_step_one,
                "zero_downside_penalty":         self.zero_downside_penalty,
            }.items()
            if v
        }

    def __repr__(self) -> str:
        active = self.active_ablations()
        if not active:
            return "AblationConfig(no ablations active — full model)"
        return f"AblationConfig({', '.join(active.keys())})"


# ---------------------------------------------------------------------------
# Ablation applier
# ---------------------------------------------------------------------------

class AblationApplier:
    """
    Applies ablation patches to a model and/or config objects in-place.

    Usage
    -----
    applier = AblationApplier(AblationConfig(zero_ticker_embeddings=True))
    patched_model = applier.apply_to_model(model)
    """

    def __init__(self, config: AblationConfig) -> None:
        self.config = config

    def apply_to_model(self, model: nn.Module, inplace: bool = False) -> nn.Module:
        """
        Apply model-level ablations (embeddings, attention) to the given model.

        Parameters
        ----------
        model   : ApexActorCritic instance
        inplace : if False (default), operates on a deep copy

        Returns
        -------
        Patched model (copy unless inplace=True).
        """
        if not inplace:
            model = copy.deepcopy(model)

        cfg = self.config

        # ------------------------------------------------------------------
        # 1. Ticker Embedding Ablation (Table 54)
        #    Replace ticker_emb weights with zeros → no per-ticker info
        # ------------------------------------------------------------------
        if cfg.zero_ticker_embeddings:
            if hasattr(model, "ticker_emb"):
                with torch.no_grad():
                    model.ticker_emb.weight.zero_()
            else:
                raise AttributeError(
                    "Model has no 'ticker_emb' attribute. "
                    "Cannot apply zero_ticker_embeddings ablation."
                )

        # ------------------------------------------------------------------
        # 2. Sector Embedding Ablation (Table 54)
        #    Replace sector_emb weights with zeros
        # ------------------------------------------------------------------
        if cfg.zero_sector_embeddings:
            if hasattr(model, "sector_emb"):
                with torch.no_grad():
                    model.sector_emb.weight.zero_()
            else:
                raise AttributeError(
                    "Model has no 'sector_emb' attribute. "
                    "Cannot apply zero_sector_embeddings ablation."
                )

        # ------------------------------------------------------------------
        # 3. Cross-Asset Attention Ablation (Table 54)
        #    Patch all CrossAssetAttention modules to be identity pass-through
        # ------------------------------------------------------------------
        if cfg.disable_cross_asset_attention:
            for name in ("actor_attn", "q1_attn", "q2_attn"):
                if hasattr(model, name):
                    setattr(model, name, _IdentityAttention())
                else:
                    raise AttributeError(
                        f"Model has no '{name}' attribute. "
                        "Cannot apply disable_cross_asset_attention ablation."
                    )

        # ------------------------------------------------------------------
        # 4. Sector Adjacency Bias Ablation (Table 54)
        #    Zero out sector_bias in all CrossAssetAttention modules
        # ------------------------------------------------------------------
        if cfg.disable_sector_adjacency_bias:
            for name in ("actor_attn", "q1_attn", "q2_attn"):
                module = getattr(model, name, None)
                if module is None:
                    continue
                _zero_sector_bias(module)

        return model

    def get_replay_n_step(self, original_n_step: int = 4) -> int:
        """
        5. n-step Return Ablation (Table 54): return n_step=1 if active.
        """
        if self.config.n_step_one:
            return 1
        return original_n_step

    def get_reward_lambda_dd(self, original_lambda_dd: float) -> float:
        """
        6. Downside Deviation Ablation (Table 54): return λ_dd=0.0 if active.
        """
        if self.config.zero_downside_penalty:
            return 0.0
        return original_lambda_dd


# ---------------------------------------------------------------------------
# Helper modules / functions
# ---------------------------------------------------------------------------

class _IdentityAttention(nn.Module):
    """
    Drop-in replacement for CrossAssetAttention that passes input through
    unchanged (no cross-asset interaction).  Used for ablation #3.
    """

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Return x unchanged regardless of attention arguments."""
        return x


def _zero_sector_bias(module: nn.Module) -> None:
    """
    Recursively zero out any parameter or buffer named 'sector_bias'
    in the given module and all its children.  Used for ablation #4.
    """
    for name, param in module.named_parameters(recurse=True):
        if "sector_bias" in name:
            with torch.no_grad():
                param.zero_()
    for name, buf in module.named_buffers(recurse=True):
        if "sector_bias" in name:
            buf.zero_()


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

ABLATION_REGISTRY: Dict[str, AblationConfig] = {
    "full_model":                  AblationConfig(),
    "no_ticker_emb":               AblationConfig(zero_ticker_embeddings=True),
    "no_sector_emb":               AblationConfig(zero_sector_embeddings=True),
    "no_cross_attn":               AblationConfig(disable_cross_asset_attention=True),
    "no_sector_adj_bias":          AblationConfig(disable_sector_adjacency_bias=True),
    "n_step_1":                    AblationConfig(n_step_one=True),
    "no_downside_penalty":         AblationConfig(zero_downside_penalty=True),
}


def get_ablation(name: str) -> AblationConfig:
    """
    Retrieve a named ablation config from the registry.

    Parameters
    ----------
    name : one of the keys in ABLATION_REGISTRY

    Returns
    -------
    AblationConfig
    """
    if name not in ABLATION_REGISTRY:
        raise KeyError(
            f"Unknown ablation '{name}'. "
            f"Available: {sorted(ABLATION_REGISTRY.keys())}"
        )
    return ABLATION_REGISTRY[name]
