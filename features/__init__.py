"""
Phase 3: Feature Engineering Package
=====================================
Bible §3 — Feature Engineering

Modules:
    per_asset_features      §3.1  Per-asset time-series features (17 features)
    cross_sectional_features §3.2  Cross-sectional z-scores and ranks (8 features)
    macro_broadcast_features §3.3  Macro/broadcast features → g_t
    portfolio_state_features §3.4  Portfolio-state features → g_t [STUB]
    benchmark_features       §3.5  Benchmark (QQQ) features → g_t
    normalizers              §3.6  Causal per-asset and cross-sectional normalizers
    feature_panel            Orchestrates full [T, K_max, F] panel precomputation
"""

from features.per_asset_features import compute_per_asset_features
from features.cross_sectional_features import compute_cross_sectional_features
from features.macro_broadcast_features import compute_macro_broadcast_features
from features.benchmark_features import compute_benchmark_features
from features.portfolio_state_features import (
    compute_portfolio_state,
    compute_portfolio_state_stub,
)
from features.normalizers import CausalPerAssetNormalizer, clip_features
from features.feature_panel import FeaturePanelBuilder

__all__ = [
    "compute_per_asset_features",
    "compute_cross_sectional_features",
    "compute_macro_broadcast_features",
    "compute_benchmark_features",
    "compute_portfolio_state_stub",
    "CausalPerAssetNormalizer",
    "clip_features",
    "FeaturePanelBuilder",
]

TS_FEATURE_NAMES = [
    "open", "close", "volume", "log_ret",
    "ret_1w", "ret_4w", "ret_13w",
    "vol_1w", "vol_4w", "vol_52w",
    "volume_z_4w", "beta_26w_mkt", "rel_strength_4w",
    "vol_ratio_1w_4w", "RSI_14",
    "bollinger_percent_b", "bollinger_bandwidth",
]

CS_FEATURE_NAMES = [
    "ret_rank_4w", "ret_z_4w", "ret_z_13w", "vol_z_4w",
    "volume_z_cs_4w", "ret_z_4w_sector", "vol_z_4w_sector",
    "momentum_sector_residual",
]

ALL_ASSET_FEATURE_NAMES = TS_FEATURE_NAMES + CS_FEATURE_NAMES

F_TS = len(TS_FEATURE_NAMES)   # 17
F_CS = len(CS_FEATURE_NAMES)   # 8
F_TOTAL = F_TS + F_CS           # 25
