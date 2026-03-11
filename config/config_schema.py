"""
Master Configuration Schema for Project Apex
============================================

Dataclass representation of all hyperparameters from Bible §0.2 plus
architectural, pipeline, and evaluation parameters from §2–§10.

Provides type validation, range checking, and single-source configuration
loading from the master YAML file.

Usage:
    from config.config_schema import ProjectConfig

    config = ProjectConfig.from_yaml("config/master_config.yaml")
    print(config.sac.gamma)          # 0.975
    print(config.architecture.K_max) # 110
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import math
import warnings
import yaml
from pathlib import Path


class ValidationError(Exception):
    """Raised when configuration validation fails"""
    pass


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _filter_keys(cls, d: dict) -> dict:
    """Return only the keys that ``cls`` accepts, so **d won't blow up."""
    valid = {f.name for f in cls.__dataclass_fields__.values()}
    return {k: v for k, v in d.items() if k in valid}


# ===================================================================
# §8  SAC Training
# ===================================================================

@dataclass
class SACConfig:
    """SAC Training Parameters — Bible §8"""

    # §8.8  Temperature / Alpha
    init_alpha: float = 0.1
    alpha_min: float = 1e-4
    alpha_max: float = 1.0
    alpha_lr: float = 1e-4
    alpha_weight_decay: float = 0.0
    entropy_scale_factor: float = 0.7
    alpha_clamp_min: float = -9.2103       # log(1e-4)
    alpha_clamp_max: float = 0.0           # log(1.0)

    # §8.1  Discount
    gamma: float = 0.975

    # §8.2  N-Step Returns
    n_step: int = 4

    # §8.7  Update Schedule
    updates_per_step: int = 20
    policy_delay: int = 2
    batch_size: int = 64
    grad_accum_steps: int = 1
    mixed_precision: bool = False
    distributed_training: bool = False

    # §8.10  Target Networks (Polyak)
    tau: float = 0.005

    # §8.11  Gradient Clipping
    grad_clip_critic: float = 1.0
    grad_clip_encoder: float = 1.0
    grad_clip_actor: float = 5.0

    def validate(self):
        assert 0 < self.gamma < 1, f"gamma must be in (0,1), got {self.gamma}"
        assert self.alpha_min > 0, f"alpha_min must be > 0, got {self.alpha_min}"
        assert self.alpha_max > self.alpha_min, "alpha_max must be > alpha_min"
        assert self.alpha_min <= self.init_alpha <= self.alpha_max, \
            f"init_alpha must be in [alpha_min, alpha_max]"
        assert self.n_step >= 1, f"n_step must be >= 1, got {self.n_step}"
        assert 5 <= self.updates_per_step <= 35, \
            f"updates_per_step must be in [5, 35], got {self.updates_per_step}"
        assert self.policy_delay >= 1, f"policy_delay must be >= 1"
        assert 0 < self.tau <= 0.1, f"tau must be in (0, 0.1], got {self.tau}"
        assert 0 < self.grad_clip_critic <= 2, "grad_clip_critic must be in (0, 2]"
        assert 0 < self.grad_clip_encoder <= 2, "grad_clip_encoder must be in (0, 2]"
        assert self.grad_clip_actor > 0, "grad_clip_actor must be > 0"
        assert self.alpha_clamp_min < self.alpha_clamp_max, \
            "alpha_clamp_min must be < alpha_clamp_max"


# ===================================================================
# §8.3–8.6  Replay Buffer
# ===================================================================

@dataclass
class ReplayConfig:
    """Replay Buffer Parameters — Bible §8.3"""

    capacity: int = 800
    recency_half_life_years: float = 3.0
    warmup_steps: int = 52
    warmup_exclusion_threshold: int = 128

    # §8.4  Transition Augmentation
    aug_reward_noise_factor: float = 0.01
    aug_obs_noise_factor: float = 0.015

    def validate(self):
        assert self.capacity > 0, "capacity must be > 0"
        assert self.recency_half_life_years > 0, "recency_half_life_years must be > 0"
        assert self.warmup_steps > 0, "warmup_steps must be > 0"
        assert self.warmup_exclusion_threshold > 0, "warmup_exclusion_threshold must be > 0"
        assert self.aug_reward_noise_factor >= 0, "aug_reward_noise_factor must be >= 0"
        assert self.aug_obs_noise_factor >= 0, "aug_obs_noise_factor must be >= 0"


# ===================================================================
# §8.9  Optimizer Configuration
# ===================================================================

@dataclass
class OptimizerConfig:
    """Optimizer Parameters — Bible §8.9"""

    # Critic (Q1, Q2)
    critic_optimizer: str = "adam"
    critic_lr: float = 3e-4
    critic_weight_decay: float = 0.0

    # Actor
    actor_optimizer: str = "adam"
    actor_lr: float = 1e-4
    actor_weight_decay: float = 0.0

    # Encoder (TCN + Attention)
    encoder_optimizer: str = "adamw"
    encoder_lr: float = 3e-4
    encoder_weight_decay: float = 1e-4

    # Alpha
    alpha_optimizer: str = "adam"

    _VALID_OPTIMIZERS = {"adam", "adamw", "sgd"}

    def validate(self):
        for name in ("critic", "actor", "encoder", "alpha"):
            opt = getattr(self, f"{name}_optimizer")
            assert opt in self._VALID_OPTIMIZERS, \
                f"{name}_optimizer '{opt}' not in {self._VALID_OPTIMIZERS}"
        assert self.critic_lr > 0, "critic_lr must be > 0"
        assert self.actor_lr > 0, "actor_lr must be > 0"
        assert self.encoder_lr > 0, "encoder_lr must be > 0"
        assert self.critic_weight_decay >= 0, "critic_weight_decay must be >= 0"
        assert self.actor_weight_decay >= 0, "actor_weight_decay must be >= 0"
        assert self.encoder_weight_decay >= 0, "encoder_weight_decay must be >= 0"


# ===================================================================
# §4, §7  Model Architecture
# ===================================================================

@dataclass
class ArchitectureConfig:
    """Model Architecture Parameters — Bible §4, §7"""

    # §4.1  Observation Space Dimensions
    K_max: int = 110
    L: int = 60                           # Lookback in days (§0.1 notation)
    F: int = 25                           # 17 TS + 8 CS features

    # §7.4  TCN Encoder
    tcn_levels: int = 5
    tcn_kernel_size: int = 3
    tcn_channels: int = 128
    tcn_dilation_base: int = 2
    tcn_activation: str = "silu"
    tcn_normalization: str = "layernorm"
    tcn_skip_connections: bool = True
    tcn_causal: bool = True

    # §7.5  Cross-Asset Attention
    attn_num_heads: int = 4
    attn_d_model: int = 128
    attn_d_ff: int = 256
    attn_dropout: float = 0.05
    attn_num_layers: int = 2
    attn_padding_mask: bool = True
    attn_sector_adjacency_bias: bool = True

    # §7.3  Embeddings
    ticker_emb_dim: int = 32
    sector_emb_dim: int = 8

    # §7.6  Actor Head MLP
    actor_hidden_dims: List[int] = field(default_factory=lambda: [128, 128])

    # §7.8  Critic Head MLP
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    n_quantiles: int = 32

    def validate(self):
        assert self.K_max > 0, "K_max must be > 0"
        assert self.L > 0, "L must be > 0"
        assert self.F > 0, "F must be > 0"
        assert self.tcn_levels > 0, "tcn_levels must be > 0"
        assert self.tcn_kernel_size > 0, "tcn_kernel_size must be > 0"
        assert self.tcn_channels > 0, "tcn_channels must be > 0"
        assert self.attn_num_heads > 0, "attn_num_heads must be > 0"
        assert self.attn_d_model > 0, "attn_d_model must be > 0"
        assert self.attn_d_model % self.attn_num_heads == 0, \
            f"attn_d_model ({self.attn_d_model}) must be divisible by attn_num_heads ({self.attn_num_heads})"
        assert self.attn_d_ff > 0, "attn_d_ff must be > 0"
        assert 0 <= self.attn_dropout < 1, "attn_dropout must be in [0, 1)"
        assert self.attn_num_layers >= 1, "attn_num_layers must be >= 1"
        assert self.n_quantiles > 0, "n_quantiles must be > 0"
        assert self.ticker_emb_dim > 0, "ticker_emb_dim must be > 0"
        assert self.sector_emb_dim > 0, "sector_emb_dim must be > 0"
        assert len(self.actor_hidden_dims) > 0, "actor_hidden_dims must not be empty"
        assert len(self.critic_hidden_dims) > 0, "critic_hidden_dims must not be empty"

        # TCN receptive field check: 1 + (k-1) × (2^levels - 1) ≥ L
        receptive_field = 1 + (self.tcn_kernel_size - 1) * (2**self.tcn_levels - 1)
        assert receptive_field >= self.L, \
            f"TCN receptive field ({receptive_field}) must be >= L ({self.L})"

        # Activation and normalization
        assert self.tcn_activation in ("silu", "relu", "gelu"), \
            f"tcn_activation '{self.tcn_activation}' not in (silu, relu, gelu)"
        assert self.tcn_normalization in ("layernorm", "batchnorm", "none"), \
            f"tcn_normalization '{self.tcn_normalization}' not in (layernorm, batchnorm, none)"


# ===================================================================
# §6  Reward Function
# ===================================================================

@dataclass
class RewardConfig:
    """Reward Function Parameters — Bible §6"""

    # §6  Penalty Weights
    lambda_slow: float = 0.75
    lambda_tail: float = 0.4
    lambda_cost: float = 1.0
    lambda_cv: float = 1.0

    # Tuning guidance ranges (from §6 reward-shaping table)
    lambda_slow_range: List[float] = field(default_factory=lambda: [0.5, 1.25])
    lambda_tail_range: List[float] = field(default_factory=lambda: [0.3, 0.5])

    # §6  Rolling Volatility Windows
    sigma_mkt_window_weeks: int = 13       # σ_mkt,t causal window
    sigma_port_window_weeks: int = 52      # σ_t causal window

    def validate(self):
        assert self.lambda_slow >= 0, "lambda_slow must be >= 0"
        assert self.lambda_tail >= 0, "lambda_tail must be >= 0"
        assert self.lambda_cost >= 0, "lambda_cost must be >= 0"
        assert self.lambda_cv >= 0, "lambda_cv must be >= 0"
        assert self.sigma_mkt_window_weeks > 0, "sigma_mkt_window_weeks must be > 0"
        assert self.sigma_port_window_weeks > 0, "sigma_port_window_weeks must be > 0"

        # Tuning guidance warnings
        lo, hi = self.lambda_slow_range
        if not (lo <= self.lambda_slow <= hi):
            warnings.warn(
                f"lambda_slow ({self.lambda_slow}) outside recommended range [{lo}, {hi}]")
        lo, hi = self.lambda_tail_range
        if not (lo <= self.lambda_tail <= hi):
            warnings.warn(
                f"lambda_tail ({self.lambda_tail}) outside recommended range [{lo}, {hi}]")


# ===================================================================
# §5.4  Transaction Cost Model  (all "fixed" in §0.2)
# ===================================================================

@dataclass
class TransactionCostConfig:
    """Transaction Cost Model Parameters — Bible §5.4"""

    commission_bps: float = 1.0

    # Spread component
    spread_coeff: float = 2.0
    spread_adv_exp: float = 0.3
    spread_vol_floor: float = 0.20

    # Market-impact component
    impact_coeff: float = 10.0
    impact_size_exp: float = 0.5

    # Gap-risk component
    gap_coeff: float = 1.5
    gap_scaling: float = 0.7979           # √(2/π)

    def validate(self):
        assert self.commission_bps >= 0, "commission_bps must be >= 0"
        assert self.spread_coeff >= 0, "spread_coeff must be >= 0"
        assert self.impact_coeff >= 0, "impact_coeff must be >= 0"
        assert self.gap_coeff >= 0, "gap_coeff must be >= 0"
        # Verify gap_scaling ≈ √(2/π)
        expected = math.sqrt(2.0 / math.pi)
        assert abs(self.gap_scaling - expected) < 0.01, \
            f"gap_scaling should be ≈ √(2/π) = {expected:.4f}, got {self.gap_scaling}"


# ===================================================================
# §4.5  Portfolio Constraints
# ===================================================================

@dataclass
class ConstraintConfig:
    """Portfolio Constraint Parameters — Bible §4.5"""

    per_name_cap: float = 0.20
    sector_cap: float = 0.50

    def validate(self):
        assert 0 < self.per_name_cap < 1, "per_name_cap must be in (0, 1)"
        assert 0 < self.sector_cap < 1, "sector_cap must be in (0, 1)"


# ===================================================================
# §3  Feature Engineering
# ===================================================================

@dataclass
class FeatureConfig:
    """Feature Engineering Parameters — Bible §3"""

    # §3.6  Normalization Strategy
    norm_window_weeks: int = 52
    norm_clip_threshold: float = 4.0

    # §3.1.1  Per-Asset Time-Series Features  (F_ts = 17)
    per_asset_ts_features: List[str] = field(default_factory=lambda: [
        "open", "close", "volume", "log_ret",
        "ret_1w", "ret_4w", "ret_13w",
        "vol_1w", "vol_4w", "vol_52w",
        "volume_z_4w", "beta_26w_mkt", "rel_strength_4w",
        "vol_ratio_1w_4w", "RSI_14",
        "bollinger_percent_b", "bollinger_bandwidth",
    ])

    # §3.2.1  Cross-Sectional Features  (F_cs = 8)
    cross_sectional_features: List[str] = field(default_factory=lambda: [
        "ret_rank_4w", "ret_z_4w", "ret_z_13w", "vol_z_4w",
        "volume_z_cs_4w", "ret_z_4w_sector", "vol_z_4w_sector",
        "momentum_sector_residual",
    ])

    # §3.3.1  Macro / Broadcast Instruments
    macro_instruments: List[Dict[str, str]] = field(default_factory=lambda: [
        {"name": "QQQ",          "symbol": "QQQ",      "purpose": "Benchmark regime"},
        {"name": "VIX",          "symbol": "^VIX",     "purpose": "Equity volatility regime"},
        {"name": "10Y Yield",    "symbol": "^TNX",     "purpose": "Discount rate"},
        {"name": "3M Yield",     "symbol": "^IRX",     "purpose": "Short rate"},
        {"name": "Yield Spread", "symbol": "^TNX-^IRX", "purpose": "Cycle phase"},
        {"name": "Oil",          "symbol": "CL=F",     "purpose": "Inflation shock"},
        {"name": "Gold",         "symbol": "GC=F",     "purpose": "Monetary stress"},
        {"name": "Dollar Index", "symbol": "DX-Y.NYB", "purpose": "Liquidity regime"},
        {"name": "High Yield ETF", "symbol": "HYG",    "purpose": "Credit stress"},
    ])

    # §3.4.1  Portfolio-State Features  (included in g_t)
    portfolio_state_features: List[Dict[str, str]] = field(default_factory=lambda: [
        {"name": "Turnover Last Step",            "window": "1 step",   "normalization": "global z-score"},
        {"name": "Realized Portfolio Volatility",  "window": "13 weeks", "normalization": "global z-score"},
        {"name": "Current Drawdown",              "window": "episode",  "normalization": "scaled [-1, 0]"},
        {"name": "Gross Exposure",                "window": "1 step",   "normalization": "none (≈1.0)"},
        {"name": "Rolling Excess Return vs QQQ",  "window": "13 weeks", "normalization": "global z-score"},
        {"name": "Estimated Cost Next Step",      "window": "1 step",   "normalization": "global z-score"},
        {"name": "Effective # of Positions",      "window": "1 step",   "normalization": "global z-score"},
        {"name": "Market Volatility Regime",      "window": "52 weeks", "normalization": "global z-score"},
    ])

    # §3.5.1  Benchmark Features  (included in g_t)
    benchmark_features: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"name": "QQQ 1-Week Return",     "window_days": 5,  "computed_from": "log return adj close"},
        {"name": "QQQ 4-Week Volatility", "window_days": 20, "computed_from": "std daily log ret × √252"},
        {"name": "QQQ 12-Week Return",    "window_days": 60, "computed_from": "log return adj close"},
    ])

    def validate(self):
        assert self.norm_window_weeks > 0, "norm_window_weeks must be > 0"
        assert self.norm_clip_threshold > 0, "norm_clip_threshold must be > 0"
        assert len(self.per_asset_ts_features) == 17, \
            f"Expected 17 per-asset TS features (Bible §3.1.1), got {len(self.per_asset_ts_features)}"
        assert len(self.cross_sectional_features) == 8, \
            f"Expected 8 cross-sectional features (Bible §3.2.1), got {len(self.cross_sectional_features)}"
        assert len(self.macro_instruments) == 9, \
            f"Expected 9 macro instruments (Bible §3.3.1), got {len(self.macro_instruments)}"
        assert len(self.portfolio_state_features) == 8, \
            f"Expected 8 portfolio-state features (Bible §3.4.1), got {len(self.portfolio_state_features)}"
        assert len(self.benchmark_features) == 3, \
            f"Expected 3 benchmark features (Bible §3.5.1), got {len(self.benchmark_features)}"


# ===================================================================
# §9  Evaluation Framework
# ===================================================================

@dataclass
class FoldConfig:
    """Single walk-forward fold definition — Bible §9.1"""
    fold: int = 1
    train_start: str = "2005-01-01"
    train_end: str = "2009-12-31"
    test_start: str = "2010-01-01"
    test_end: str = "2011-12-31"


@dataclass
class EvaluationConfig:
    """Evaluation Parameters — Bible §9"""

    n_folds: int = 8
    embargo_weeks: int = 4
    bootstrap_resamples: int = 10000
    ci_level: float = 0.95
    equal_weight_threshold: float = 0.01

    # §9.1  Fold Definitions
    folds: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"fold": 1, "train_start": "2005-01-01", "train_end": "2009-12-31",
         "test_start": "2010-01-01", "test_end": "2011-12-31"},
        {"fold": 2, "train_start": "2006-01-01", "train_end": "2011-12-31",
         "test_start": "2012-01-01", "test_end": "2013-12-31"},
        {"fold": 3, "train_start": "2008-01-01", "train_end": "2013-12-31",
         "test_start": "2014-01-01", "test_end": "2015-12-31"},
        {"fold": 4, "train_start": "2010-01-01", "train_end": "2015-12-31",
         "test_start": "2016-01-01", "test_end": "2017-12-31"},
        {"fold": 5, "train_start": "2012-01-01", "train_end": "2017-12-31",
         "test_start": "2018-01-01", "test_end": "2019-12-31"},
        {"fold": 6, "train_start": "2014-01-01", "train_end": "2019-12-31",
         "test_start": "2020-01-01", "test_end": "2021-12-31"},
        {"fold": 7, "train_start": "2016-01-01", "train_end": "2021-12-31",
         "test_start": "2022-01-01", "test_end": "2023-12-31"},
        {"fold": 8, "train_start": "2018-01-01", "train_end": "2023-12-31",
         "test_start": "2024-01-01", "test_end": "present"},
    ])

    # §1.2  Success Criteria
    success_criteria: Optional[Dict[str, Any]] = None

    def validate(self):
        assert self.n_folds > 0, "n_folds must be > 0"
        assert self.embargo_weeks > 0, "embargo_weeks must be > 0"
        assert self.bootstrap_resamples > 0, "bootstrap_resamples must be > 0"
        assert 0 < self.ci_level < 1, "ci_level must be in (0, 1)"
        assert self.equal_weight_threshold > 0, "equal_weight_threshold must be > 0"
        assert len(self.folds) == self.n_folds, \
            f"Expected {self.n_folds} fold definitions, got {len(self.folds)}"


# ===================================================================
# §8.5  Collection
# ===================================================================

@dataclass
class CollectionConfig:
    """Multi-Episode Collection — Bible §8.5"""
    n_episodes_per_fold: int = 3

    def validate(self):
        assert self.n_episodes_per_fold > 0, "n_episodes_per_fold must be > 0"


# ===================================================================
# §10  Logging & Observability
# ===================================================================

@dataclass
class LoggingConfig:
    """Logging Parameters — Bible §10"""

    update_cadence: int = 250

    # §10.7  Cadence Specification
    cadences: Optional[Dict[str, int]] = None

    # §10.6  Regression Alarms
    alarms: Optional[Dict[str, Any]] = None

    # §8.12  Stability Diagnostics
    stability_diagnostics: Optional[Dict[str, Any]] = None

    def validate(self):
        assert self.update_cadence > 0, "update_cadence must be > 0"


# ===================================================================
# §4.1  Observation Space Tensor Specs
# ===================================================================

@dataclass
class ObservationSpaceConfig:
    """Observation Space Tensor Specifications — Bible §4.1"""

    x_t: Optional[Dict[str, Any]] = None
    g_t: Optional[Dict[str, Any]] = None
    mask_t: Optional[Dict[str, Any]] = None
    active_ids_t: Optional[Dict[str, Any]] = None

    def validate(self):
        pass  # Reference data; no range validation needed


# ===================================================================
# §2  Data Pipeline
# ===================================================================

@dataclass
class DataConfig:
    """Data Pipeline Parameters — Bible §2"""

    data_dir: str = "Ticker_Data"
    daily_bars_file: str = "daily_bars.parquet"
    ndx_membership_file: str = "ndx_membership.parquet"
    macro_features_file: str = "macro_features.parquet"
    trading_calendar_file: str = "trading_calendar.parquet"
    ticker_alias_file: str = "ticker_alias.parquet"

    # §2.1  Schemas (reference data)
    daily_bars_schema: Optional[List[Dict[str, str]]] = None
    ndx_membership_schema: Optional[List[Dict[str, str]]] = None

    def validate(self):
        assert self.data_dir, "data_dir must not be empty"


# ===================================================================
# §5.7  Missingness Handling
# ===================================================================

@dataclass
class MissingnessConfig:
    """Missingness Handling Parameters — Bible §5.7"""

    missing_in_window_threshold: int = 2
    consecutive_missing_threshold: int = 3
    temporal_window_days: int = 60

    def validate(self):
        assert self.missing_in_window_threshold > 0, "missing_in_window_threshold must be > 0"
        assert self.consecutive_missing_threshold > 0, "consecutive_missing_threshold must be > 0"
        assert self.temporal_window_days > 0, "temporal_window_days must be > 0"


# ===================================================================
# Random Seed & Determinism
# ===================================================================

@dataclass
class RandomSeedConfig:
    """Random Seed Configuration"""

    use_deterministic: bool = True
    base_seed: int = 42

    def validate(self):
        assert self.base_seed >= 0, "base_seed must be >= 0"


# ===================================================================
# §13  Unresolved Items
# ===================================================================

@dataclass
class UnresolvedConfig:
    """Unresolved Parameters — placeholders requiring team decision (Bible §13)"""

    lr_scheduler: Optional[str] = None
    D_global: Optional[int] = None
    K_max_note: Optional[str] = None
    attn_num_layers_note: Optional[str] = None
    sigma_mkt_window_note: Optional[str] = None
    sigma_port_window_note: Optional[str] = None
    updates_per_step_note: Optional[str] = None

    def validate(self):
        pass  # Placeholders; no validation


# ===================================================================
# Metadata
# ===================================================================

@dataclass
class MetadataConfig:
    """Configuration Metadata"""

    config_version: str = "3.0"
    bible_version: str = "v5"
    last_updated: str = "2026-03-05"
    description: str = "Master configuration — Project Apex"
    bible_sections_covered: Optional[List[str]] = None

    def validate(self):
        pass


# ===================================================================
# Top-level Project Config
# ===================================================================

@dataclass
class ProjectConfig:
    """
    Master Configuration for Project Apex

    Single-source YAML/dataclass holding every hyperparameter from Bible §0.2,
    plus architectural, pipeline, and evaluation parameters from §2–§10.
    Validates on load (types, ranges).  Includes all [UNRESOLVED] flags as
    explicit placeholders.
    """

    sac: SACConfig = field(default_factory=SACConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    transaction_costs: TransactionCostConfig = field(default_factory=TransactionCostConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    observation_space: ObservationSpaceConfig = field(default_factory=ObservationSpaceConfig)
    data: DataConfig = field(default_factory=DataConfig)
    missingness: MissingnessConfig = field(default_factory=MissingnessConfig)
    random_seed: RandomSeedConfig = field(default_factory=RandomSeedConfig)
    unresolved: UnresolvedConfig = field(default_factory=UnresolvedConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self):
        """Validate all sections and cross-parameter constraints."""
        self.sac.validate()
        self.replay.validate()
        self.optimizer.validate()
        self.architecture.validate()
        self.reward.validate()
        self.transaction_costs.validate()
        self.constraints.validate()
        self.features.validate()
        self.evaluation.validate()
        self.collection.validate()
        self.logging.validate()
        self.observation_space.validate()
        self.data.validate()
        self.missingness.validate()
        self.random_seed.validate()
        self.unresolved.validate()
        self.metadata.validate()

        # --- Cross-parameter validation (Bible §9.1) ---
        assert self.sac.n_step == self.evaluation.embargo_weeks, \
            f"embargo_weeks ({self.evaluation.embargo_weeks}) must equal " \
            f"n_step ({self.sac.n_step}): embargo_weeks must equal n_step to prevent " \
            f"n-step returns from crossing fold boundaries (§9.1)"

        # Feature count consistency: F = len(TS) + len(CS)
        expected_F = (len(self.features.per_asset_ts_features)
                      + len(self.features.cross_sectional_features))
        assert self.architecture.F == expected_F, \
            f"architecture.F ({self.architecture.F}) must equal " \
            f"len(TS) + len(CS) = {expected_F}"

    # ------------------------------------------------------------------
    # YAML I/O
    # ------------------------------------------------------------------

    # Map from YAML section key → dataclass type
    _SECTION_MAP = {
        "sac":               SACConfig,
        "replay":            ReplayConfig,
        "optimizer":         OptimizerConfig,
        "architecture":      ArchitectureConfig,
        "reward":            RewardConfig,
        "transaction_costs": TransactionCostConfig,
        "constraints":       ConstraintConfig,
        "features":          FeatureConfig,
        "evaluation":        EvaluationConfig,
        "collection":        CollectionConfig,
        "logging":           LoggingConfig,
        "observation_space": ObservationSpaceConfig,
        "data":              DataConfig,
        "missingness":       MissingnessConfig,
        "random_seed":       RandomSeedConfig,
        "unresolved":        UnresolvedConfig,
        "metadata":          MetadataConfig,
    }

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ProjectConfig':
        """Load configuration from YAML file with validation.

        Raises:
            FileNotFoundError: If *yaml_path* does not exist.
            ValidationError / AssertionError: If any parameter is invalid.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            raw = yaml.safe_load(f)

        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> 'ProjectConfig':
        """Build a validated ProjectConfig from a plain dictionary."""
        kwargs = {}
        for section_key, section_cls in cls._SECTION_MAP.items():
            section_data = raw.get(section_key, {})
            if section_data is None:
                section_data = {}
            kwargs[section_key] = section_cls(**_filter_keys(section_cls, section_data))

        config = cls(**kwargs)
        config.validate()
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a plain dictionary."""
        return asdict(self)

    def save_yaml(self, yaml_path: str):
        """Write config to YAML."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
