"""
Causal Normalizers — Bible §3.6
================================
All normalization must be causal (no future statistics) and consistent
across training and inference.

§3.6.1  Per-Asset Normalization
    Running per-asset statistics (mean, std) over a 52-week (260 day)
    sliding window.  z_i = (f_i - running_mean_i) / (running_std_i + ε).

§3.6.2  Cross-Sectional Normalization
    Applied after per-asset normalization for §3.2 features.
    Mean and std computed at each t from active K_active assets.
    (Cross-sectional features are already z-scored during computation.)

§3.6.3  Fixed-Scale Normalization
    For macro features with well-defined scales (VIX, yield spreads).
    Uses long-term historical statistics pre-computed from training data.

§3.6.4  Clipping
    All features clipped to [-norm_clip_threshold, +norm_clip_threshold].
"""

import numpy as np
import pandas as pd
from typing import Optional

EPS = 1e-8
DEFAULT_NORM_WINDOW = 260   # 52 weeks × 5 days
DEFAULT_CLIP        = 4.0


# ---------------------------------------------------------------------------
# Per-Asset Causal Normalizer
# ---------------------------------------------------------------------------

class CausalPerAssetNormalizer:
    """
    Per-asset causal z-score normalizer (§3.6.1).

    Fits a rolling mean + std over `window` days (causal).
    At each time step t, uses only data from [t - window + 1, t].

    Usage:
        norm = CausalPerAssetNormalizer(window=260, clip=4.0)
        z_df = norm.fit_transform(raw_feature_df)   # full-history (causal)
    """

    def __init__(self, window: int = DEFAULT_NORM_WINDOW, clip: float = DEFAULT_CLIP):
        self.window = window
        self.clip   = clip

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply causal rolling z-score normalization to every column.

        Args:
            df : DataFrame(DatetimeIndex, feature_columns).

        Returns:
            Normalized DataFrame (same shape). NaN where window not yet full
            are filled forward then filled with 0 to handle early rows.
        """
        df = df.astype(float)
        rolling_mean = df.rolling(self.window, min_periods=1).mean()
        rolling_std  = df.rolling(self.window, min_periods=2).std()

        z = (df - rolling_mean) / (rolling_std.fillna(EPS) + EPS)

        # Forward-fill NaNs from window warm-up, then fill with 0
        z = z.ffill().fillna(0.0)

        return clip_features(z, self.clip)

    def transform_online(
        self,
        value: float,
        running_mean: float,
        running_std: float,
    ) -> float:
        """
        Normalize a single value at inference time using pre-computed stats.
        """
        z = (value - running_mean) / (running_std + EPS)
        return float(np.clip(z, -self.clip, self.clip))


# ---------------------------------------------------------------------------
# Fixed-Scale Normalizer (macro features)
# ---------------------------------------------------------------------------

class FixedScaleNormalizer:
    """
    Fixed-scale normalizer for macro features (§3.6.3).

    Uses long-term historical mean and std (computed once from training data).
    Same statistics are used at inference time — no refitting on OOS data.
    """

    def __init__(self, clip: float = DEFAULT_CLIP):
        self.clip   = clip
        self._stats: Optional[pd.DataFrame] = None   # columns: [mean, std]

    def fit(self, df: pd.DataFrame) -> "FixedScaleNormalizer":
        """Compute global mean and std from the (training) DataFrame."""
        self._stats = pd.DataFrame({
            "mean": df.mean(skipna=True),
            "std":  df.std(skipna=True),
        })
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fixed-scale normalization using fitted statistics."""
        if self._stats is None:
            raise RuntimeError("Call fit() before transform().")
        mu  = self._stats["mean"]
        std = self._stats["std"].replace(0.0, EPS)
        z   = (df - mu) / (std + EPS)
        z   = z.ffill().fillna(0.0)
        return clip_features(z, self.clip)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# ---------------------------------------------------------------------------
# Cross-sectional post-clip (§3.6.2 — applied to already-z-scored CS feats)
# ---------------------------------------------------------------------------

def clip_cross_sectional(
    cs_values: np.ndarray,
    clip: float = DEFAULT_CLIP,
) -> np.ndarray:
    """Clip cross-sectional feature values to [-clip, +clip]."""
    return np.clip(np.nan_to_num(cs_values, nan=0.0), -clip, clip)


# ---------------------------------------------------------------------------
# Shared clip utility
# ---------------------------------------------------------------------------

def clip_features(
    df: pd.DataFrame,
    clip: float = DEFAULT_CLIP,
) -> pd.DataFrame:
    """
    Clip all values to [-clip, +clip] and replace any residual NaN with 0.
    This is the final step of §3.6.4.
    """
    return df.clip(lower=-clip, upper=clip).fillna(0.0)


# ---------------------------------------------------------------------------
# Utility: normalize a numpy feature matrix row-by-row (at panel-build time)
# ---------------------------------------------------------------------------

def apply_causal_norm_to_matrix(
    matrix: np.ndarray,
    window: int = DEFAULT_NORM_WINDOW,
    clip: float = DEFAULT_CLIP,
) -> np.ndarray:
    """
    Apply per-column causal rolling z-score normalization to a 2-D matrix.

    Args:
        matrix : shape [T, F] — raw feature values over time.
        window : rolling window in time steps.
        clip   : clipping threshold.

    Returns:
        Normalized matrix, same shape, dtype float32.
    """
    T, F = matrix.shape
    out  = np.zeros_like(matrix, dtype=np.float32)

    for f in range(F):
        series = pd.Series(matrix[:, f].astype(float))
        roll_mean = series.rolling(window, min_periods=1).mean()
        roll_std  = series.rolling(window, min_periods=2).std().fillna(EPS)
        z = (series - roll_mean) / (roll_std + EPS)
        z = z.ffill().fillna(0.0).clip(-clip, clip)
        out[:, f] = z.values.astype(np.float32)

    return out
