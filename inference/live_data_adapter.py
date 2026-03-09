"""
inference/live_data_adapter.py
================================
Live data adapter for Project Apex inference (Bible §12.4).

Responsibilities:
  - Fetch adjusted prices and NDX membership on rebalance morning (Monday open).
  - Verify adjustment factors are up-to-date.
  - Construct the observation tensor (x_panel, g_panel, mask) from live data.
  - Expose a stub interface that can be backed by a real data provider or
    a synthetic/test implementation.
"""

from __future__ import annotations

import datetime
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LiveBar:
    """A single weekly bar for one asset."""
    ticker:      str
    date:        datetime.date
    adj_close:   float
    adj_open:    float
    adv63:       float       # 63-day average dollar volume
    vol_252:     float       # 252-day annualised volatility
    gap_vol_252: float       # 252-day gap volatility
    is_stale:    bool = False


@dataclass
class LiveUniverse:
    """NDX membership snapshot for a given date."""
    as_of_date:  datetime.date
    tickers:     List[str]
    sector_ids:  Dict[str, int]   # ticker → GICS sector int
    ticker_ids:  Dict[str, int]   # ticker → security integer id


@dataclass
class LiveObservation:
    """
    Fully constructed observation ready for model inference.

    Shapes mirror the panel arrays used in training:
      x_panel   [1, L, K_max, F]   float32  — per-asset features (batch=1)
      g_panel   [1, D_g]           float32  — global features
      mask      [1, K_max]         float32  — 1=active, 0=inactive
      sector_ids [1, K_max]        int64
      ticker_ids [1, K_max]        int64
    """
    date:        datetime.date
    x_panel:     np.ndarray         # [1, L, K_max, F]
    g_panel:     np.ndarray         # [1, D_g]
    mask:        np.ndarray         # [1, K_max]
    sector_ids:  np.ndarray         # [1, K_max]
    ticker_ids:  np.ndarray         # [1, K_max]
    active_tickers: List[str]
    stale_tickers:  List[str]       # tickers with stale data this week
    is_valid:    bool = True
    warnings:    List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract data provider interface
# ---------------------------------------------------------------------------

class DataProvider(ABC):
    """
    Abstract base for live data providers.  Implement this to plug in a
    real market data source (Refinitiv, Bloomberg, Tiingo, etc.).
    """

    @abstractmethod
    def fetch_universe(self, as_of: datetime.date) -> LiveUniverse:
        """Return the NDX membership snapshot as-of the given date."""

    @abstractmethod
    def fetch_bars(
        self,
        tickers: List[str],
        end_date: datetime.date,
        lookback_weeks: int,
    ) -> Dict[str, List[LiveBar]]:
        """
        Return lookback_weeks of weekly bars for each ticker.
        Returns dict: ticker → list of LiveBar (oldest first).
        Missing weeks should be filled as None entries (caller handles).
        """

    @abstractmethod
    def fetch_global_features(
        self,
        date: datetime.date,
    ) -> Dict[str, float]:
        """
        Return global features for the given date:
          qqq_close, vix, weekly_idx, and any other D_g fields.
        """


# ---------------------------------------------------------------------------
# Synthetic data provider (for testing / paper-trading smoke tests)
# ---------------------------------------------------------------------------

class SyntheticDataProvider(DataProvider):
    """
    Deterministic synthetic data provider for testing.
    Generates plausible (but fake) bars and membership data.
    """

    def __init__(
        self,
        tickers:    List[str] = None,
        K_max:      int       = 16,
        F:          int       = 10,
        D_g:        int       = 8,
        seed:       int       = 42,
    ) -> None:
        if tickers is None:
            tickers = [f"TICKER{i:03d}" for i in range(8)]
        self._tickers  = tickers
        self._K_max    = K_max
        self._F        = F
        self._D_g      = D_g
        self._rng      = np.random.default_rng(seed)

    def fetch_universe(self, as_of: datetime.date) -> LiveUniverse:
        return LiveUniverse(
            as_of_date = as_of,
            tickers    = list(self._tickers),
            sector_ids = {t: i % 8 for i, t in enumerate(self._tickers)},
            ticker_ids = {t: i     for i, t in enumerate(self._tickers)},
        )

    def fetch_bars(
        self,
        tickers: List[str],
        end_date: datetime.date,
        lookback_weeks: int,
    ) -> Dict[str, List[LiveBar]]:
        result: Dict[str, List[LiveBar]] = {}
        for ticker in tickers:
            bars = []
            for w in range(lookback_weeks):
                d = end_date - datetime.timedelta(weeks=lookback_weeks - 1 - w)
                bars.append(LiveBar(
                    ticker      = ticker,
                    date        = d,
                    adj_close   = float(self._rng.uniform(50, 500)),
                    adj_open    = float(self._rng.uniform(50, 500)),
                    adv63       = float(self._rng.uniform(1e6, 1e9)),
                    vol_252     = float(self._rng.uniform(0.10, 0.50)),
                    gap_vol_252 = float(self._rng.uniform(0.01, 0.10)),
                ))
            result[ticker] = bars
        return result

    def fetch_global_features(self, date: datetime.date) -> Dict[str, float]:
        seed_val = int(date.strftime("%Y%m%d"))
        rng = np.random.default_rng(seed_val)
        return {
            "qqq_close":  float(rng.uniform(300, 500)),
            "vix":        float(rng.uniform(10, 40)),
            "weekly_idx": float(rng.integers(0, 1000)),
        }


# ---------------------------------------------------------------------------
# LiveDataAdapter
# ---------------------------------------------------------------------------

class LiveDataAdapter:
    """
    §12.4 Live data adapter.

    Fetches live data, verifies adjustment factors, and constructs the
    observation tensor for model inference.

    Parameters
    ----------
    provider       : DataProvider implementation
    K_max          : maximum universe size (must match model)
    F              : per-asset feature dimension (must match model)
    D_g            : global feature dimension (must match model)
    L_lookback     : temporal lookback window (default 60)
    norm_stats     : {"mean": [F], "std": [F]} IS-only normalization stats
    stale_threshold_weeks : weeks without price update = stale
    """

    def __init__(
        self,
        provider:               DataProvider,
        K_max:                  int,
        F:                      int,
        D_g:                    int,
        L_lookback:             int = 60,
        norm_stats:             Optional[Dict[str, np.ndarray]] = None,
        stale_threshold_weeks:  int = 2,
    ) -> None:
        self._provider     = provider
        self._K_max        = K_max
        self._F            = F
        self._D_g          = D_g
        self._L            = L_lookback
        self._norm_stats   = norm_stats
        self._stale_weeks  = stale_threshold_weeks

    # ======================================================================
    # Main entry point
    # ======================================================================

    def build_observation(
        self,
        rebalance_date: datetime.date,
    ) -> LiveObservation:
        """
        Construct a complete observation for a given rebalance date.

        Pipeline:
          1. Fetch NDX universe as-of rebalance_date
          2. Fetch L_lookback + 1 weeks of bars for each active ticker
          3. Fetch global features
          4. Construct x_panel [1, L, K_max, F], g_panel [1, D_g], mask
          5. Apply normalization if norm_stats provided
          6. Verify adjustment factors (check for anomalous price jumps)
          7. Return LiveObservation

        Parameters
        ----------
        rebalance_date : Monday (or first trading day of week) for signal gen
        """
        warnings: List[str] = []

        # Step 1: Universe
        universe = self._provider.fetch_universe(rebalance_date)
        tickers  = universe.tickers[:self._K_max]   # cap at K_max

        # Step 2: Bars
        bars_by_ticker = self._provider.fetch_bars(
            tickers       = tickers,
            end_date      = rebalance_date,
            lookback_weeks = self._L + 1,
        )

        # Step 3: Global features
        global_feats = self._provider.fetch_global_features(rebalance_date)

        # Step 4: Build arrays
        x_panel    = np.zeros((1, self._L, self._K_max, self._F), dtype=np.float32)
        mask       = np.zeros((1, self._K_max), dtype=np.float32)
        sector_ids = np.full((1, self._K_max), -1, dtype=np.int64)
        ticker_ids = np.full((1, self._K_max), -1, dtype=np.int64)
        stale_list: List[str] = []

        for k, ticker in enumerate(tickers):
            if k >= self._K_max:
                break
            bars = bars_by_ticker.get(ticker, [])

            # Staleness check
            if not bars or bars[-1].date < rebalance_date - datetime.timedelta(weeks=self._stale_weeks):
                stale_list.append(ticker)
                warnings.append(f"Stale data for {ticker}")
                continue

            if len(bars) < self._L:
                warnings.append(f"Insufficient history for {ticker}: {len(bars)} < {self._L}")
                continue

            # Fill x_panel: last L bars for ticker k
            for l, bar in enumerate(bars[-self._L:]):
                feat = self._bar_to_features(bar)
                x_panel[0, l, k, :len(feat)] = feat[:self._F]

            mask[0, k] = 1.0
            sector_ids[0, k] = universe.sector_ids.get(ticker, -1)
            ticker_ids[0, k] = universe.ticker_ids.get(ticker, k)

        # Step 5: Normalization
        if self._norm_stats is not None:
            mean = self._norm_stats["mean"]   # [F]
            std  = self._norm_stats["std"]    # [F]
            x_panel = (x_panel - mean) / np.maximum(std, 1e-8)

        # Step 6: Global panel
        g_panel = self._global_to_array(global_feats)

        # Verify adjustment factors
        self._verify_adjustment_factors(bars_by_ticker, warnings)

        return LiveObservation(
            date           = rebalance_date,
            x_panel        = x_panel,
            g_panel        = g_panel,
            mask           = mask,
            sector_ids     = sector_ids,
            ticker_ids     = ticker_ids,
            active_tickers = [t for k, t in enumerate(tickers)
                               if k < self._K_max and mask[0, k] > 0.5],
            stale_tickers  = stale_list,
            is_valid       = len(tickers) > 0 and float(mask.sum()) > 0,
            warnings       = warnings,
        )

    # ======================================================================
    # Helpers
    # ======================================================================

    def _bar_to_features(self, bar: LiveBar) -> np.ndarray:
        """
        Convert a LiveBar to a feature vector of length F.
        In production this mirrors the feature pipeline (§3).
        For the adapter stub, encodes the key numeric fields.
        """
        feats = np.zeros(self._F, dtype=np.float32)
        raw = [
            math.log(max(bar.adj_close, 1e-8)),
            math.log(max(bar.adj_open,  1e-8)),
            math.log(max(bar.adv63,     1e-8)),
            bar.vol_252,
            bar.gap_vol_252,
        ]
        for i, v in enumerate(raw[:self._F]):
            feats[i] = float(v)
        return feats

    def _global_to_array(self, global_feats: Dict[str, float]) -> np.ndarray:
        """Pack global features into a [1, D_g] array."""
        g = np.zeros((1, self._D_g), dtype=np.float32)
        keys = sorted(global_feats.keys())
        for i, key in enumerate(keys[:self._D_g]):
            g[0, i] = float(global_feats[key])
        return g

    def _verify_adjustment_factors(
        self,
        bars_by_ticker: Dict[str, List[LiveBar]],
        warnings:       List[str],
    ) -> None:
        """
        Verify adjustment factors are up-to-date (§12.4).
        Flags anomalous week-on-week price changes > 50% (potential bad adjust).
        """
        for ticker, bars in bars_by_ticker.items():
            if len(bars) < 2:
                continue
            for i in range(1, len(bars)):
                prev = bars[i-1].adj_close
                curr = bars[i].adj_close
                if prev > 0 and abs(curr / prev - 1.0) > 0.50:
                    warnings.append(
                        f"Possible bad adjustment factor for {ticker} "
                        f"on {bars[i].date}: "
                        f"{prev:.2f} → {curr:.2f} ({(curr/prev-1)*100:+.1f}%)"
                    )


