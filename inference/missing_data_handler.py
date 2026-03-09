"""
inference/missing_data_handler.py
===================================
Missing data handling at inference time (Bible §12.3).

Policy:
  - Short gap (≤ freeze_weeks): freeze previous position weight for the asset.
  - Prolonged gap (> freeze_weeks): flag asset for forced liquidation at next
    rebalance; set mask=0 (non-tradeable) for that step.
  - Feature window gap: if the full lookback window cannot be constructed,
    treat the asset as non-tradeable (mask=0) for that step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Asset missingness state
# ---------------------------------------------------------------------------

@dataclass
class AssetMissingnessState:
    """Tracks per-asset missing data state across steps."""
    ticker:              str
    consecutive_missing: int   = 0    # weeks of consecutive missing price
    flagged_liquidate:   bool  = False
    last_weight:         float = 0.0  # most recent valid weight (for freezing)


# ---------------------------------------------------------------------------
# MissingDataHandler
# ---------------------------------------------------------------------------

class MissingDataHandler:
    """
    §12.3 Missing data policy for inference.

    Parameters
    ----------
    freeze_weeks      : number of consecutive missing weeks before forced
                        liquidation is triggered (default 2)
    feature_window    : minimum number of valid historical bars required to
                        construct a feature vector (default 4)
    """

    def __init__(
        self,
        freeze_weeks:   int = 2,
        feature_window: int = 4,
    ) -> None:
        self.freeze_weeks   = int(freeze_weeks)
        self.feature_window = int(feature_window)

        # Per-ticker state (initialised on first call)
        self._state: Dict[str, AssetMissingnessState] = {}

    # ======================================================================
    # Main entry point
    # ======================================================================

    def process_step(
        self,
        tickers:         List[str],
        has_price:       Dict[str, bool],    # ticker → price available this week
        has_feature_window: Dict[str, bool], # ticker → full lookback available
        w_prev:          Dict[str, float],   # ticker → previous weight
    ) -> Tuple[np.ndarray, Dict[str, bool], List[str]]:
        """
        Apply §12.3 missing data policy.

        Parameters
        ----------
        tickers             : ordered list of tickers (len=K)
        has_price           : True if adjusted close is available this week
        has_feature_window  : True if full feature lookback window available
        w_prev              : previous executed weight per ticker

        Returns
        -------
        adjusted_mask : np.ndarray [K] float; 1=tradeable, 0=masked-out
        frozen_weights: Dict[str, float]  tickers whose weights are frozen
        liquidate_list: List[str]  tickers flagged for forced liquidation
        """
        K              = len(tickers)
        adjusted_mask  = np.ones(K, dtype=np.float32)
        frozen_weights : Dict[str, float] = {}
        liquidate_list : List[str] = []

        for i, ticker in enumerate(tickers):
            # Initialise state if first seen
            if ticker not in self._state:
                self._state[ticker] = AssetMissingnessState(
                    ticker       = ticker,
                    last_weight  = w_prev.get(ticker, 0.0),
                )

            state = self._state[ticker]
            state.last_weight = w_prev.get(ticker, state.last_weight)

            price_ok   = has_price.get(ticker, False)
            feature_ok = has_feature_window.get(ticker, False)

            if not price_ok:
                # Missing price: increment counter
                state.consecutive_missing += 1

                if state.consecutive_missing <= self.freeze_weeks:
                    # Short gap: freeze position
                    frozen_weights[ticker] = state.last_weight
                else:
                    # Prolonged gap: flag for liquidation, mask out
                    state.flagged_liquidate = True
                    adjusted_mask[i] = 0.0
                    liquidate_list.append(ticker)
            else:
                # Price available: reset counter
                state.consecutive_missing = 0
                state.flagged_liquidate   = False

            # Feature window check (independent of price check)
            if not feature_ok:
                adjusted_mask[i] = 0.0

        return adjusted_mask, frozen_weights, liquidate_list

    def apply_frozen_weights(
        self,
        w_exec:        np.ndarray,   # [K] current weights (may be modified)
        tickers:       List[str],
        frozen_weights: Dict[str, float],
    ) -> np.ndarray:
        """
        Overwrite frozen-asset weights in w_exec with their last valid weight,
        renormalising the remaining active weights to sum to 1.

        Parameters
        ----------
        w_exec         : [K] executed weights from model
        tickers        : ordered list of K tickers
        frozen_weights : ticker → frozen weight

        Returns
        -------
        w_adjusted : [K] with frozen slots overwritten and active slots renormed
        """
        if not frozen_weights:
            return w_exec.copy()

        w = w_exec.copy()
        ticker_idx = {t: i for i, t in enumerate(tickers)}

        frozen_total = 0.0
        frozen_idxs  = set()
        for ticker, fw in frozen_weights.items():
            if ticker in ticker_idx:
                idx          = ticker_idx[ticker]
                w[idx]       = float(fw)
                frozen_total += float(fw)
                frozen_idxs.add(idx)

        # Renormalise non-frozen active weights
        non_frozen_sum = sum(
            float(w[i]) for i in range(len(tickers))
            if i not in frozen_idxs
        )
        target_non_frozen = max(0.0, 1.0 - frozen_total)

        if non_frozen_sum > 1e-9 and target_non_frozen > 0:
            scale = target_non_frozen / non_frozen_sum
            for i in range(len(tickers)):
                if i not in frozen_idxs:
                    w[i] *= scale

        return w

    # ======================================================================
    # State inspection
    # ======================================================================

    def get_state(self, ticker: str) -> Optional[AssetMissingnessState]:
        return self._state.get(ticker)

    def reset(self, ticker: Optional[str] = None) -> None:
        """Reset state for a ticker, or all tickers if ticker is None."""
        if ticker is None:
            self._state.clear()
        elif ticker in self._state:
            del self._state[ticker]

    def flagged_for_liquidation(self) -> List[str]:
        """Return list of tickers currently flagged for forced liquidation."""
        return [t for t, s in self._state.items() if s.flagged_liquidate]

    def summary(self) -> Dict[str, Dict]:
        return {
            t: {
                "consecutive_missing": s.consecutive_missing,
                "flagged_liquidate":   s.flagged_liquidate,
                "last_weight":         s.last_weight,
            }
            for t, s in self._state.items()
        }
