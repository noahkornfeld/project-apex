"""
evaluation/fold_manager.py
==========================
Walk-forward fold design for Project Apex (Bible §9.1 / Table 31).

8 expanding-window folds:
  - 4-week embargo between train_end and test_start
  - Drops first L_lookback steps (insufficient history)
  - Drops last embargo_weeks training steps (n-step return leakage prevention)
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Fold specification table (§9.1 / Table 31)
# ---------------------------------------------------------------------------

@dataclass
class FoldSpec:
    fold_id:       int
    train_start:   str            # "YYYY-MM-DD"  inclusive
    train_end:     str            # "YYYY-MM-DD"  inclusive
    test_start:    str            # "YYYY-MM-DD"  inclusive
    test_end:      Optional[str]  # "YYYY-MM-DD"  inclusive; None = present
    embargo_weeks: int = 4


FOLD_SPECS: List[FoldSpec] = [
    FoldSpec(1, "2005-01-01", "2009-12-31", "2010-01-01", "2011-12-31"),
    FoldSpec(2, "2006-01-01", "2011-12-31", "2012-01-01", "2013-12-31"),
    FoldSpec(3, "2008-01-01", "2013-12-31", "2014-01-01", "2015-12-31"),
    FoldSpec(4, "2010-01-01", "2015-12-31", "2016-01-01", "2017-12-31"),
    FoldSpec(5, "2012-01-01", "2017-12-31", "2018-01-01", "2019-12-31"),
    FoldSpec(6, "2014-01-01", "2019-12-31", "2020-01-01", "2021-12-31"),
    FoldSpec(7, "2016-01-01", "2021-12-31", "2022-01-01", "2023-12-31"),
    FoldSpec(8, "2018-01-01", "2023-12-31", "2024-01-01", None),
]


# ---------------------------------------------------------------------------
# Date helper
# ---------------------------------------------------------------------------

def _to_date(d) -> datetime.date:
    """Convert str / datetime.date / np.datetime64 / pd.Timestamp → date."""
    if isinstance(d, datetime.date):
        return d
    if isinstance(d, str):
        return datetime.date.fromisoformat(d[:10])
    if hasattr(d, "astype"):          # np.datetime64
        ts = (d - np.datetime64("1970-01-01", "D")) / np.timedelta64(1, "D")
        return datetime.date.fromordinal(datetime.date(1970, 1, 1).toordinal() + int(ts))
    if hasattr(d, "date"):            # pd.Timestamp or datetime.datetime
        return d.date()
    return datetime.date.fromisoformat(str(d)[:10])


# ---------------------------------------------------------------------------
# FoldManager
# ---------------------------------------------------------------------------

class FoldManager:
    """
    8 expanding-window walk-forward folds (§9.1).

    Parameters
    ----------
    dates      : array-like of length T; element i is the calendar date for t_idx=i.
                 Accepts str "YYYY-MM-DD", datetime.date, np.datetime64.
    L_lookback : first L steps have insufficient lookback history → excluded from
                 valid training indices.
    """

    N_FOLDS = 8

    def __init__(
        self,
        dates:      np.ndarray,
        L_lookback: int = 60,
    ) -> None:
        self.dates      = np.array([_to_date(d) for d in dates], dtype=object)
        self.L_lookback = int(L_lookback)
        self._specs: Dict[int, FoldSpec] = {s.fold_id: s for s in FOLD_SPECS}

    # ======================================================================
    # Public API
    # ======================================================================

    def get_fold_spec(self, fold_id: int) -> FoldSpec:
        if fold_id not in self._specs:
            raise ValueError(f"fold_id must be 1–8, got {fold_id}")
        return self._specs[fold_id]

    def get_train_indices(
        self,
        fold_id:                int,
        include_embargo:        bool = False,
        include_lookback_warmup: bool = False,
    ) -> np.ndarray:
        """
        Return valid training t_idx array.

        Excludes by default:
          - The last embargo_weeks weeks (prevent n-step leakage, §9.1)
          - The first L_lookback steps (insufficient history)

        Parameters
        ----------
        include_embargo        : if True, keep embargo-window steps
        include_lookback_warmup : if True, keep first L_lookback steps
        """
        spec        = self.get_fold_spec(fold_id)
        train_start = _to_date(spec.train_start)
        train_end   = _to_date(spec.train_end)

        embargo_cutoff = train_end - datetime.timedelta(weeks=spec.embargo_weeks)

        mask = (self.dates >= train_start) & (self.dates <= train_end)
        if not include_embargo:
            mask &= (self.dates <= embargo_cutoff)

        indices = np.where(mask)[0]

        if not include_lookback_warmup:
            indices = indices[indices >= self.L_lookback]

        return indices

    def get_embargo_indices(self, fold_id: int) -> np.ndarray:
        """Return t_idx values within the 4-week embargo window."""
        spec           = self.get_fold_spec(fold_id)
        train_end      = _to_date(spec.train_end)
        embargo_cutoff = train_end - datetime.timedelta(weeks=spec.embargo_weeks)

        mask = (self.dates > embargo_cutoff) & (self.dates <= train_end)
        return np.where(mask)[0]

    def get_test_indices(self, fold_id: int) -> np.ndarray:
        """Return t_idx values within the OOS test window."""
        spec       = self.get_fold_spec(fold_id)
        test_start = _to_date(spec.test_start)

        if spec.test_end is None:
            mask = self.dates >= test_start
        else:
            test_end = _to_date(spec.test_end)
            mask = (self.dates >= test_start) & (self.dates <= test_end)

        return np.where(mask)[0]

    def validate_embargo(
        self,
        fold_id:         int,
        train_t_indices: np.ndarray,
    ) -> Tuple[bool, str]:
        """
        Embargo assertion (Gate 11): no training transition has t_idx within
        the embargo window.

        Returns
        -------
        (passed: bool, message: str)
        """
        embargo_set = set(self.get_embargo_indices(fold_id).tolist())
        violations  = [int(i) for i in train_t_indices if int(i) in embargo_set]

        if violations:
            return False, (
                f"Fold {fold_id}: EMBARGO VIOLATION — "
                f"{len(violations)} training transitions in embargo window. "
                f"First 5: {violations[:5]}"
            )
        return True, f"Fold {fold_id}: embargo check PASSED (0 violations)"

    def n_steps_dropped_embargo(self, fold_id: int) -> int:
        """Steps dropped due to embargo = embargo_weeks."""
        return self.get_fold_spec(fold_id).embargo_weeks

    def n_steps_dropped_lookback(self) -> int:
        """Steps dropped due to insufficient lookback = L_lookback."""
        return self.L_lookback

    def fold_metadata(self, fold_id: int) -> Dict:
        """Full metadata dict for §10.3 per-fold summary."""
        spec        = self.get_fold_spec(fold_id)
        train_idx   = self.get_train_indices(fold_id)
        test_idx    = self.get_test_indices(fold_id)
        embargo_idx = self.get_embargo_indices(fold_id)

        train_start = _to_date(spec.train_start)
        train_end   = _to_date(spec.train_end)
        n_raw = int(np.sum(
            (self.dates >= train_start) & (self.dates <= train_end)
        ))

        return {
            "fold_id":                            fold_id,
            "train_start":                        spec.train_start,
            "train_end":                          spec.train_end,
            "test_start":                         spec.test_start,
            "test_end":                           spec.test_end,
            "embargo_weeks":                      spec.embargo_weeks,
            "n_train_steps_raw":                  n_raw,
            "n_train_steps_used":                 len(train_idx),
            "n_test_steps":                       len(test_idx),
            "n_steps_dropped_embargo":            len(embargo_idx),
            "n_steps_dropped_insufficient_lookback": self.L_lookback,
            "train_t_indices":                    train_idx.tolist(),
            "test_t_indices":                     test_idx.tolist(),
            "embargo_t_indices":                  embargo_idx.tolist(),
        }

    def all_fold_metadata(self) -> List[Dict]:
        return [self.fold_metadata(i) for i in range(1, self.N_FOLDS + 1)]
