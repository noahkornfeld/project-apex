"""
evaluation/checkpoint_selector.py
==================================
Checkpoint selection policy for Project Apex (Bible §9.6).

Selection criteria:
  Primary   : best OOS Sortino on most recent walk-forward fold
  Secondary : among similar Sortino, prefer lower max drawdown (tiebreak)
  Stability : selected checkpoint must not show Q-value divergence or entropy
              collapse in its training logs
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Checkpoint record
# ---------------------------------------------------------------------------

@dataclass
class CheckpointRecord:
    """Metadata for a single saved checkpoint."""
    checkpoint_id:    str            # unique ID (e.g., "fold1_step5000")
    fold_id:          int
    update_step:      int
    oos_sortino:      float          # OOS Sortino on the most recent fold
    oos_max_drawdown: float          # OOS max drawdown (negative value)
    oos_excess_cagr:  float = 0.0
    q_divergence:     bool  = False  # True if training logs show Q-value divergence
    entropy_collapse: bool  = False  # True if training logs show entropy collapse
    extra:            Dict  = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CheckpointSelector
# ---------------------------------------------------------------------------

class CheckpointSelector:
    """
    §9.6 Checkpoint selection policy.

    Usage
    -----
    selector = CheckpointSelector()
    selector.add(CheckpointRecord(...))
    ...
    best = selector.select(most_recent_fold_id=8)
    """

    def __init__(
        self,
        sortino_tol: float = 0.05,    # Sortino difference below which tiebreak applies
    ) -> None:
        self._records:   List[CheckpointRecord] = []
        self._sortino_tol = float(sortino_tol)

    def add(self, record: CheckpointRecord) -> None:
        """Register a checkpoint."""
        self._records.append(record)

    def add_many(self, records: List[CheckpointRecord]) -> None:
        for r in records:
            self.add(r)

    def select(
        self,
        most_recent_fold_id: Optional[int] = None,
    ) -> Tuple[Optional[CheckpointRecord], str]:
        """
        Select the best checkpoint per §9.6.

        Parameters
        ----------
        most_recent_fold_id : if provided, only consider checkpoints from this
                              fold; otherwise uses all registered checkpoints.

        Returns
        -------
        (best: CheckpointRecord | None, reason: str)
        """
        candidates = self._records

        if not candidates:
            return None, "No checkpoints registered."

        # Filter to most recent fold if specified
        if most_recent_fold_id is not None:
            fold_candidates = [r for r in candidates if r.fold_id == most_recent_fold_id]
            if fold_candidates:
                candidates = fold_candidates
            # else fall back to all candidates

        # Stability filter: exclude Q-value divergence or entropy collapse
        stable = [r for r in candidates if not r.q_divergence and not r.entropy_collapse]
        if not stable:
            # If all candidates are unstable, warn but still pick the "least unstable"
            stable = candidates
            stability_note = " [WARNING: no fully stable checkpoints; selected least unstable]"
        else:
            stability_note = ""

        # Primary criterion: best OOS Sortino
        best_sortino = max(r.oos_sortino for r in stable if math.isfinite(r.oos_sortino))

        # Candidates within tiebreak tolerance of best Sortino
        top = [
            r for r in stable
            if math.isfinite(r.oos_sortino)
            and r.oos_sortino >= best_sortino - self._sortino_tol
        ]

        if not top:
            # All Sortino values are NaN
            top = stable

        # Secondary criterion: lower max drawdown (less negative = better)
        top.sort(key=lambda r: (
            -r.oos_sortino if math.isfinite(r.oos_sortino) else float("-inf"),
            r.oos_max_drawdown * (-1),   # less negative = preferred (sort ascending by |dd|)
        ))

        best = top[0]
        reason = (
            f"Selected {best.checkpoint_id}: "
            f"fold={best.fold_id}, step={best.update_step}, "
            f"Sortino={best.oos_sortino:.4f}, "
            f"MaxDD={best.oos_max_drawdown:.4f}"
            f"{stability_note}"
        )
        return best, reason

    def best_by_fold(self, fold_id: int) -> Optional[CheckpointRecord]:
        """Return the best checkpoint for a specific fold (highest Sortino, stable)."""
        fold_records = [
            r for r in self._records
            if r.fold_id == fold_id
            and not r.q_divergence
            and not r.entropy_collapse
            and math.isfinite(r.oos_sortino)
        ]
        if not fold_records:
            return None
        return max(fold_records, key=lambda r: (r.oos_sortino, -abs(r.oos_max_drawdown)))

    def all_records(self) -> List[CheckpointRecord]:
        return list(self._records)

    def summary(self) -> List[Dict]:
        return [
            {
                "checkpoint_id":    r.checkpoint_id,
                "fold_id":          r.fold_id,
                "update_step":      r.update_step,
                "oos_sortino":      r.oos_sortino,
                "oos_max_drawdown": r.oos_max_drawdown,
                "oos_excess_cagr":  r.oos_excess_cagr,
                "q_divergence":     r.q_divergence,
                "entropy_collapse": r.entropy_collapse,
            }
            for r in self._records
        ]
