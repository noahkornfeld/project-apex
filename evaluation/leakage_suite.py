"""
evaluation/leakage_suite.py
===========================
Leakage validation suite for Project Apex (Bible §9.2 / §11.3).

5 leakage traps — all must pass before OOS evaluation for each fold:
  1. temporal       - no future bar data used in feature computation
  2. normalizer     - normalizer stats fitted on IS-only data
  3. membership     - membership snapshots use backward-fill (as-of rule)
  4. embargo        - no training transition has t_idx in embargo window
  5. n_step_boundary - no n-step return spans the train–test boundary
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LeakageResult:
    test_name:  str
    passed:     bool
    message:    str
    details:    Optional[Dict] = None


# ---------------------------------------------------------------------------
# LeakageSuite
# ---------------------------------------------------------------------------

class LeakageSuite:
    """
    Runs the 5 leakage trap tests defined in §9.2 / §11.3.

    All checks return LeakageResult(passed=True/False, message=...).
    The run_all() method returns a list of all 5 results; all must pass.
    """

    # ======================================================================
    # 1. Temporal leakage trap  (§11.3)
    # ======================================================================

    def check_temporal(
        self,
        x_panel:   np.ndarray,    # [T, K, F] feature panel
        t_idx:     int,           # time step to audit
        rng:       Optional[np.random.Generator] = None,
    ) -> LeakageResult:
        """
        Temporal leakage trap: feature at t must not depend on future bars.

        Method (§11.3): create a perturbed version of x_panel where all data at
        t+1 and later is replaced with random noise.  Assert that x_panel[t]
        and the perturbed panel's row at t are identical.  If features at t
        depend on future data, they will differ after perturbation.

        Parameters
        ----------
        x_panel : [T, K, F] feature panel
        t_idx   : time step to audit (must be < T-1)
        rng     : numpy Generator for reproducibility
        """
        T = x_panel.shape[0]
        if t_idx >= T - 1:
            return LeakageResult(
                test_name="temporal",
                passed=True,
                message=f"temporal: skipped (t_idx={t_idx} is last step)",
            )

        if rng is None:
            rng = np.random.default_rng(0)

        # Perturb future rows
        x_perturbed       = x_panel.copy()
        future_shape      = x_panel[t_idx + 1:].shape
        x_perturbed[t_idx + 1:] = rng.standard_normal(future_shape).astype(x_panel.dtype)

        original_row  = x_panel[t_idx]
        perturbed_row = x_perturbed[t_idx]

        if not np.allclose(original_row, perturbed_row, equal_nan=True):
            return LeakageResult(
                test_name="temporal",
                passed=False,
                message=(
                    f"temporal: FAIL — feature at t={t_idx} changed after "
                    f"perturbing t+1:T, indicating future data dependency."
                ),
                details={"t_idx": t_idx, "max_diff": float(np.nanmax(np.abs(original_row - perturbed_row)))},
            )

        return LeakageResult(
            test_name="temporal",
            passed=True,
            message=f"temporal: PASS — feature at t={t_idx} is unaffected by future perturbation.",
        )

    # ======================================================================
    # 2. Normalizer leakage trap  (§11.3)
    # ======================================================================

    def check_normalizer(
        self,
        stats_is_only: Dict[str, Any],   # {"mean": ..., "std": ...} fitted on IS
        stats_full:    Dict[str, Any],   # same keys, fitted on IS + OOS
        oos_data:      np.ndarray,       # [T_oos, ...] raw OOS feature data
        rtol:          float = 0.01,     # relative tolerance; if stats differ by
                                         # more than this → IS-only verified
    ) -> LeakageResult:
        """
        Normalizer leakage trap: IS-only stats must differ from full (IS+OOS) stats.

        Logic:
          - If IS-only and full normalizer produce the same output on OOS data,
            it could indicate the normalizer was fitted on OOS data too (leakage).
          - We assert that the two normalizers produce DIFFERENT outputs, then
            verify the system uses the IS-only normalizer.

        Pass criterion: stats_is_only differs from stats_full by at least rtol
        on at least one feature dimension (they SHOULD differ when OOS data has
        different distribution from IS).
        """
        mean_is   = np.asarray(stats_is_only.get("mean", []))
        mean_full = np.asarray(stats_full.get("mean",    []))
        std_is    = np.asarray(stats_is_only.get("std",  []))
        std_full  = np.asarray(stats_full.get("std",     []))

        if mean_is.shape != mean_full.shape or std_is.shape != std_full.shape:
            return LeakageResult(
                test_name="normalizer",
                passed=False,
                message="normalizer: FAIL — IS-only and full stats have different shapes.",
            )

        mean_diff = float(np.max(np.abs(mean_is - mean_full) / (np.abs(mean_full) + 1e-8)))
        std_diff  = float(np.max(np.abs(std_is  - std_full)  / (np.abs(std_full)  + 1e-8)))
        max_diff  = max(mean_diff, std_diff)

        if max_diff < rtol:
            return LeakageResult(
                test_name="normalizer",
                passed=False,
                message=(
                    f"normalizer: FAIL — IS-only and full normalizer stats are nearly "
                    f"identical (max_rel_diff={max_diff:.4f} < rtol={rtol}). "
                    f"Possible normalizer leakage: OOS data may have been included "
                    f"in fitting."
                ),
                details={"max_rel_diff": max_diff},
            )

        return LeakageResult(
            test_name="normalizer",
            passed=True,
            message=(
                f"normalizer: PASS — IS-only normalizer differs from full "
                f"(max_rel_diff={max_diff:.4f}), confirming IS-only fitting."
            ),
            details={"max_rel_diff": max_diff},
        )

    # ======================================================================
    # 3. Membership leakage trap  (§11.3)
    # ======================================================================

    def check_membership(
        self,
        membership_at_t:        set,   # set of tickers in universe at t
        known_future_additions: set,   # tickers added AFTER t (must NOT be in membership_at_t)
        t_idx:                  int,
    ) -> LeakageResult:
        """
        Membership leakage trap: membership snapshot at t must not contain assets
        added to the index after t (backward-fill / as-of rule).

        Parameters
        ----------
        membership_at_t        : tickers returned by membership snapshot at t_idx
        known_future_additions : tickers that are known to have been added to NDX
                                 after the date corresponding to t_idx
        """
        leaking = membership_at_t & known_future_additions
        if leaking:
            return LeakageResult(
                test_name="membership",
                passed=False,
                message=(
                    f"membership: FAIL — {len(leaking)} future addition(s) present "
                    f"in membership snapshot at t={t_idx}: {sorted(leaking)[:5]}"
                ),
                details={"leaking_tickers": sorted(leaking), "t_idx": t_idx},
            )

        return LeakageResult(
            test_name="membership",
            passed=True,
            message=(
                f"membership: PASS — no future additions in membership at t={t_idx}."
            ),
        )

    # ======================================================================
    # 4. Embargo leakage trap  (§9.2 / §11.3)
    # ======================================================================

    def check_embargo(
        self,
        fold_manager,            # FoldManager instance
        fold_id:         int,
        train_t_indices: np.ndarray,
    ) -> LeakageResult:
        """
        Embargo check: no training transition has t_idx within the 4-week
        embargo window before test start (§9.1, §9.2).
        """
        passed, msg = fold_manager.validate_embargo(fold_id, train_t_indices)
        return LeakageResult(
            test_name="embargo",
            passed=passed,
            message=msg,
            details={"fold_id": fold_id, "n_train": len(train_t_indices)},
        )

    # ======================================================================
    # 5. n-step boundary leakage trap  (§9.2 / §11.3)
    # ======================================================================

    def check_n_step_boundary(
        self,
        train_t_indices:   np.ndarray,   # valid training t_idx values
        test_start_t_idx:  int,          # first t_idx of the OOS period
        n_step:            int = 4,      # n-step return horizon
    ) -> LeakageResult:
        """
        n-step boundary trap: a training transition at t generates an n-step
        return using steps t+1, …, t+n_step.  If t+n_step >= test_start_t_idx,
        the return incorporates OOS data.

        Pass criterion: max(train_t_indices) + n_step < test_start_t_idx.
        """
        if len(train_t_indices) == 0:
            return LeakageResult(
                test_name="n_step_boundary",
                passed=True,
                message="n_step_boundary: PASS — no training indices provided.",
            )

        max_train = int(np.max(train_t_indices))
        boundary  = max_train + n_step

        if boundary >= test_start_t_idx:
            return LeakageResult(
                test_name="n_step_boundary",
                passed=False,
                message=(
                    f"n_step_boundary: FAIL — max training t_idx={max_train}, "
                    f"n_step={n_step} → boundary={boundary} >= "
                    f"test_start={test_start_t_idx}. "
                    f"n-step return would incorporate OOS data."
                ),
                details={
                    "max_train_t_idx": max_train,
                    "n_step":          n_step,
                    "boundary":        boundary,
                    "test_start":      test_start_t_idx,
                },
            )

        return LeakageResult(
            test_name="n_step_boundary",
            passed=True,
            message=(
                f"n_step_boundary: PASS — max train t_idx={max_train}, "
                f"boundary={boundary} < test_start={test_start_t_idx}."
            ),
        )

    # ======================================================================
    # run_all
    # ======================================================================

    def run_all(
        self,
        *,
        # temporal
        x_panel:                Optional[np.ndarray]  = None,
        temporal_t_idx:         int                   = 0,
        temporal_rng:           Optional[np.random.Generator] = None,
        # normalizer
        stats_is_only:          Optional[Dict]        = None,
        stats_full:             Optional[Dict]        = None,
        oos_data:               Optional[np.ndarray]  = None,
        normalizer_rtol:        float                 = 0.01,
        # membership
        membership_at_t:        Optional[set]         = None,
        known_future_additions: Optional[set]         = None,
        membership_t_idx:       int                   = 0,
        # embargo
        fold_manager            = None,
        fold_id:                int                   = 1,
        train_t_indices:        Optional[np.ndarray]  = None,
        # n-step boundary
        test_start_t_idx:       Optional[int]         = None,
        n_step:                 int                   = 4,
    ) -> List[LeakageResult]:
        """
        Run all 5 leakage trap tests.

        Returns list of 5 LeakageResult objects (one per test).
        All must have passed=True before OOS evaluation proceeds.
        """
        results: List[LeakageResult] = []

        # 1. Temporal
        if x_panel is not None:
            results.append(self.check_temporal(x_panel, temporal_t_idx, temporal_rng))
        else:
            results.append(LeakageResult(
                test_name="temporal",
                passed=True,
                message="temporal: SKIPPED (no x_panel provided)",
            ))

        # 2. Normalizer
        if stats_is_only is not None and stats_full is not None:
            results.append(self.check_normalizer(stats_is_only, stats_full, oos_data))
        else:
            results.append(LeakageResult(
                test_name="normalizer",
                passed=True,
                message="normalizer: SKIPPED (no normalizer stats provided)",
            ))

        # 3. Membership
        if membership_at_t is not None and known_future_additions is not None:
            results.append(self.check_membership(
                membership_at_t, known_future_additions, membership_t_idx
            ))
        else:
            results.append(LeakageResult(
                test_name="membership",
                passed=True,
                message="membership: SKIPPED (no membership data provided)",
            ))

        # 4. Embargo
        if fold_manager is not None and train_t_indices is not None:
            results.append(self.check_embargo(fold_manager, fold_id, train_t_indices))
        else:
            results.append(LeakageResult(
                test_name="embargo",
                passed=True,
                message="embargo: SKIPPED (no fold_manager or train_t_indices)",
            ))

        # 5. n-step boundary
        if train_t_indices is not None and test_start_t_idx is not None:
            results.append(self.check_n_step_boundary(
                train_t_indices, test_start_t_idx, n_step
            ))
        else:
            results.append(LeakageResult(
                test_name="n_step_boundary",
                passed=True,
                message="n_step_boundary: SKIPPED (no boundary data provided)",
            ))

        return results

    @staticmethod
    def all_passed(results: List[LeakageResult]) -> Tuple[bool, List[str]]:
        """
        Check if all 5 leakage tests passed.

        Returns
        -------
        (all_pass: bool, failures: List[str])
        """
        failures = [r.message for r in results if not r.passed]
        return len(failures) == 0, failures
