"""
training/replay_buffer.py
Replay Buffer — Phase 8 (Bible §8.2–8.6).

Design:
  Fixed-size circular buffer (capacity=800) padded to K_max.
  Observations (x_t, g_t) are NOT stored — reconstructed from t_idx at sample time.

Per-transition storage:
  t_idx      (int64)       : panel row index for observation reconstruction
  t_idx_next (int64)       : panel row index of s_{t+n} for bootstrap
  mask_t     (float32, K)  : active-slot mask at time t
  w_pre      (float32, K)  : pre-projection action
  w_exec     (float32, K)  : executed (post-projection) weights
  R_n        (float32)     : n-step discounted return  Σ γ^k r_{t+k}
  gamma_n    (float32)     : γ^{n_step} — fixed discount for bootstrap (§8.2)
  done_n     (bool)        : True if any done_{t+k} in the n-step window
  warmup_flag (bool)       : True for warmup-generated transitions

Key rules:
  §8.2  n-step returns must NOT cross episode boundaries; last n_step−1 warmup
        transitions (steps 49, 50, 51 of 52-step warmup) are DROPPED entirely.
  §8.3.3 Recency weighting: w_i = exp(−ln2/half_life_weeks × age_weeks).
         Age is measured by t_idx (calendar date), not insertion order.
  §8.6  Warmup exclusion: once ≥128 policy transitions exist, warmup slots
        receive zero sampling weight.
  §8.4  Transition augmentation (critic path only):
          R_aug = R_n + N(0, (0.01 × buffer_reward_std)²)
          Feature noise is signalled via aug_obs_std_factor in the batch dict;
          actual obs noise is applied by the caller during obs reconstruction.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Dict, List, Optional

EPS = 1e-8


class ReplayBuffer:
    """
    Fixed-size circular replay buffer with n-step returns, recency
    weighting, warmup exclusion, and transition augmentation.

    Parameters
    ----------
    capacity                  : buffer size (default 800)
    K_max                     : max asset slots — per-slot tensors padded to this
    n_step                    : n-step return horizon  (default 4)
    gamma                     : per-step discount factor  (default 0.975)
    half_life_weeks           : recency half-life in weeks  (default 156 = 3 yr)
    warmup_steps              : warmup episode length  (default 52)
    warmup_exclusion_threshold: min policy transitions before warmup excluded
    aug_reward_noise_factor   : σ_r = factor × buffer_reward_std  (§8.4)
    aug_obs_noise_factor      : σ_x = factor × per_feature_std    (§8.4)
    trading_days_per_week     : converts t_idx (days) to weeks for recency
    """

    def __init__(
        self,
        capacity:                    int   = 800,
        K_max:                       int   = 110,
        n_step:                      int   = 4,
        gamma:                       float = 0.975,
        half_life_weeks:             float = 156.0,
        warmup_steps:                int   = 52,
        warmup_exclusion_threshold:  int   = 128,
        aug_reward_noise_factor:     float = 0.01,
        aug_obs_noise_factor:        float = 0.015,
        trading_days_per_week:       int   = 5,
    ):
        self._cap        = int(capacity)
        self._K          = int(K_max)
        self._n          = int(n_step)
        self._gamma      = float(gamma)
        self._gamma_n    = float(gamma ** n_step)    # γ^{n_step} — always stored
        self._half_life  = float(half_life_weeks)
        self._wu_steps   = int(warmup_steps)
        self._wu_excl    = int(warmup_exclusion_threshold)
        self._aug_r      = float(aug_reward_noise_factor)
        self._aug_x      = float(aug_obs_noise_factor)
        self._days_pw    = int(trading_days_per_week)

        # ------------------------------------------------------------------
        # Circular storage (pre-allocated)
        # ------------------------------------------------------------------
        self._t_idx       = np.zeros(self._cap, dtype=np.int64)
        self._t_idx_next  = np.zeros(self._cap, dtype=np.int64)
        self._mask_t      = np.zeros((self._cap, self._K), dtype=np.float32)
        self._w_pre       = np.zeros((self._cap, self._K), dtype=np.float32)
        self._w_exec      = np.zeros((self._cap, self._K), dtype=np.float32)
        self._R_n         = np.zeros(self._cap, dtype=np.float32)
        self._gn          = np.full(self._cap, self._gamma_n, dtype=np.float32)
        self._done_n      = np.zeros(self._cap, dtype=bool)
        self._warmup_flag = np.zeros(self._cap, dtype=bool)

        self._head  = 0    # next write slot (circular)
        self._size  = 0    # current fill count  (≤ capacity)

        # Per-feature std for obs augmentation — updated externally (§8.4)
        self.per_feature_std: Optional[np.ndarray] = None

        # ------------------------------------------------------------------
        # Staging deque for step-by-step insertion (add_step API)
        # ------------------------------------------------------------------
        self._stage: deque = deque()

    # ======================================================================
    # Properties
    # ======================================================================

    @property
    def size(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._cap

    @property
    def is_full(self) -> bool:
        return self._size >= self._cap

    @property
    def policy_count(self) -> int:
        """Number of non-warmup (policy-generated) transitions in buffer."""
        if self._size == 0:
            return 0
        return int((~self._warmup_flag[:self._size]).sum())

    @property
    def buffer_reward_std(self) -> float:
        """Running std of all R_n values currently in buffer (§8.3.2)."""
        if self._size < 2:
            return 1.0
        return float(np.std(self._R_n[:self._size], ddof=0))

    # ======================================================================
    # Insertion — batch (episode-end) API
    # ======================================================================

    def add_episode(
        self,
        transitions: List[Dict],
        is_warmup:   bool = False,
    ) -> int:
        """
        Compute n-step returns for every eligible step and insert into buffer.

        Parameters
        ----------
        transitions : list of dicts, one per env step, with keys:
            t_idx      (int)
            t_idx_next (int)
            mask_t     (array-like [K])
            w_pre      (array-like [K])
            w_exec     (array-like [K])
            reward     (float)
            done       (bool)
        is_warmup : True for warmup-phase episodes.

        Returns
        -------
        Number of transitions actually stored (after warmup-drop filtering).
        """
        T = len(transitions)
        stored = 0

        for t in range(T):
            # §8.6 / §8.2: drop last n_step−1 warmup steps
            if is_warmup and t >= self._wu_steps - (self._n - 1):
                continue

            # ------------------------------------------------------------------
            # Compute n-step return (§8.2)
            # R_n = Σ_{k=0}^{n−1} γ^k r_{t+k}  (truncated at episode end)
            # done_n = True if any done in the window
            # t_idx_next_n = t_idx_next of the last included step
            # ------------------------------------------------------------------
            R_n    = 0.0
            g_acc  = 1.0
            done_n = False
            last   = t   # index of final step included in this n-step window

            for k in range(self._n):
                idx = t + k
                if idx >= T:
                    break
                step    = transitions[idx]
                R_n    += g_acc * float(step["reward"])
                g_acc  *= self._gamma
                last    = idx
                if step["done"]:
                    done_n = True
                    break

            t_idx_next_n = int(transitions[last]["t_idx_next"])

            self._write(
                t_idx       = int(transitions[t]["t_idx"]),
                t_idx_next  = t_idx_next_n,
                mask_t      = np.asarray(transitions[t]["mask_t"],  dtype=np.float32),
                w_pre       = np.asarray(transitions[t]["w_pre"],   dtype=np.float32),
                w_exec      = np.asarray(transitions[t]["w_exec"],  dtype=np.float32),
                R_n         = float(R_n),
                done_n      = done_n,
                warmup_flag = is_warmup,
            )
            stored += 1

        return stored

    # ======================================================================
    # Insertion — step-by-step API (staging deque)
    # ======================================================================

    def add_step(
        self,
        t_idx:      int,
        mask_t:     np.ndarray,
        w_pre:      np.ndarray,
        w_exec:     np.ndarray,
        reward:     float,
        t_idx_next: int,
        done:       bool,
        is_warmup:  bool = False,
        step_idx:   int  = 0,
    ) -> None:
        """
        Stage one step; flush the oldest staged transition once n_step
        rewards are available.  Call flush_episode() at episode end to
        commit remaining staged steps.

        step_idx : 0-based within-episode index (used for warmup-drop rule).
        """
        self._stage.append({
            "t_idx":      t_idx,
            "t_idx_next": t_idx_next,
            "mask_t":     np.asarray(mask_t,  dtype=np.float32),
            "w_pre":      np.asarray(w_pre,   dtype=np.float32),
            "w_exec":     np.asarray(w_exec,  dtype=np.float32),
            "reward":     float(reward),
            "done":       bool(done),
            "step_idx":   int(step_idx),
            "is_warmup":  bool(is_warmup),
        })

        # Flush once we can compute a full n-step return, or on episode end
        if len(self._stage) >= self._n or done:
            self._flush_stage_oldest(force=done)

    def flush_episode(self) -> None:
        """Flush all remaining staged transitions at episode end."""
        while self._stage:
            self._flush_stage_oldest(force=True)

    # ======================================================================
    # Sampling
    # ======================================================================

    def sample(
        self,
        batch_size: int,
        critic:     bool = False,
        rng:        Optional[np.random.Generator] = None,
    ) -> Dict:
        """
        Sample batch_size transitions with exponential recency weighting (§8.3.3).

        critic=True  → applies reward augmentation (§8.4); includes
                        aug_obs_std_factor hint for observation augmentation.
        critic=False → returns clean batch (actor path).

        Returns a dict with keys:
          idx, t_idx, t_idx_next, mask_t, w_pre, w_exec,
          R_n, R_n_clean, gamma_n, done_n, warmup_flag,
          aug_obs_std_factor
        """
        assert self._size > 0, "Cannot sample from empty buffer"
        if rng is None:
            rng = np.random.default_rng()

        weights = self._recency_weights()   # [size]

        # §8.6 warmup exclusion
        if self.policy_count >= self._wu_excl:
            weights[self._warmup_flag[:self._size]] = 0.0

        total = weights.sum()
        assert total > 0, "All sampling weights are zero (check warmup_exclusion_threshold)"
        weights /= total

        idx = rng.choice(self._size, size=batch_size, replace=True, p=weights)

        R_n_clean = self._R_n[idx].copy()
        R_n       = R_n_clean.copy()

        # §8.4 Reward augmentation (critic only)
        if critic:
            r_std = max(self.buffer_reward_std, EPS)
            noise = rng.normal(0.0, self._aug_r * r_std, size=R_n.shape)
            R_n   = (R_n + noise).astype(np.float32)

        return {
            "idx":              idx,
            "t_idx":            self._t_idx[idx].copy(),
            "t_idx_next":       self._t_idx_next[idx].copy(),
            "mask_t":           self._mask_t[idx].copy(),
            "w_pre":            self._w_pre[idx].copy(),
            "w_exec":           self._w_exec[idx].copy(),
            "R_n":              R_n,
            "R_n_clean":        R_n_clean,
            "gamma_n":          self._gn[idx].copy(),
            "done_n":           self._done_n[idx].copy(),
            "warmup_flag":      self._warmup_flag[idx].copy(),
            "aug_obs_std_factor": float(self._aug_x) if critic else 0.0,
        }

    # ======================================================================
    # Utilities
    # ======================================================================

    def clear(self) -> None:
        """Reset the buffer to empty state."""
        self._head = 0
        self._size = 0
        self._stage.clear()

    def update_per_feature_std(self, std: np.ndarray) -> None:
        """Set the per-feature std used for observation augmentation (§8.4)."""
        self.per_feature_std = np.asarray(std, dtype=np.float32)

    def stored_rewards(self) -> np.ndarray:
        """Return a copy of all stored R_n values (for testing / diagnostics)."""
        return self._R_n[:self._size].copy()

    def stored_t_idx(self) -> np.ndarray:
        """Return a copy of all stored t_idx values."""
        return self._t_idx[:self._size].copy()

    def stored_warmup_flags(self) -> np.ndarray:
        """Return a copy of all stored warmup_flag values."""
        return self._warmup_flag[:self._size].copy()

    # ======================================================================
    # Internal helpers
    # ======================================================================

    def _write(
        self,
        t_idx:       int,
        t_idx_next:  int,
        mask_t:      np.ndarray,
        w_pre:       np.ndarray,
        w_exec:      np.ndarray,
        R_n:         float,
        done_n:      bool,
        warmup_flag: bool,
    ) -> None:
        """Write one transition to the circular buffer at _head (wraps on full)."""
        h = self._head
        K = self._K

        self._t_idx[h]       = t_idx
        self._t_idx_next[h]  = t_idx_next
        self._mask_t[h]      = mask_t[:K]  if len(mask_t) >= K else np.pad(mask_t, (0, K - len(mask_t)))
        self._w_pre[h]       = w_pre[:K]   if len(w_pre)  >= K else np.pad(w_pre,  (0, K - len(w_pre)))
        self._w_exec[h]      = w_exec[:K]  if len(w_exec) >= K else np.pad(w_exec, (0, K - len(w_exec)))
        self._R_n[h]         = float(R_n)
        self._gn[h]          = self._gamma_n   # γ^{n_step} fixed (§8.2)
        self._done_n[h]      = bool(done_n)
        self._warmup_flag[h] = bool(warmup_flag)

        self._head = (self._head + 1) % self._cap
        self._size = min(self._size + 1, self._cap)

    def _recency_weights(self) -> np.ndarray:
        """
        Compute exponential recency weights for active slots [0 .. size−1] (§8.3.3).

        w_i = exp(−ln2 / half_life_weeks × age_weeks_i)
        age_weeks_i = (t_max − t_idx_i) / trading_days_per_week
        """
        t_arr     = self._t_idx[:self._size].astype(np.float64)
        t_max     = t_arr.max()
        age_weeks = (t_max - t_arr) / self._days_pw
        lam       = np.log(2.0) / self._half_life
        return np.exp(-lam * age_weeks)

    def _flush_stage_oldest(self, force: bool) -> None:
        """Compute n-step return for stage[0] and write to buffer if eligible."""
        if not self._stage:
            return

        t0 = self._stage[0]

        # §8.6 / §8.2: drop last n_step−1 warmup transitions
        if t0["is_warmup"] and t0["step_idx"] >= self._wu_steps - (self._n - 1):
            self._stage.popleft()
            return

        n = min(self._n, len(self._stage))

        R_n   = 0.0
        g_acc = 1.0
        done_n = False
        last   = 0

        for k in range(n):
            step   = self._stage[k]
            R_n   += g_acc * step["reward"]
            g_acc *= self._gamma
            last   = k
            if step["done"]:
                done_n = True
                break

        t_idx_next_n = self._stage[last]["t_idx_next"]

        self._write(
            t_idx       = t0["t_idx"],
            t_idx_next  = t_idx_next_n,
            mask_t      = t0["mask_t"],
            w_pre       = t0["w_pre"],
            w_exec      = t0["w_exec"],
            R_n         = R_n,
            done_n      = done_n,
            warmup_flag = t0["is_warmup"],
        )
        self._stage.popleft()


# ---------------------------------------------------------------------------
# Config-driven constructor
# ---------------------------------------------------------------------------

def from_config(replay_cfg: dict, arch_cfg: dict, sac_cfg: dict) -> ReplayBuffer:
    """Construct ReplayBuffer from config sections of master_config.yaml."""
    return ReplayBuffer(
        capacity                   = replay_cfg.get("capacity",                   800),
        K_max                      = arch_cfg.get("K_max",                        110),
        n_step                     = sac_cfg.get("n_step",                          4),
        gamma                      = sac_cfg.get("gamma",                       0.975),
        half_life_weeks            = replay_cfg.get("recency_half_life_years", 3) * 52,
        warmup_steps               = replay_cfg.get("warmup_steps",               52),
        warmup_exclusion_threshold = replay_cfg.get("warmup_exclusion_threshold", 128),
        aug_reward_noise_factor    = replay_cfg.get("aug_reward_noise_factor",  0.01),
        aug_obs_noise_factor       = replay_cfg.get("aug_obs_noise_factor",    0.015),
    )
